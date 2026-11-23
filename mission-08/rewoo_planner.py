"""
Mission 08, Part 1: Systems Architect
ReWOO decoupled DAG planner (Planner -> Worker -> Solver) and token-cost profiler.
Standard library only (Python 3.11+).

Public interface:
    parse_blueprint(text)                    -> [(var, plan_desc, tool, raw_arg), ...]
    to_plan_steps(tuples)                    -> [PlanStep, ...]  (validated, deps computed)
    build_dependency_levels(steps)           -> [["#E1", "#E2"], ["#E3"], ...]
    execute_dag(steps, tools, fallback_replan_fn) -> ExecutionResult
    compare_token_cost(k, base, step, obs)   -> dict
Run:  python rewoo_planner.py            (demo + cost table)
      python rewoo_planner.py --selftest (checks)
"""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

ERR_PREFIX = "ERR_"
_VAR = re.compile(r"#E(\d+)(?!\d)")                      # whole variables only: #E1 never matches #E10
_STEP = re.compile(r"^\s*(#E\d+)\s*=\s*([A-Za-z_]\w*)\s*\[(.*)\]\s*$")  # greedy: nested brackets ok
_PLAN = re.compile(r"^\s*Plan\s*:\s*(.*)$", re.IGNORECASE)


class BlueprintError(ValueError):
    """Malformed blueprint (message includes line numbers where possible)."""


@dataclass
class PlanStep:
    var: str
    plan: str
    tool: str
    raw_arg: str
    deps: list[str] = field(default_factory=list)

    def as_tuple(self):
        return (self.var, self.plan, self.tool, self.raw_arg)


@dataclass
class StepTrace:
    var: str
    level: int
    tool: str
    arg: str
    output: str
    status: str          # ok | error | recovered | replanned | skipped
    duration_ms: float


@dataclass
class ExecutionResult:
    outputs: dict[str, str]
    trace: list[StepTrace]
    levels: list[list[str]]
    steps: list[PlanStep]


# --------------------------------------------------------------------------- #
# 1. Planner output parsing
# --------------------------------------------------------------------------- #
def parse_blueprint(blueprint_text: str) -> list[tuple[str, str, str, str]]:
    """Parse 'Plan: ...' + '#E_i = Tool[arg]' lines into (var, plan, tool, raw_arg) tuples.
    A Plan may span several lines; it attaches to the next #E line."""
    tuples: list[tuple[str, str, str, str]] = []
    plan_lines: list[str] = []
    for no, line in enumerate(blueprint_text.splitlines(), 1):
        if not line.strip():
            continue
        m_plan = _PLAN.match(line)
        if m_plan:
            if plan_lines:
                raise BlueprintError(f"line {no}: 'Plan:' without a following #E step")
            plan_lines = [m_plan.group(1).strip()]
            continue
        if line.lstrip().startswith("#E"):
            m = _STEP.match(line)
            if not m:
                raise BlueprintError(f"line {no}: malformed step {line.strip()!r} "
                                     f"(expected '#E<n> = Tool[arg]')")
            if not plan_lines:
                raise BlueprintError(f"line {no}: step {m.group(1)} has no preceding 'Plan:'")
            tuples.append((m.group(1), " ".join(plan_lines), m.group(2), m.group(3)))
            plan_lines = []
        elif plan_lines:                       # continuation of a multi-line Plan
            plan_lines.append(line.strip())
        else:
            raise BlueprintError(f"line {no}: unexpected text {line.strip()!r}")
    if plan_lines:
        raise BlueprintError("trailing 'Plan:' without a #E step")
    if not tuples:
        raise BlueprintError("blueprint contains no steps")
    return tuples


def to_plan_steps(tuples: list[tuple[str, str, str, str]]) -> list[PlanStep]:
    """Build PlanStep nodes and validate: duplicates, self-references, unknown variables."""
    steps = [PlanStep(*t) for t in tuples]
    known = set()
    for s in steps:
        if s.var in known:
            raise BlueprintError(f"duplicate variable {s.var}")
        known.add(s.var)
    for s in steps:
        deps = []
        for m in _VAR.finditer(s.raw_arg):
            d = f"#E{m.group(1)}"
            if d == s.var:
                raise BlueprintError(f"{s.var} references itself")
            if d not in known:
                raise BlueprintError(f"{s.var} references unknown variable {d}")
            if d not in deps:
                deps.append(d)
        s.deps = deps
    return steps


# --------------------------------------------------------------------------- #
# 2. Topological levels (Kahn by frontier)
# --------------------------------------------------------------------------- #
def build_dependency_levels(steps) -> list[list[str]]:
    """Group variables into levels; every step in a level depends only on earlier levels,
    so a level can run in parallel. Accepts PlanStep list or raw tuples."""
    if steps and not isinstance(steps[0], PlanStep):
        steps = to_plan_steps(steps)
    deps = {s.var: set(s.deps) for s in steps}
    order = [s.var for s in steps]
    levels, done = [], set()
    while len(done) < len(deps):
        frontier = [v for v in order if v not in done and deps[v] <= done]
        if not frontier:
            cyc = sorted(v for v in deps if v not in done)
            raise BlueprintError(f"dependency cycle among {cyc}")
        levels.append(frontier)
        done.update(frontier)
    return levels


# --------------------------------------------------------------------------- #
# 3. Worker: substitution + parallel execution + fault handling
# --------------------------------------------------------------------------- #
def substitute(raw_arg: str, outputs: dict[str, str]) -> str:
    return _VAR.sub(lambda m: outputs.get(f"#E{m.group(1)}", m.group(0)), raw_arg)


def execute_dag(steps, tools: dict[str, Callable[[str], str]] | None = None,
                fallback_replan_fn: Callable[[PlanStep, str, str], str | None] | None = None,
                max_workers: int = 8) -> ExecutionResult:
    """Run the DAG level by level (parallel inside a level).

    A tool output starting with 'ERR_' marks the step failed. Then:
      * no fallback_replan_fn -> dependants are skipped with 'ERR_UPSTREAM'
      * fallback_replan_fn(step, resolved_arg, error) may return
          - a plain string          -> used as the replacement value (status 'recovered')
          - blueprint text (#E = ..) -> spliced into the DAG; the failed variable resolves to
                                       the fragment's last step (status 'replanned')
          - None                    -> step stays failed
    """
    tools = tools if tools is not None else DEFAULT_TOOLS
    if steps and not isinstance(steps[0], PlanStep):
        steps = to_plan_steps(steps)
    steps = list(steps)
    build_dependency_levels(steps)                       # validates acyclicity up front

    by_var = {s.var: s for s in steps}
    outputs: dict[str, str] = {}
    failed: set[str] = set()
    alias: dict[str, str] = {}                           # failed var -> replacement last var
    trace: list[StepTrace] = []
    levels_seen: list[list[str]] = []
    pending = [s.var for s in steps]

    def settled(v: str) -> bool:
        return v in outputs or v in failed

    def resolve_alias(v: str) -> str:
        while v in alias:
            v = alias[v]
        return v

    def run_one(s: PlanStep, arg: str):
        t0 = time.perf_counter()
        fn = tools.get(s.tool)
        try:
            out = fn(arg) if fn else f"{ERR_PREFIX}UNKNOWN_TOOL:{s.tool}"
        except Exception as e:                           # tool crash == runtime fault
            out = f"{ERR_PREFIX}{type(e).__name__}:{e}"
        return out, (time.perf_counter() - t0) * 1000

    level_no = 0
    while pending:
        # propagate aliases whose target finished
        for v, tgt in list(alias.items()):
            t = resolve_alias(tgt)
            if t in outputs and v not in outputs:
                outputs[v] = outputs[t]
                failed.discard(v)
        ready, blocked = [], []
        for v in pending:
            s = by_var[v]
            targets = [resolve_alias(d) for d in s.deps]
            if any(t in failed and t not in alias for t in targets):
                blocked.append(v)
            elif all(t in outputs for t in targets):
                ready.append(v)
        for v in blocked:                                # upstream failure, no recovery
            failed.add(v)
            outputs_err = f"{ERR_PREFIX}UPSTREAM"
            trace.append(StepTrace(v, level_no, by_var[v].tool, by_var[v].raw_arg,
                                   outputs_err, "skipped", 0.0))
        pending = [v for v in pending if v not in blocked]
        if not ready:
            if pending and all(v in alias for v in pending):   # waiting only on spliced steps
                pending = [v for v in pending if v not in outputs]
            if not ready:
                break
        levels_seen.append(ready)
        args = {v: substitute(by_var[v].raw_arg,
                              {d: outputs[resolve_alias(d)] for d in by_var[v].deps}) for v in ready}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda v: run_one(by_var[v], args[v]), ready))
        for v, (out, ms) in zip(ready, results):
            s = by_var[v]
            if not out.startswith(ERR_PREFIX):
                outputs[v] = out
                trace.append(StepTrace(v, level_no, s.tool, args[v], out, "ok", ms))
                pending.remove(v)
                continue
            fb = fallback_replan_fn(s, args[v], out) if fallback_replan_fn else None
            status = "error"
            if fb is None:
                failed.add(v)
                pending.remove(v)
            elif "#E" in fb and "=" in fb and "[" in fb:   # blueprint fragment -> splice
                new = to_plan_steps(parse_blueprint(fb))
                clash = [n.var for n in new if n.var in by_var]
                if clash:
                    raise BlueprintError(f"re-plan reuses existing variables {clash}")
                for n in new:
                    by_var[n.var] = n
                    pending.append(n.var)
                alias[v] = new[-1].var
                pending.remove(v)
                status = "replanned"
            else:
                outputs[v] = fb
                pending.remove(v)
                status = "recovered"
            trace.append(StepTrace(v, level_no, s.tool, args[v], fb if fb else out, status, ms))
        level_no += 1

    for v, tgt in alias.items():                         # final alias settlement
        t = resolve_alias(tgt)
        if t in outputs:
            outputs[v] = outputs[t]
    final_steps = list(by_var.values())
    return ExecutionResult(outputs, trace, levels_seen, final_steps)


# --------------------------------------------------------------------------- #
# 4. Solver stage
# --------------------------------------------------------------------------- #
def build_solver_prompt(question: str, result: ExecutionResult) -> str:
    lines = [f"Question: {question}", "", "Plan and evidence:"]
    for s in result.steps:
        lines.append(f"Plan: {s.plan}")
        lines.append(f"{s.var} = {s.tool}[{s.raw_arg}]  ->  {result.outputs.get(s.var, 'ERR_UNRESOLVED')}")
    lines.append("\nUsing only the evidence above, give the final answer.")
    return "\n".join(lines)


def solve(question: str, result: ExecutionResult,
          solver_fn: Callable[[str], str] | None = None) -> str:
    prompt = build_solver_prompt(question, result)
    return solver_fn(prompt) if solver_fn else prompt


# --------------------------------------------------------------------------- #
# 5. Token-cost profiler
# --------------------------------------------------------------------------- #
def compare_token_cost(k_steps: int, base_tokens: int = 1200,
                       step_tokens: int = 150, obs_tokens: int = 250) -> dict:
    """Cumulative input tokens for a K-step task.
    ReAct : K*L0 + K(K-1)/2 * (Lstep + Lobs)      -> O(K^2)
    ReWOO : 2*L0 + K*(Lstep + Lobs)               -> O(K)   (planner + solver prompt)"""
    if k_steps < 1:
        raise ValueError("k_steps must be >= 1")
    per_step = step_tokens + obs_tokens
    react = k_steps * base_tokens + (k_steps * (k_steps - 1) // 2) * per_step
    rewoo = 2 * base_tokens + k_steps * per_step
    return {
        "k_steps": k_steps,
        "react_tokens": react,
        "rewoo_tokens": rewoo,
        "savings_pct": round((react - rewoo) / react * 100, 2),
        "speedup_ratio": round(react / rewoo, 4),
    }


# --------------------------------------------------------------------------- #
# Mock tools + demo
# --------------------------------------------------------------------------- #
def _sleepy(fn, delay=0.05):
    def wrapped(arg):
        time.sleep(delay)
        return fn(arg)
    return wrapped


DEFAULT_TOOLS: dict[str, Callable[[str], str]] = {
    "Telemetry": _sleepy(lambda a: f"flux({a.strip()})=7.3"),
    "Sensor": _sleepy(lambda a: f"reading[{a.strip()}]=ok"),
    "Analyze": _sleepy(lambda a: f"analysis<{a.strip()}>"),
    "Echo": lambda a: a,
    "Fail": lambda a: f"{ERR_PREFIX}SENSOR_BITFLIP",
}

DEMO_BLUEPRINT = """\
Plan: Read X-ray flux from the corona sensor.
#E1 = Telemetry[corona]
Plan: Read bit-flip rate of the firmware
#E2 = Sensor[firmware]
Plan: Read nebula density.
#E3 = Sensor[veil nebula]
Plan: Correlate flux and bit-flips.
#E4 = Analyze[#E1 vs #E2]
Plan: Combine with density.
#E5 = Analyze[#E4 + #E3]
Plan: Final integrity estimate.
#E6 = Analyze[#E5]
"""


def _demo():
    steps = to_plan_steps(parse_blueprint(DEMO_BLUEPRINT))
    levels = build_dependency_levels(steps)
    print("Levels:", levels)
    t0 = time.perf_counter()
    res = execute_dag(steps)
    par = (time.perf_counter() - t0) * 1000
    for t in res.trace:
        print(f"  L{t.level} {t.var} {t.tool}[{t.arg}] -> {t.output} ({t.status}, {t.duration_ms:.0f} ms)")
    print(f"Parallel wall time: {par:.0f} ms vs ~{50*len(steps)} ms serial\n")
    print("Token cost (L0=1200, step=150, obs=250)")
    print(f"{'K':>3} {'ReAct':>8} {'ReWOO':>8} {'savings':>8} {'speedup':>8}")
    for k in (1, 2, 3, 5, 8, 16):
        c = compare_token_cost(k)
        print(f"{k:>3} {c['react_tokens']:>8,} {c['rewoo_tokens']:>8,} "
              f"{c['savings_pct']:>7.1f}% {c['speedup_ratio']:>8.2f}")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    n = 0

    def check(cond, name):
        nonlocal n
        n += 1
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            raise SystemExit(1)

    def raises(fn):
        try:
            fn()
        except BlueprintError:
            return True
        return False

    t = parse_blueprint(DEMO_BLUEPRINT)
    check(len(t) == 6 and t[0] == ("#E1", "Read X-ray flux from the corona sensor.", "Telemetry", "corona"),
          "parse tuples")
    check(parse_blueprint("Plan: a\nb\n#E1 = T[x[1]]")[0][3] == "x[1]", "nested brackets + multi-line plan")
    check(raises(lambda: parse_blueprint("Plan: a\n#E1 Tool[x]")), "malformed step")
    check(raises(lambda: to_plan_steps(parse_blueprint("Plan: a\n#E1 = T[#E1]"))), "self reference")
    check(raises(lambda: to_plan_steps(parse_blueprint("Plan: a\n#E1 = T[#E9]"))), "unknown variable")
    check(raises(lambda: to_plan_steps(parse_blueprint("Plan: a\n#E1 = T[x]\nPlan: b\n#E1 = T[y]"))),
          "duplicate variable")
    steps = to_plan_steps(t)
    check(build_dependency_levels(steps) == [["#E1", "#E2", "#E3"], ["#E4"], ["#E5"], ["#E6"]]
          or build_dependency_levels(steps)[0] == ["#E1", "#E2", "#E3"], "levels")
    cyc = to_plan_steps(parse_blueprint("Plan: a\n#E1 = T[#E2]\nPlan: b\n#E2 = T[#E1]"))
    check(raises(lambda: build_dependency_levels(cyc)), "cycle detection")
    fw = to_plan_steps(parse_blueprint("Plan: a\n#E2 = Echo[#E1]\nPlan: b\n#E1 = Echo[hi]"))
    check(build_dependency_levels(fw) == [["#E1"], ["#E2"]], "forward references")
    tools = {"Echo": lambda a: a}
    ten = to_plan_steps(parse_blueprint(
        "Plan: a\n#E1 = Echo[A]\nPlan: b\n#E10 = Echo[B]\nPlan: c\n#E3 = Echo[#E1|#E10]"))
    check(execute_dag(ten, tools).outputs["#E3"] == "A|B", "#E1 does not touch #E10")
    r = execute_dag(steps)
    check(all(not v.startswith("ERR_") for v in r.outputs.values()) and len(r.outputs) == 6, "full DAG run")
    check(r.outputs["#E4"] == "analysis<flux(corona)=7.3 vs reading[firmware]=ok>", "substitution")
    # faults
    ft = {"Echo": lambda a: a, "Fail": lambda a: "ERR_X"}
    fs = to_plan_steps(parse_blueprint("Plan: a\n#E1 = Fail[x]\nPlan: b\n#E2 = Echo[#E1]\nPlan: c\n#E3 = Echo[ok]"))
    r = execute_dag(fs, ft)
    check(r.outputs.get("#E3") == "ok" and "#E2" not in r.outputs
          and any(x.var == "#E2" and x.output == "ERR_UPSTREAM" for x in r.trace), "ERR_UPSTREAM skip")
    r = execute_dag(fs, ft, lambda s, a, e: "fixed")
    check(r.outputs["#E2"] == "fixed" and r.trace[0].status == "recovered", "fallback value")
    frag = "Plan: retry\n#E10 = Echo[alt]\nPlan: polish\n#E11 = Echo[#E10!]"
    r = execute_dag(fs, ft, lambda s, a, e: frag)
    check(r.outputs["#E1"] == "alt!" and r.outputs["#E2"] == "alt!", "dynamic re-planning splice")
    # tokens
    c = compare_token_cost(5, 1200, 150, 250)
    check((c["react_tokens"], c["rewoo_tokens"]) == (10000, 4400), "token formulas K=5")
    check(compare_token_cost(1)["react_tokens"] == 1200 and compare_token_cost(1)["rewoo_tokens"] == 2800,
          "K=1 ReWOO costlier")
    check(compare_token_cost(3)["savings_pct"] == 25.0, "K=3 crossover savings")
    d2 = [compare_token_cost(k)["react_tokens"] for k in range(1, 6)]
    check({d2[i + 2] - 2 * d2[i + 1] + d2[i] for i in range(3)} == {400}, "ReAct quadratic")
    print(f"\n{n}/{n} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else (_demo() or 0))
