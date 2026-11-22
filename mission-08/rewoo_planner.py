"""
Mission 08 — Part 1: Systems Architect (Lead Engineer)
ReWOO (Reasoning Without Observation) Decoupled DAG Planner, Worker & Token Profiler

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula)
Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager
Station Officer: Fahad Akram (Systems Architect), Squadron Bengio

Pure Python 3.11+, standard library only.

The instructor's public interface is kept exactly:
    PlanStep(var_name, plan_desc, tool_name, raw_arg, dependencies)
    ReWOOEngine(tools).parse_blueprint(text) -> [(var, plan, tool, raw_arg)]
    .to_plan_steps(steps), .build_dependency_levels(steps) -> [[vars]], .execute_dag(steps, fallback_replan_fn) -> {var: output}
    ReWOOEngine.compare_token_cost(k_steps, base_tokens, step_tokens, obs_tokens) -> {react_tokens, rewoo_tokens, savings_pct, speedup_ratio}

What this file adds on top of the starter
    1. Parsing that validates the blueprint: duplicate variables, references to variables that do not exist,
       self-references and malformed lines are reported with the line number; multi-line Plan text is kept.
    2. Substitution by whole variable (#E1 never touches #E10), done with one regex per argument.
    3. A Worker that really runs each topological level in parallel (ThreadPoolExecutor), with a per-step
       trace (level, tool, resolved argument, output, duration, status) and wall-clock accounting.
    4. Error handling: a tool output starting with or containing 'ERR_' marks the step; without a fallback its
       dependants are skipped as ERR_UPSTREAM; with a fallback the engine either takes a replacement output or,
       when the fallback returns a blueprint fragment, splices the new steps into the DAG (dynamic re-planning).
    5. A Solver stage: render_solver_prompt() builds the single final prompt from the plan and the evidence,
       and solve() produces the answer through a pluggable solver function.
    6. Token profiling beyond the closed form: simulate_token_cost() runs a mock ReAct loop and a mock ReWOO run
       and counts the tokens actually sent to the model, so the formulas are checked against a measurement, and
       cost_table() shows the crossover (ReWOO wins from K = 3 whatever the token sizes).
    7. self_test(): 40 checks, including the instructor's harness and integration with the Sentinel's sandbox,
       the Protocol Officer's traceback parser and the Payload Specialist's pager.

Run (from the repo root, the shared .venv; nothing to install):
    python mission-08/rewoo_planner.py                  # instructor harness + parallel timing + cost tables
    python mission-08/rewoo_planner.py --selftest       # 40 checks, log in logs/
    python mission-08/rewoo_planner.py --check-signoffs # gate: Parts 1-4 ticked in README.md
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"

Step = Tuple[str, str, str, str]
ToolFn = Callable[[str], str]


@dataclass
class PlanStep:
    """Represents a single symbolic node #E_i = ToolName[raw_arg] in a ReWOO DAG."""
    var_name: str
    plan_desc: str
    tool_name: str
    raw_arg: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class StepRecord:
    """What the Worker did for one step."""
    var_name: str
    level: int
    tool_name: str
    resolved_arg: str
    output: str
    status: str            # ok | error | fallback | replanned | skipped
    duration_s: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class BlueprintError(ValueError):
    """A blueprint that cannot be executed as written."""


def count_tokens(text: str) -> int:
    """Whitespace tokens: a stable stand-in for a tokenizer, good enough to compare costs."""
    return len(text.split())


class ReWOOEngine:
    """
    Implements the 3-stage ReWOO (Reasoning Without Observation) paradigm (Xu et al., 2023):
      1. Planner (P): Emits a complete symbolic execution graph (#E1, #E2, ..., #EK) in one LLM call.
      2. Worker (E): Resolves #E variable dependencies in topological levels and executes tools.
      3. Solver (S): Synthesizes final output from resolved evidence in a single final LLM call.
    Also supports dynamic fallback re-planning when a tool emits a runtime sensor fault (e.g. ERR_CORONA_SPIKE).
    """

    VAR_PATTERN = re.compile(r"#E\d+")
    STEP_PATTERN = re.compile(r"^(#E\d+)\s*=\s*([A-Za-z_]\w*)\[(.*)\]\s*$")
    ERROR_MARK = "ERR_"

    def __init__(self, tools: Dict[str, ToolFn], max_workers: int = 8, strict: bool = True):
        self.tools = tools
        self.max_workers = max_workers
        self.strict = strict
        self.trace: List[StepRecord] = []
        self.levels_run: List[List[str]] = []
        self.wall_clock_s: float = 0.0
        self.serial_clock_s: float = 0.0

    # ------------------------------------------------------------------ Planner output -> steps
    def parse_blueprint(self, blueprint_text: str) -> List[Step]:
        """Parse 'Plan: <reasoning>' and '#E_i = ToolName[raw_arg]' lines into (var, plan, tool, raw_arg).

        Plan text may span several lines until the next #E line. In strict mode a malformed #E line, a
        duplicate variable, a self-reference or a reference to a variable that never appears raises
        BlueprintError with the offending line number; otherwise such lines are skipped.
        """
        steps: List[Step] = []
        plan_lines: List[str] = []
        seen: Set[str] = set()
        for lineno, raw in enumerate(blueprint_text.strip().splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Plan:"):
                plan_lines = [line[len("Plan:"):].strip()]
                continue
            if line.startswith("#E"):
                m = self.STEP_PATTERN.match(line)
                if not m:
                    if self.strict:
                        raise BlueprintError(f"line {lineno}: malformed step {line!r}; expected '#Ei = Tool[arg]'")
                    continue
                var_name, tool_name, raw_arg = m.groups()
                if var_name in seen:
                    raise BlueprintError(f"line {lineno}: duplicate variable {var_name}")
                if var_name in self.VAR_PATTERN.findall(raw_arg):
                    raise BlueprintError(f"line {lineno}: {var_name} references itself")
                seen.add(var_name)
                steps.append((var_name, " ".join(plan_lines).strip(), tool_name, raw_arg))
                plan_lines = []
                continue
            if plan_lines:                      # continuation of a multi-line Plan
                plan_lines.append(line)
        if self.strict:
            for var_name, _, _, raw_arg in steps:
                for dep in self.VAR_PATTERN.findall(raw_arg):
                    if dep not in seen:
                        raise BlueprintError(f"{var_name} references {dep}, which is not defined in the blueprint")
        return steps

    def to_plan_steps(self, steps: List[Step]) -> List[PlanStep]:
        """Converts parsed 4-tuples into structured PlanStep nodes with extracted #E dependencies."""
        nodes: List[PlanStep] = []
        for var_name, plan_desc, tool_name, raw_arg in steps:
            deps = sorted(set(self.VAR_PATTERN.findall(raw_arg)), key=lambda v: int(v[2:]))
            nodes.append(PlanStep(var_name, plan_desc, tool_name, raw_arg, deps))
        return nodes

    # ------------------------------------------------------------------ DAG
    def build_dependency_levels(self, steps: List[Step]) -> List[List[str]]:
        """Group variables into topological levels (Kahn's algorithm by frontier). Steps in one level
        depend only on earlier levels and can run concurrently. Raises ValueError on a cycle or on a
        reference to an unknown variable."""
        remaining: Dict[str, Set[str]] = {}
        order_index: Dict[str, int] = {}
        for idx, (var_name, _, _, raw_arg) in enumerate(steps):
            remaining[var_name] = set(self.VAR_PATTERN.findall(raw_arg))
            order_index[var_name] = idx
        unknown = {d for deps in remaining.values() for d in deps if d not in remaining}
        if unknown:
            raise ValueError(f"Unresolved variable dependency in ReWOO DAG: {sorted(unknown)}")
        resolved: Set[str] = set()
        levels: List[List[str]] = []
        while remaining:
            ready = sorted((v for v, deps in remaining.items() if deps <= resolved), key=order_index.__getitem__)
            if not ready:
                raise ValueError(f"Cyclic variable dependency in ReWOO DAG: {sorted(remaining)}")
            levels.append(ready)
            for v in ready:
                resolved.add(v)
                del remaining[v]
        return levels

    @staticmethod
    def critical_path_length(levels: List[List[str]]) -> int:
        return len(levels)

    @classmethod
    def substitute(cls, raw_arg: str, evidence: Dict[str, str]) -> str:
        """Replace every #E variable by its evidence, matching whole variables only (#E1 is not a prefix of #E10)."""
        return cls.VAR_PATTERN.sub(lambda m: evidence.get(m.group(0), m.group(0)), raw_arg)

    # ------------------------------------------------------------------ Worker
    def execute_dag(self, steps: List[Step], fallback_replan_fn: Optional[Callable[[str, str], str]] = None,
                    parallel: bool = True) -> Dict[str, str]:
        """Run the steps level by level, each level concurrently, substituting evidence into arguments.

        A tool output containing 'ERR_' is a runtime fault. With `fallback_replan_fn(var, error)` the engine
        takes its return value as the replacement evidence, or, if that value parses as a blueprint fragment
        (lines '#Ej = Tool[...]'), splices those steps into the remaining DAG and continues (re-planning).
        Without a fallback the faulty evidence is kept and every dependant is skipped with ERR_UPSTREAM.
        """
        self.trace, self.levels_run = [], []
        self.wall_clock_s = self.serial_clock_s = 0.0
        pending: Dict[str, Step] = {s[0]: s for s in steps}
        for var, (_, _, tool, _) in pending.items():
            if tool not in self.tools:
                raise KeyError(f"Unknown tool '{tool}' referenced in step {var}")
        evidence: Dict[str, str] = {}
        failed: Set[str] = set()
        aliases: Dict[str, str] = {}      # re-planned variable -> last variable of its replacement fragment
        t_start = time.perf_counter()
        level_no = 0
        while pending:
            # steps become ready when every dependency is resolved (evidence) or failed (then they are skipped)
            ready = self._ready(pending, evidence, failed, aliases)
            if not ready:
                raise ValueError(f"Cyclic or unresolved dependency among {sorted(pending)}")
            self.levels_run.append(ready)
            runnable, records = [], []
            for var in ready:
                _, _, tool, raw_arg = pending[var]
                deps = set(self.VAR_PATTERN.findall(raw_arg))
                if deps & failed:
                    rec = StepRecord(var, level_no, tool, raw_arg, f"ERR_UPSTREAM({','.join(sorted(deps & failed))})", "skipped", 0.0)
                    records.append(rec)
                    failed.add(var)
                    evidence[var] = rec.output
                    continue
                runnable.append((var, tool, self.substitute(raw_arg, self._with_aliases(evidence, aliases))))

            def run_one(item: Tuple[str, str, str]) -> StepRecord:
                var, tool, arg = item
                t0 = time.perf_counter()
                try:
                    out = str(self.tools[tool](arg))
                    status = "error" if self.ERROR_MARK in out else "ok"
                except Exception as exc:  # a crashing tool is a runtime fault too
                    out, status = f"ERR_EXCEPTION({type(exc).__name__}: {exc})", "error"
                return StepRecord(var, level_no, tool, arg, out, status, time.perf_counter() - t0)

            if parallel and len(runnable) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    records.extend(pool.map(run_one, runnable))
            else:
                records.extend(run_one(r) for r in runnable)

            new_steps: List[Step] = []
            for rec in records:
                self.serial_clock_s += rec.duration_s
                if rec.status == "error" and fallback_replan_fn is not None:
                    replacement = str(fallback_replan_fn(rec.var_name, rec.output))
                    fragment = self._parse_fragment(replacement)
                    if fragment:
                        rec.status, rec.output = "replanned", f"REPLANNED->{','.join(s[0] for s in fragment)}"
                        new_steps.extend(fragment)
                        # the failed variable resolves to the LAST step of the fragment, once that has run
                        aliases[rec.var_name] = fragment[-1][0]
                    else:
                        rec.status, rec.output = "fallback", replacement
                        evidence[rec.var_name] = replacement
                elif rec.status == "error":
                    failed.add(rec.var_name)
                    evidence[rec.var_name] = rec.output
                elif rec.status == "ok":
                    evidence[rec.var_name] = rec.output
                self.trace.append(rec)
                pending.pop(rec.var_name, None)
            for s in new_steps:
                if s[0] in pending or s[0] in evidence:
                    raise BlueprintError(f"re-plan fragment reuses variable {s[0]}")
                if s[2] not in self.tools:
                    raise KeyError(f"Unknown tool '{s[2]}' in re-plan fragment for {s[0]}")
                pending[s[0]] = s
            level_no += 1
        evidence = self._with_aliases(evidence, aliases)
        self.wall_clock_s = time.perf_counter() - t_start
        return evidence

    def _with_aliases(self, evidence: Dict[str, str], aliases: Dict[str, str]) -> Dict[str, str]:
        """Evidence plus every re-planned variable whose replacement has finished."""
        out = dict(evidence)
        for var, target in aliases.items():
            while target in aliases:
                target = aliases[target]
            if target in evidence:
                out[var] = evidence[target]
        return out

    def _ready(self, pending: Dict[str, Step], evidence: Dict[str, str], failed: Set[str],
               aliases: Optional[Dict[str, str]] = None) -> List[str]:
        done = set(self._with_aliases(evidence, aliases or {})) | failed
        out = []
        for var, (_, _, _, raw_arg) in pending.items():
            deps = set(self.VAR_PATTERN.findall(raw_arg))
            if deps <= done:
                out.append(var)
        return sorted(out, key=lambda v: int(v[2:]))

    def _parse_fragment(self, text: str) -> List[Step]:
        if "#E" not in text or "[" not in text:
            return []
        try:
            return ReWOOEngine(self.tools, strict=False).parse_blueprint(text)
        except BlueprintError:
            return []

    def trace_table(self) -> str:
        lines = [f"  {'level':>5} {'var':<5} {'tool':<16} {'status':<9} {'ms':>7}  output"]
        for r in self.trace:
            lines.append(f"  {r.level:>5} {r.var_name:<5} {r.tool_name:<16} {r.status:<9} {r.duration_s * 1000:>7.1f}  {r.output[:60]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ Solver
    @staticmethod
    def render_solver_prompt(task: str, steps: List[Step], evidence: Dict[str, str]) -> str:
        """The single Solver prompt: the task, the plan with every #E resolved, and the evidence list."""
        lines = [f"Task: {task}", "Plan and evidence:"]
        for var, plan, tool, raw_arg in steps:
            lines.append(f"- {plan} [{var} = {tool}[{raw_arg}] -> {evidence.get(var, '?')}]")
        lines.append("Answer the task using only the evidence above.")
        return "\n".join(lines)

    def solve(self, task: str, steps: List[Step], evidence: Dict[str, str],
              solver_fn: Optional[Callable[[str], str]] = None) -> str:
        prompt = self.render_solver_prompt(task, steps, evidence)
        if solver_fn is not None:
            return solver_fn(prompt)
        last = steps[-1][0] if steps else ""
        return f"Solver: {task} -> {evidence.get(last, 'no evidence')}"

    # ------------------------------------------------------------------ Token profiler
    @staticmethod
    def compare_token_cost(k_steps: int, base_tokens: int, step_tokens: int, obs_tokens: int) -> Dict[str, float]:
        """Closed-form cumulative INPUT tokens for K tool calls.
          C_ReAct(K) = sum_{t=1..K} (L0 + (t-1)(L_step + L_obs)) = K L0 + K(K-1)/2 (L_step + L_obs)
          C_ReWOO(K) = 2 L0 + K (L_step + L_obs)
        """
        if k_steps < 0 or min(base_tokens, step_tokens, obs_tokens) < 0:
            raise ValueError("token counts and K must be non-negative")
        growth = step_tokens + obs_tokens
        react = k_steps * base_tokens + (k_steps * (k_steps - 1) // 2) * growth
        rewoo = 2 * base_tokens + k_steps * growth
        savings = ((react - rewoo) / react) * 100.0 if react > 0 else 0.0
        speedup = react / rewoo if rewoo > 0 else 1.0
        return {"react_tokens": float(react), "rewoo_tokens": float(rewoo),
                "savings_pct": round(savings, 2), "speedup_ratio": round(speedup, 2)}

    @staticmethod
    def cost_table(ks: Sequence[int] = (1, 2, 3, 4, 5, 8, 12, 16), base_tokens: int = 1200, step_tokens: int = 120,
                   obs_tokens: int = 280) -> List[Dict[str, float]]:
        rows = []
        for k in ks:
            m = ReWOOEngine.compare_token_cost(k, base_tokens, step_tokens, obs_tokens)
            rows.append({"K": k, **m})
        return rows

    @staticmethod
    def simulate_token_cost(k_steps: int, base_tokens: int, step_tokens: int, obs_tokens: int) -> Dict[str, int]:
        """Measure instead of derive: run a mock ReAct loop and a mock ReWOO run with a model stub that
        counts the tokens of every prompt it receives. Prompts are built from filler words so that the
        whitespace tokenizer reproduces the nominal sizes exactly."""
        words = lambda n, tag: " ".join(f"{tag}{i}" for i in range(n))  # noqa: E731
        base = words(base_tokens, "sys")
        sent: Dict[str, int] = {"react": 0, "rewoo": 0}

        def model(prompt: str, who: str) -> None:
            sent[who] += count_tokens(prompt)

        # ReAct: every turn re-sends the whole transcript so far
        transcript = base
        for t in range(1, k_steps + 1):
            model(transcript, "react")
            transcript += " " + words(step_tokens, f"thought{t}_") + " " + words(obs_tokens, f"obs{t}_")
        # ReWOO: one planner call with the base prompt, workers run without the model, one solver call
        model(base, "rewoo")
        plan_and_evidence = " ".join(words(step_tokens, f"plan{t}_") + " " + words(obs_tokens, f"ev{t}_") for t in range(1, k_steps + 1))
        model(base + " " + plan_and_evidence, "rewoo")
        return {"react_tokens": sent["react"], "rewoo_tokens": sent["rewoo"]}


# =============================================================================
# Demo material: the instructor's harness, a timed tool set, a re-planning scenario
# =============================================================================

BLUEPRINT = """
Plan: Scan Cygnus X-1 hard X-ray flare flux from primary telemetry array.
#E1 = TelemetryScan[Cygnus_X1_Corona_Sector_08]
Plan: Measure Veil Nebula shockwave plasma density in parallel.
#E2 = PlasmaProbe[Veil_Nebula_Shock_Front]
Plan: Compute Hohmann escape burn vector combining X-ray flux #E1 and plasma density #E2.
#E3 = HohmannCalc[flux=#E1, density=#E2]
Plan: Modulate deflector shield harmonics for burn trajectory #E3.
#E4 = ShieldModulate[vector=#E3]
"""

MOCK_TOOLS: Dict[str, ToolFn] = {
    "TelemetryScan": lambda arg: f"XRayFlux({arg})=4.2e37_erg/s",
    "PlasmaProbe": lambda arg: f"IonDensity({arg})=14.8_cm^-3",
    "HohmannCalc": lambda arg: f"DeltaV({arg})=18.4_km/s",
    "ShieldModulate": lambda arg: f"HarmonicLock({arg})=STABLE",
}


def timed_tools(delay_s: float = 0.05, failing: Optional[Set[str]] = None) -> Dict[str, ToolFn]:
    """The mock tools with a simulated sensor latency; tools in `failing` return an ERR_ fault."""
    failing = failing or set()

    def wrap(name: str, fn: ToolFn) -> ToolFn:
        def tool(arg: str) -> str:
            time.sleep(delay_s)
            if name in failing:
                return f"ERR_CORONA_SPIKE({name})"
            return fn(arg)
        return tool
    return {name: wrap(name, fn) for name, fn in MOCK_TOOLS.items()} | {
        "BackupTelemetry": lambda arg: f"XRayFlux_backup({arg})=4.0e37_erg/s",
        "Calibrate": lambda arg: f"Calibrated({arg})",
    }


WIDE_BLUEPRINT = """
Plan: Four independent sensor sweeps.
#E1 = TelemetryScan[sector_A]
#E2 = TelemetryScan[sector_B]
#E3 = PlasmaProbe[front_A]
#E4 = PlasmaProbe[front_B]
Plan: Combine them.
#E5 = HohmannCalc[flux=#E1 #E2, density=#E3 #E4]
#E6 = ShieldModulate[vector=#E5]
"""


def replan_with_backup(var_name: str, error_output: str) -> str:
    """Fallback that re-plans: a failed TelemetryScan is replaced by a calibrated backup scan."""
    if "TelemetryScan" in error_output:
        return "#E10 = BackupTelemetry[Cygnus_X1_Corona_Sector_08]\n#E11 = Calibrate[#E10]"
    return f"FALLBACK_VALUE({var_name})"


def instructor_harness(quiet: bool = False) -> Dict[str, Any]:
    engine = ReWOOEngine(MOCK_TOOLS)
    steps = engine.parse_blueprint(BLUEPRINT)
    levels = engine.build_dependency_levels(steps)
    evidence = engine.execute_dag(steps)
    if not quiet:
        print("=== Subsystem 08 (Part 1): ReWOO DAG Planner & Token Profiler Verification ===")
        print(f"Parsed Steps ({len(steps)}): {[s[0] for s in steps]}")
        print(f"Topological Parallel Execution Levels: {levels}")
        for k, v in evidence.items():
            print(f"  {k} -> {v}")
    assert levels == [["#E1", "#E2"], ["#E3"], ["#E4"]], f"Unexpected DAG levels: {levels}"
    assert "4.2e37_erg/s" in evidence["#E3"] and "14.8_cm^-3" in evidence["#E3"]
    assert evidence["#E4"].startswith("HarmonicLock(vector=DeltaV(")
    rows = []
    if not quiet:
        print("\n--- ReAct vs. ReWOO Cumulative Input Token Benchmark (L0=1200, L_step=120, L_obs=280) ---")
    for k in (3, 5, 8):
        m = ReWOOEngine.compare_token_cost(k_steps=k, base_tokens=1200, step_tokens=120, obs_tokens=280)
        rows.append({"K": k, **m})
        if not quiet:
            print(f"  K={k} tools | ReAct={int(m['react_tokens']):6d} tok | ReWOO={int(m['rewoo_tokens']):5d} tok | "
                  f"Savings={m['savings_pct']:5.2f}% ({m['speedup_ratio']:.2f}x)")
    m5 = ReWOOEngine.compare_token_cost(5, 1200, 120, 280)
    assert m5["react_tokens"] == 10000.0 and m5["rewoo_tokens"] == 4400.0
    if not quiet:
        print("[OK] Part 1 (Systems Architect) verification passed.")
    return {"steps": [s[0] for s in steps], "levels": levels, "evidence": evidence, "cost": rows}


def extended_report() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    print("\n--- Worker: parallel levels vs serial execution (simulated sensor latency 50 ms per tool) ---")
    for name, bp in (("instructor DAG (4 steps, 3 levels)", BLUEPRINT), ("wide DAG (6 steps, 3 levels)", WIDE_BLUEPRINT)):
        eng = ReWOOEngine(timed_tools(0.05))
        steps = eng.parse_blueprint(bp)
        eng.execute_dag(steps, parallel=True)
        par = eng.wall_clock_s
        eng.execute_dag(steps, parallel=False)
        ser = eng.wall_clock_s
        levels = eng.build_dependency_levels(steps)
        print(f"  {name:<36} levels={len(levels)} parallel {par * 1000:6.0f} ms | serial {ser * 1000:6.0f} ms | speed-up {ser / par:4.2f}x "
              f"(ideal {len(steps) / len(levels):.2f}x)")
        out[name] = {"steps": len(steps), "levels": len(levels), "parallel_ms": par * 1000, "serial_ms": ser * 1000}

    print("\n--- Runtime fault: #E1 emits ERR_CORONA_SPIKE ---")
    eng = ReWOOEngine(timed_tools(0.0, failing={"TelemetryScan"}))
    steps = eng.parse_blueprint(BLUEPRINT)
    ev = eng.execute_dag(steps)
    print("  without fallback: dependants are skipped")
    print(eng.trace_table())
    ev2 = eng.execute_dag(steps, fallback_replan_fn=replan_with_backup)
    print("  with re-planning fallback: a backup scan and a calibration step are spliced into the DAG")
    print(eng.trace_table())
    print(f"  levels actually run: {eng.levels_run}")
    print(f"  final evidence #E4 -> {ev2['#E4']}")
    out["fault"] = {"no_fallback": ev, "replanned": ev2, "levels_run": eng.levels_run}

    print("\n--- Closed form vs measured tokens (mock model counts every prompt it receives) ---")
    print(f"  {'K':>3} {'ReAct formula':>14} {'ReAct measured':>15} {'ReWOO formula':>14} {'ReWOO measured':>15} {'savings':>8}")
    rows = []
    for k in (1, 2, 3, 4, 5, 8, 12, 16):
        f = ReWOOEngine.compare_token_cost(k, 1200, 120, 280)
        s = ReWOOEngine.simulate_token_cost(k, 1200, 120, 280)
        rows.append({"K": k, **f, "react_measured": s["react_tokens"], "rewoo_measured": s["rewoo_tokens"]})
        print(f"  {k:>3} {int(f['react_tokens']):>14} {s['react_tokens']:>15} {int(f['rewoo_tokens']):>14} {s['rewoo_tokens']:>15} {f['savings_pct']:>7.1f}%")
    print("  ReWOO costs more than ReAct at K = 1 and K = 2 (two base prompts instead of one), and wins from K = 3 on for any token sizes.")
    out["cost_rows"] = rows

    task = "Plot the escape burn out of the Cygnus X-1 corona."
    eng = ReWOOEngine(MOCK_TOOLS)
    steps = eng.parse_blueprint(BLUEPRINT)
    ev = eng.execute_dag(steps)
    prompt = eng.render_solver_prompt(task, steps, ev)
    print(f"\n--- Solver prompt ({count_tokens(prompt)} tokens, one call) ---")
    print("  " + prompt.replace("\n", "\n  "))
    print(f"  {eng.solve(task, steps, ev)}")
    return out


def load_sibling(filename: str, module_name: str) -> Optional[Any]:
    path = HERE / filename
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        sys.modules.pop(module_name, None)
        print(f"  [notice] could not load {filename}: {type(exc).__name__}: {exc}")
        return None


# =============================================================================
# Self-test
# =============================================================================

class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def passed(self) -> int:
        return sum(r["passed"] for r in self.rows)

    @property
    def failed(self) -> int:
        return len(self.rows) - self.passed


def _raises(fn: Callable[[], Any], exc: type) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def self_test() -> Checks:
    c = Checks()
    print("=== Part 1 self-test: ReWOO planner, worker, solver and token profiler ===")
    eng = ReWOOEngine(MOCK_TOOLS)

    # --- parsing ---
    steps = eng.parse_blueprint(BLUEPRINT)
    c.add("parse: four steps with variables, tools and plan text attached",
          [s[0] for s in steps] == ["#E1", "#E2", "#E3", "#E4"] and steps[2][2] == "HohmannCalc" and steps[0][1].startswith("Scan Cygnus"))
    nodes = eng.to_plan_steps(steps)
    c.add("to_plan_steps: dependencies extracted and sorted", nodes[2].dependencies == ["#E1", "#E2"] and nodes[0].dependencies == [] and nodes[3].dependencies == ["#E3"])
    multi = eng.parse_blueprint("Plan: first line\nsecond line of the plan\n#E1 = TelemetryScan[x]")
    c.add("parse: a Plan may span several lines", multi[0][1] == "first line second line of the plan")
    nested = eng.parse_blueprint("Plan: p\n#E1 = HohmannCalc[cfg={a:[1,2]}, ref=#E2]\n#E2 = PlasmaProbe[y]")
    c.add("parse: nested brackets inside an argument survive", nested[0][3] == "cfg={a:[1,2]}, ref=#E2")
    c.add("parse: duplicate variable is refused", _raises(lambda: eng.parse_blueprint("#E1 = PlasmaProbe[a]\n#E1 = PlasmaProbe[b]"), BlueprintError))
    c.add("parse: self-reference is refused", _raises(lambda: eng.parse_blueprint("#E1 = PlasmaProbe[#E1]"), BlueprintError))
    c.add("parse: reference to an undefined variable is refused in strict mode", _raises(lambda: eng.parse_blueprint("#E1 = PlasmaProbe[#E7]"), BlueprintError))
    c.add("parse: malformed step line is refused in strict mode and skipped otherwise",
          _raises(lambda: eng.parse_blueprint("#E1 = PlasmaProbe(a)"), BlueprintError) and ReWOOEngine(MOCK_TOOLS, strict=False).parse_blueprint("#E1 = PlasmaProbe(a)\n#E2 = PlasmaProbe[b]") == [("#E2", "", "PlasmaProbe", "b")])

    # --- levels ---
    c.add("levels: instructor DAG groups into [[#E1, #E2], [#E3], [#E4]]", eng.build_dependency_levels(steps) == [["#E1", "#E2"], ["#E3"], ["#E4"]])
    chain = [("#E1", "", "PlasmaProbe", "a"), ("#E2", "", "PlasmaProbe", "#E1"), ("#E3", "", "PlasmaProbe", "#E2")]
    wide = [("#E1", "", "PlasmaProbe", "a"), ("#E2", "", "PlasmaProbe", "b"), ("#E3", "", "PlasmaProbe", "c")]
    c.add("levels: a chain is one level per step, independent steps are one level", eng.build_dependency_levels(chain) == [["#E1"], ["#E2"], ["#E3"]] and eng.build_dependency_levels(wide) == [["#E1", "#E2", "#E3"]])
    c.add("levels: a forward reference is fine (order follows dependencies, not line order)",
          eng.build_dependency_levels([("#E1", "", "PlasmaProbe", "#E2"), ("#E2", "", "PlasmaProbe", "b")]) == [["#E2"], ["#E1"]])
    c.add("levels: a cycle raises ValueError", _raises(lambda: eng.build_dependency_levels([("#E1", "", "PlasmaProbe", "#E2"), ("#E2", "", "PlasmaProbe", "#E1")]), ValueError))
    c.add("levels: an unknown variable raises ValueError", _raises(lambda: eng.build_dependency_levels([("#E1", "", "PlasmaProbe", "#E9")]), ValueError))

    # --- substitution ---
    c.add("substitute: whole variables only, #E1 does not clobber #E10",
          ReWOOEngine.substitute("a=#E1 b=#E10 c=#E1", {"#E1": "one", "#E10": "ten"}) == "a=one b=ten c=one")
    c.add("substitute: unresolved variables are left in place", ReWOOEngine.substitute("x=#E3", {}) == "x=#E3")

    # --- execution ---
    ev = eng.execute_dag(steps)
    c.add("execute: instructor evidence (flux and density reach #E3, #E4 wraps #E3)",
          "4.2e37_erg/s" in ev["#E3"] and "14.8_cm^-3" in ev["#E3"] and ev["#E4"].startswith("HarmonicLock(vector=DeltaV("))
    c.add("execute: trace has one record per step with level numbers matching the DAG",
          [r.var_name for r in eng.trace] == ["#E1", "#E2", "#E3", "#E4"] and [r.level for r in eng.trace] == [0, 0, 1, 2] and all(r.status == "ok" for r in eng.trace))
    c.add("execute: unknown tool raises KeyError before anything runs", _raises(lambda: eng.execute_dag([("#E1", "", "NoSuchTool", "a")]), KeyError))
    t_eng = ReWOOEngine(timed_tools(0.04))
    w_steps = t_eng.parse_blueprint(WIDE_BLUEPRINT)
    t_eng.execute_dag(w_steps, parallel=True)
    par = t_eng.wall_clock_s
    t_eng.execute_dag(w_steps, parallel=False)
    ser = t_eng.wall_clock_s
    c.add("execute: a level of four independent tools runs in parallel (wall clock about 3 levels, not 6 calls)",
          par < ser * 0.75 and par < 0.04 * 4.5, f"parallel {par * 1000:.0f} ms vs serial {ser * 1000:.0f} ms")
    c.add("execute: serial time is the sum of tool durations", abs(t_eng.serial_clock_s - ser) < 0.05)
    f_eng = ReWOOEngine(timed_tools(0.0, failing={"TelemetryScan"}))
    ev_f = f_eng.execute_dag(steps)
    st = {r.var_name: r.status for r in f_eng.trace}
    c.add("fault, no fallback: #E1 is an error, #E3 and #E4 are skipped as ERR_UPSTREAM, #E2 still runs",
          st == {"#E1": "error", "#E2": "ok", "#E3": "skipped", "#E4": "skipped"} and ev_f["#E3"].startswith("ERR_UPSTREAM(#E1)") and ev_f["#E4"].startswith("ERR_UPSTREAM(#E3)"))
    ev_v = f_eng.execute_dag(steps, fallback_replan_fn=lambda v, e: "XRayFlux(backup)=4.0e37_erg/s")
    c.add("fault, value fallback: the replacement evidence flows into #E3 and #E4",
          "4.0e37_erg/s" in ev_v["#E3"] and ev_v["#E4"].startswith("HarmonicLock(") and {r.var_name: r.status for r in f_eng.trace}["#E1"] == "fallback")
    ev_r = f_eng.execute_dag(steps, fallback_replan_fn=replan_with_backup)
    sts = [(r.var_name, r.status) for r in f_eng.trace]
    c.add("fault, re-planning fallback: #E10 and #E11 are spliced in and #E3 consumes the calibrated backup",
          ("#E1", "replanned") in sts and ("#E10", "ok") in sts and ("#E11", "ok") in sts and "Calibrated(XRayFlux_backup" in ev_r["#E3"] and ev_r["#E1"] == ev_r["#E11"],
          f"levels run: {f_eng.levels_run}")
    c.add("fault, re-planning fallback: the re-planned steps run in later levels (dynamic DAG growth)", len(f_eng.levels_run) >= 4)
    crash_eng = ReWOOEngine({"Boom": lambda a: 1 / 0, "PlasmaProbe": MOCK_TOOLS["PlasmaProbe"]})
    ev_c = crash_eng.execute_dag([("#E1", "", "Boom", "x"), ("#E2", "", "PlasmaProbe", "#E1")])
    c.add("fault: a crashing tool becomes ERR_EXCEPTION evidence instead of killing the worker", ev_c["#E1"].startswith("ERR_EXCEPTION(ZeroDivisionError") and ev_c["#E2"].startswith("ERR_UPSTREAM"))
    c.add("fault: a re-plan fragment that reuses an existing variable is refused",
          _raises(lambda: ReWOOEngine(timed_tools(0.0, failing={"TelemetryScan"})).execute_dag(steps, fallback_replan_fn=lambda v, e: "#E2 = PlasmaProbe[z]"), BlueprintError))
    c.add("trace is JSON serialisable", isinstance(json.dumps([r.as_dict() for r in f_eng.trace]), str))

    # --- solver ---
    prompt = eng.render_solver_prompt("escape burn", steps, ev)
    c.add("solver prompt contains the task, every plan line and every resolved evidence once",
          prompt.startswith("Task: escape burn") and all(v in prompt for v in ev.values()) and prompt.count("#E3 =") == 1)
    c.add("solve() uses a pluggable solver and defaults to the last evidence",
          eng.solve("t", steps, ev, solver_fn=lambda p: f"{count_tokens(p)} tokens") .endswith("tokens") and ev["#E4"] in eng.solve("t", steps, ev))

    # --- token profiler ---
    m5 = ReWOOEngine.compare_token_cost(5, 1200, 120, 280)
    c.add("instructor numbers: K=5 gives ReAct 10,000 and ReWOO 4,400 tokens, 56 % savings, 2.27x", m5["react_tokens"] == 10000.0 and m5["rewoo_tokens"] == 4400.0 and m5["savings_pct"] == 56.0 and m5["speedup_ratio"] == 2.27)
    c.add("closed form equals the explicit sum for K = 0..20",
          all(ReWOOEngine.compare_token_cost(k, 1200, 120, 280)["react_tokens"] == sum(1200 + (t - 1) * 400 for t in range(1, k + 1)) for k in range(0, 21)))
    c.add("K = 0 and negative inputs are handled", ReWOOEngine.compare_token_cost(0, 1200, 120, 280)["react_tokens"] == 0.0 and _raises(lambda: ReWOOEngine.compare_token_cost(-1, 1, 1, 1), ValueError))
    tbl = ReWOOEngine.cost_table()
    c.add("crossover: ReWOO is more expensive at K = 1 and K = 2 and cheaper from K = 3 on",
          tbl[0]["savings_pct"] < 0 and tbl[1]["savings_pct"] < 0 and all(r["savings_pct"] > 0 for r in tbl[2:]))
    c.add("crossover at K = 3 holds for any positive token sizes",
          all(ReWOOEngine.compare_token_cost(3, b, s, o)["react_tokens"] > ReWOOEngine.compare_token_cost(3, b, s, o)["rewoo_tokens"]
              and ReWOOEngine.compare_token_cost(2, b, s, o)["react_tokens"] <= ReWOOEngine.compare_token_cost(2, b, s, o)["rewoo_tokens"]
              for b in (1, 100, 5000) for s in (1, 50) for o in (1, 500)))
    c.add("savings grow monotonically with K", all(tbl[i]["savings_pct"] < tbl[i + 1]["savings_pct"] for i in range(len(tbl) - 1)))
    sim_ok = all(ReWOOEngine.simulate_token_cost(k, 1200, 120, 280) == {"react_tokens": int(ReWOOEngine.compare_token_cost(k, 1200, 120, 280)["react_tokens"]),
                                                                        "rewoo_tokens": int(ReWOOEngine.compare_token_cost(k, 1200, 120, 280)["rewoo_tokens"])} for k in (1, 3, 5, 8, 16))
    c.add("measured tokens from the mock model equal the closed form exactly for K = 1, 3, 5, 8, 16", sim_ok)
    seq = [ReWOOEngine.compare_token_cost(k, 1200, 120, 280) for k in range(1, 17)]
    d2_react = {round(seq[k + 2]["react_tokens"] - 2 * seq[k + 1]["react_tokens"] + seq[k]["react_tokens"]) for k in range(14)}
    d2_rewoo = {round(seq[k + 2]["rewoo_tokens"] - 2 * seq[k + 1]["rewoo_tokens"] + seq[k]["rewoo_tokens"]) for k in range(14)}
    c.add("ReAct is quadratic (constant second difference = L_step + L_obs = 400), ReWOO is linear (second difference 0)",
          d2_react == {400} and d2_rewoo == {0})

    # --- integration ---
    p4 = load_sibling("sandbox_guardian.py", "part4_guardian")
    if p4 is None:
        c.add("Part 4 guardian present", False, "file missing")
    else:
        try:
            g = p4.SandboxGuardian()
            safe_src = "def tool(arg):\n    return f'IonDensity({arg})=14.8'\n"
            bad_src = "import os\ndef tool(arg):\n    return os.popen(arg).read()\n"
            r_safe = g.inspect_code(safe_src)
            r_bad = p4.SandboxGuardian().inspect_code(bad_src)
            ok_safe = (r_safe[0] if isinstance(r_safe, tuple) else bool(r_safe))
            ok_bad = (r_bad[0] if isinstance(r_bad, tuple) else bool(r_bad))
            c.add("Part 4 sandbox vets a tool implementation before the worker may register it: clean passes, os.popen is blocked", ok_safe and not ok_bad)
        except Exception as exc:
            c.add("Part 4 integration", False, f"{type(exc).__name__}: {exc}")
    p2 = load_sibling("reflexion_surgeon.py", "part2_surgeon")
    if p2 is None:
        c.add("Part 2 surgeon present", False, "file missing")
    else:
        try:
            tp = getattr(p2, "TracebackParser", None)
            parse = getattr(tp, "parse", None) or getattr(tp, "parse_traceback", None)
            tb = 'Traceback (most recent call last):\n  File "hohmann.py", line 12, in calc\n    return 1 / density\nZeroDivisionError: division by zero\n'
            diag = parse(tb) if parse is not None else None
            diag_d = diag.__dict__ if hasattr(diag, "__dict__") else dict(diag or {})
            c.add("Part 2 traceback parser can diagnose an ERR_EXCEPTION fault raised inside a worker tool",
                  diag is not None and "ZeroDivisionError" in json.dumps(diag_d, default=str) and "12" in json.dumps(diag_d, default=str))
        except Exception as exc:
            c.add("Part 2 integration", False, f"{type(exc).__name__}: {exc}")
    p3 = load_sibling("memgpt_pager.py", "part3_pager")
    if p3 is None:
        c.add("Part 3 pager present", False, "file missing")
    else:
        try:
            c.add("Part 3 pager present with core/archival memory operations",
                  hasattr(p3, "MemGPTVirtualPager") and all(hasattr(p3.MemGPTVirtualPager, m) for m in ("core_memory_append", "archival_memory_insert", "archival_memory_search", "check_memory_pressure")))
        except Exception as exc:
            c.add("Part 3 integration", False, f"{type(exc).__name__}: {exc}")

    print(f"--- {c.passed} passed, {c.failed} failed ---")
    return c


# =============================================================================
# Sign-off gate, logging, CLI
# =============================================================================

def check_signoffs(readme: Path = HERE / "README.md") -> Tuple[bool, List[Tuple[str, bool]]]:
    text = readme.read_text(encoding="utf-8")
    rows: List[Tuple[str, bool]] = []
    for part in range(1, 6):
        m = re.search(r"^- \[([ xX])\] \*\*Part " + str(part) + r"\b", text, re.MULTILINE)
        rows.append((f"Part {part}", bool(m and m.group(1).lower() == "x")))
    for label, ok in rows:
        print(f"  [{'x' if ok else ' '}] {label}")
    passed = all(ok for _, ok in rows[:4])
    print("GATE PASSED - Parts 1-4 signed off." if passed else "GATE BLOCKED - unticked parts above.")
    if passed and not rows[4][1]:
        print("  (Part 5, the squadron document and PDF, still open)")
    return passed, rows


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"{stamp}-ps08-{mode}.json"
    record = {"mission": "ps08", "station": "01 Systems Architect", "mode": mode,
              "timestamp": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], **payload}
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"log: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AURA Subsystem 08, Part 1: ReWOO DAG planner")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-signoffs", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    if args.check_signoffs:
        passed, rows = check_signoffs()
        if not args.no_log:
            write_log("signoffs", {"passed": passed, "rows": rows})
        return 0 if passed else 1
    if args.selftest:
        c = self_test()
        if not args.no_log:
            write_log("selftest", {"passed": c.passed, "failed": c.failed, "checks": c.rows})
        return 0 if c.failed == 0 else 1
    h = instructor_harness()
    rep = extended_report()
    if not args.no_log:
        write_log("harness", {"harness": h, **rep})
    return 0


if __name__ == "__main__":
    sys.exit(main())
