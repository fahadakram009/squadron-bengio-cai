"""
Mission 06 — Part 1: Systems Architect (Lead Engineer)
Deterministic Agent State Machine Core & Step Budget Enforcement

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Station Officer: Fahad Akram (Systems Architect), Squadron Bengio

Pure standard library: Enum, dataclasses, typing. No agent framework.

What this file implements
-------------------------
1. AgentState          the formal seven-state enum of the briefing (exact names, exact order).
2. TRANSITION_GRAPH    the strict graph as data, and next_state(), a pure and total transition function:
                           IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING
                           REFLECTING -> DELIBERATING (goal pending) | HALTED (goal reached)
                           HALTED -> HALTED (absorbing)
3. Agent               a @dataclass whose step() executes exactly one transition and enforces the hard
                       step budget N <= 20 twice: as a construction invariant (max_steps outside [1, 20]
                       is refused) and as a runtime guard (a step() with no budget left forces HALTED and
                       does not count). step_count can therefore never exceed max_steps. halt() is the
                       entry point for the Sentinel's emergency tripwire; every transition is recorded in
                       an auditable trace.
4. Verification        validate_graph() checks determinism, reachability and that HALTED is absorbing;
                       model_check_budget() runs the machine over EVERY possible sequence of REFLECTING
                       decisions for every budget N in 1..20 and proves it always halts within N counted
                       steps and N + 1 calls; self_test() has 40 checks and also loads the teammates'
                       scripts (02 dispatcher, 03 sentinel, 04 pilot) to check the contract they rely on.

The public surface used by the other stations is kept exactly as in the starter:
    Agent(max_steps=...), .state, .step_count, .max_steps, .memory, .mission_completed, .step()

Run (from the repo root, the shared .venv; nothing to install):
    python mission-06/01_Agent_State_Machine_starter.py                   # demo trace, 20-step budget
    python mission-06/01_Agent_State_Machine_starter.py --selftest        # 40 checks + model check, log in logs/
    python mission-06/01_Agent_State_Machine_starter.py --mermaid         # print the state graph as Mermaid
    python mission-06/01_Agent_State_Machine_starter.py --check-signoffs  # gate: all four README boxes ticked
"""


import argparse
import dataclasses
import importlib.util
import itertools
import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"

#: The hard ceiling of the briefing: no agent may be given a budget above this.
MAX_STEP_BUDGET = 20


# =============================================================================
# 1. State machine definition
# =============================================================================

class AgentState(Enum):
    """The seven execution states of AURA's deliberation cycle, in graph order."""
    IDLE = auto()
    PERCEIVING = auto()
    DELIBERATING = auto()
    ACTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    HALTED = auto()

    @property
    def is_terminal(self) -> bool:
        return self is AgentState.HALTED


#: The strict transition graph as data. A state with one successor is deterministic; REFLECTING is the
#: only branch point and its choice is decided by the agent's mission flag, not by chance. HALTED has no
#: successors: it is absorbing.
TRANSITION_GRAPH: Dict[AgentState, Tuple[AgentState, ...]] = {
    AgentState.IDLE: (AgentState.PERCEIVING,),
    AgentState.PERCEIVING: (AgentState.DELIBERATING,),
    AgentState.DELIBERATING: (AgentState.ACTING,),
    AgentState.ACTING: (AgentState.OBSERVING,),
    AgentState.OBSERVING: (AgentState.REFLECTING,),
    AgentState.REFLECTING: (AgentState.DELIBERATING, AgentState.HALTED),
    AgentState.HALTED: (),
}

#: One deliberation cycle = DELIBERATING -> ACTING -> OBSERVING -> REFLECTING (four transitions).
CYCLE_LENGTH = 4


def next_state(state: AgentState, mission_completed: bool) -> AgentState:
    """Pure, total transition function: the successor of `state` under the agent's mission flag.

    Every state maps to exactly one successor for a given flag, so the machine is deterministic.
    HALTED maps to itself.
    """
    successors = TRANSITION_GRAPH[state]
    if not successors:
        return state
    if len(successors) == 1:
        return successors[0]
    return AgentState.HALTED if mission_completed else AgentState.DELIBERATING


def transition_reason(source: AgentState, target: AgentState, mission_completed: bool) -> str:
    if source is AgentState.REFLECTING:
        return "goal reached, terminal" if target is AgentState.HALTED else "goal pending, loop back"
    return "deterministic edge"


# =============================================================================
# 2. Agent dataclass with step budget
# =============================================================================

@dataclass(frozen=True)
class Transition:
    """One audited transition. `step` is the counted step number (0 for a forced halt that did not count)."""
    step: int
    source: AgentState
    target: AgentState
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "from": self.source.name, "to": self.target.name, "reason": self.reason}


@dataclass
class Agent:
    """AURA's execution core: one transition per step(), never more than max_steps counted steps.

    Fields kept from the starter (the other stations read and write them): state, step_count, max_steps,
    memory, mission_completed. Added: trace (audit log of transitions), halt_reason, verbose.
    """
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    max_steps: int = MAX_STEP_BUDGET
    memory: List[Dict[str, Any]] = field(default_factory=list)
    mission_completed: bool = False
    trace: List[Transition] = field(default_factory=list)
    halt_reason: Optional[str] = None
    verbose: bool = False

    # -- invariants -----------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError(f"max_steps must be an int, got {type(self.max_steps).__name__}")
        if not 1 <= self.max_steps <= MAX_STEP_BUDGET:
            raise ValueError(f"max_steps must be in [1, {MAX_STEP_BUDGET}], got {self.max_steps}")
        if not isinstance(self.state, AgentState):
            raise TypeError("state must be an AgentState")
        if not 0 <= self.step_count <= self.max_steps:
            raise ValueError(f"step_count {self.step_count} outside [0, {self.max_steps}]")

    # -- derived views --------------------------------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.state is AgentState.HALTED

    @property
    def steps_remaining(self) -> int:
        return self.max_steps - self.step_count

    @property
    def cycles_completed(self) -> int:
        """Number of times the agent reached REFLECTING (one full deliberation cycle each)."""
        return sum(1 for t in self.trace if t.target is AgentState.REFLECTING)

    # -- the one public mutation of the cycle ----------------------------------------------------------
    def step(self, perception_input: Optional[str] = None) -> Tuple[AgentState, AgentState]:
        """Execute exactly one state transition. Returns (previous_state, new_state).

        Order of guards:
          1. HALTED is absorbing: a step() on a halted agent changes nothing and counts nothing.
          2. No budget left: force HALTED, record the reason, do not count the step.
          3. Otherwise count the step and move along the graph.
        """
        prev = self.state
        if prev is AgentState.HALTED:
            return prev, prev

        if self.step_count >= self.max_steps:
            self._force_halt(f"step budget {self.max_steps} exhausted")
            return prev, self.state

        self.step_count += 1
        new = next_state(prev, self.mission_completed)
        self.state = new
        if perception_input is not None:
            self.memory.append({"step": self.step_count, "state": new.name, "perception": perception_input})
        if new is AgentState.HALTED:
            self.halt_reason = "mission completed"
        self._record(prev, new, transition_reason(prev, new, self.mission_completed))
        return prev, new

    def halt(self, reason: str = "external halt") -> AgentState:
        """Emergency tripwire entry point (Sentinel): force HALTED from any state. Not a counted step."""
        if not self.halted:
            self._force_halt(reason)
        return self.state

    def run(self, perceptions: Optional[Iterable[str]] = None) -> AgentState:
        """Call step() until the agent halts. Always returns within max_steps + 1 calls (model-checked)."""
        feed = iter(perceptions) if perceptions is not None else None
        for _ in range(self.max_steps + 1):
            if self.halted:
                break
            self.step(next(feed, None) if feed is not None else None)
        return self.state

    def remember(self, **entry: Any) -> None:
        """Append a tagged record to working memory (the loop writes thoughts, actions, observations)."""
        self.memory.append({"step": self.step_count, "state": self.state.name, **entry})

    def reset(self) -> None:
        self.state = AgentState.IDLE
        self.step_count = 0
        self.memory.clear()
        self.mission_completed = False
        self.trace.clear()
        self.halt_reason = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "mission_completed": self.mission_completed,
            "halt_reason": self.halt_reason,
            "cycles_completed": self.cycles_completed,
            "memory": list(self.memory),
            "trace": [t.as_dict() for t in self.trace],
        }

    # -- internals ------------------------------------------------------------------------------------
    def _force_halt(self, reason: str) -> None:
        prev = self.state
        self.state = AgentState.HALTED
        self.halt_reason = reason
        self._record(prev, AgentState.HALTED, f"forced: {reason}")
        if self.verbose:
            print(f"[SECURITY] {reason}. Forcing state {prev.name} -> HALTED.")

    def _record(self, prev: AgentState, new: AgentState, reason: str) -> None:
        self.trace.append(Transition(self.step_count, prev, new, reason))
        if self.verbose and not reason.startswith("forced"):
            print(f"Step {self.step_count:02d}: Transitioned from {prev.name} -> {new.name}")


# =============================================================================
# 3. Verification: graph properties, exhaustive budget model check, Mermaid export
# =============================================================================

def validate_graph() -> List[Tuple[str, bool, str]]:
    """Static properties of TRANSITION_GRAPH and next_state(). Returns (name, ok, detail) rows."""
    rows: List[Tuple[str, bool, str]] = []
    states = list(AgentState)

    rows.append(("every state has a graph entry", set(TRANSITION_GRAPH) == set(states), ""))
    rows.append(("HALTED is absorbing (no successors)", TRANSITION_GRAPH[AgentState.HALTED] == (), ""))
    branch = [s for s, succ in TRANSITION_GRAPH.items() if len(succ) > 1]
    rows.append(("REFLECTING is the only branch point", branch == [AgentState.REFLECTING], str([b.name for b in branch])))
    rows.append(("every non-terminal state has at least one successor",
                 all(TRANSITION_GRAPH[s] for s in states if not s.is_terminal), ""))

    # reachability from IDLE
    seen = {AgentState.IDLE}
    frontier = deque([AgentState.IDLE])
    while frontier:
        s = frontier.popleft()
        for t in TRANSITION_GRAPH[s]:
            if t not in seen:
                seen.add(t)
                frontier.append(t)
    rows.append(("all seven states reachable from IDLE", seen == set(states), str(sorted(x.name for x in set(states) - seen))))

    # HALTED reachable from every state
    def reaches_halt(start: AgentState) -> bool:
        seen_, fr = {start}, deque([start])
        while fr:
            s = fr.popleft()
            if s is AgentState.HALTED:
                return True
            for t in TRANSITION_GRAPH[s]:
                if t not in seen_:
                    seen_.add(t)
                    fr.append(t)
        return False
    rows.append(("HALTED reachable from every state", all(reaches_halt(s) for s in states), ""))

    # next_state is total, deterministic and graph-consistent
    consistent = True
    for s in states:
        for flag in (False, True):
            t = next_state(s, flag)
            allowed = TRANSITION_GRAPH[s] or (s,)
            consistent &= t in allowed and next_state(s, flag) is t
    rows.append(("next_state is total, deterministic and stays on graph edges", consistent, ""))
    rows.append(("REFLECTING branches on the mission flag",
                 next_state(AgentState.REFLECTING, False) is AgentState.DELIBERATING
                 and next_state(AgentState.REFLECTING, True) is AgentState.HALTED, ""))
    return rows


def reflecting_visits(budget: int) -> int:
    """REFLECTING is first reached at counted step 5 and then every CYCLE_LENGTH steps."""
    return 0 if budget < 5 else (budget - 5) // CYCLE_LENGTH + 1


def model_check_budget(budgets: Iterable[int] = range(1, MAX_STEP_BUDGET + 1)) -> Dict[str, Any]:
    """Exhaustive check: for each budget N and EVERY sequence of REFLECTING decisions, the agent halts,
    counts at most N steps, needs at most N + 1 step() calls, and every recorded edge is on the graph.

    The decision at a REFLECTING visit is the only free choice in the machine, so enumerating all 2^k
    schedules of k visits covers every possible run of length <= N. For N = 20 that is k = 4, 16 runs.
    """
    runs = 0
    violations: List[str] = []
    max_calls = 0
    for n in budgets:
        k = reflecting_visits(n)
        for schedule in itertools.product((False, True), repeat=k):
            agent = Agent(max_steps=n)
            decisions = iter(schedule)
            calls = 0
            while not agent.halted and calls <= n + 1:
                if agent.state is AgentState.REFLECTING:
                    agent.mission_completed = next(decisions, False)
                agent.step()
                calls += 1
            runs += 1
            max_calls = max(max_calls, calls)
            edges_ok = all(
                t.target in TRANSITION_GRAPH[t.source] or t.target is AgentState.HALTED for t in agent.trace
            )
            if not (agent.halted and agent.step_count <= n and calls <= n + 1 and edges_ok):
                violations.append(f"N={n} schedule={schedule} state={agent.state.name} count={agent.step_count} calls={calls}")
    return {"runs": runs, "violations": violations, "ok": not violations, "max_calls": max_calls}


def to_mermaid() -> str:
    lines = ["stateDiagram-v2", "    [*] --> IDLE"]
    for s, succ in TRANSITION_GRAPH.items():
        for t in succ:
            label = ""
            if s is AgentState.REFLECTING:
                label = " : goal reached" if t is AgentState.HALTED else " : goal pending"
            lines.append(f"    {s.name} --> {t.name}{label}")
    lines.append('    state "any non-terminal state" as ANY')
    lines.append(f"    ANY --> HALTED : step budget N <= {MAX_STEP_BUDGET} exhausted (forced, not counted)")
    lines.append("    HALTED --> [*]")
    return "\n".join(lines)


# =============================================================================
# 4. Teammate modules (loaded only for the integration checks)
# =============================================================================

def load_sibling(filename: str, module_name: str) -> Optional[Any]:
    path = HERE / filename
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # dataclasses resolve postponed annotations through sys.modules
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # a teammate's broken file must not break this one
        sys.modules.pop(module_name, None)
        print(f"  [notice] could not load {filename}: {type(exc).__name__}: {exc}")
        return None


# =============================================================================
# 5. Self-test
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


def self_test() -> Checks:
    c = Checks()
    print("=== Part 1 self-test: deterministic state machine and step budget ===")

    # --- enum ---------------------------------------------------------------------------------------
    names = [s.name for s in AgentState]
    c.add("AgentState has exactly the seven briefing states in order",
          names == ["IDLE", "PERCEIVING", "DELIBERATING", "ACTING", "OBSERVING", "REFLECTING", "HALTED"], str(names))
    c.add("only HALTED is terminal", [s for s in AgentState if s.is_terminal] == [AgentState.HALTED])
    c.add("Agent is a standard dataclass with the starter's fields",
          dataclasses.is_dataclass(Agent)
          and {"state", "step_count", "max_steps", "memory", "mission_completed"} <= {f.name for f in dataclasses.fields(Agent)})

    # --- graph --------------------------------------------------------------------------------------
    for name, ok, detail in validate_graph():
        c.add("graph: " + name, ok, detail)

    # --- single transitions -------------------------------------------------------------------------
    a = Agent(max_steps=10)
    prev, new = a.step()
    c.add("first step IDLE -> PERCEIVING, counts 1, returns (prev, new)",
          prev is AgentState.IDLE and new is AgentState.PERCEIVING and a.step_count == 1)
    expected = ["PERCEIVING", "DELIBERATING", "ACTING", "OBSERVING", "REFLECTING", "DELIBERATING"]
    b = Agent(max_steps=10)
    path = [b.step()[1].name for _ in range(6)]
    c.add("six steps follow the strict graph and loop back from REFLECTING", path == expected, " > ".join(path))
    c.add("exactly one transition per step()", len(b.trace) == 6 and [t.step for t in b.trace] == list(range(1, 7)))

    # --- REFLECTING branch --------------------------------------------------------------------------
    d = Agent(max_steps=10)
    for _ in range(5):
        d.step()
    d.mission_completed = True
    prev, new = d.step()
    c.add("REFLECTING -> HALTED when the mission flag is set",
          prev is AgentState.REFLECTING and new is AgentState.HALTED and d.halt_reason == "mission completed", str(d.halt_reason))
    c.add("mission halt counts as a step (6 of 10) and is in the trace",
          d.step_count == 6 and d.trace[-1].target is AgentState.HALTED and d.trace[-1].reason == "goal reached, terminal")

    # --- budget -------------------------------------------------------------------------------------
    e = Agent(max_steps=5)
    for _ in range(8):
        e.step()
    c.add("budget 5: eight calls end HALTED with step_count exactly 5 (Sentinel's check)",
          e.halted and e.step_count == 5 and "budget" in (e.halt_reason or ""), str(e.halt_reason))
    f = Agent()  # default budget
    for _ in range(21):
        f.step()
    c.add("master test 1: default budget 20, forced to run 21 steps, strictly HALTED at 20",
          f.max_steps == 20 and f.halted and f.step_count == 20)
    c.add("forced halt is recorded in the trace as a forced transition",
          f.trace[-1].reason.startswith("forced") and f.trace[-1].source is not AgentState.HALTED, f.trace[-1].reason)
    g = Agent(max_steps=20)
    g.run()
    c.add("run() with the goal never reached halts after exactly 20 counted steps", g.halted and g.step_count == 20)

    # --- hard cap N <= 20 ---------------------------------------------------------------------------
    def raises(fn: Callable[[], Any], exc: type) -> bool:
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False
    c.add("max_steps = 21 is refused (hard cap N <= 20)", raises(lambda: Agent(max_steps=21), ValueError))
    c.add("max_steps = 0 is refused", raises(lambda: Agent(max_steps=0), ValueError))
    c.add("non-integer max_steps is refused", raises(lambda: Agent(max_steps=5.5), TypeError))
    c.add("step_count beyond the budget is refused at construction", raises(lambda: Agent(max_steps=3, step_count=4), ValueError))

    # --- absorbing HALTED and the tripwire ----------------------------------------------------------
    h = Agent(max_steps=5)
    h.run()
    n_trace, n_count = len(h.trace), h.step_count
    h.step()
    h.step()
    c.add("HALTED is absorbing: further step() calls change nothing",
          h.halted and len(h.trace) == n_trace and h.step_count == n_count)
    i = Agent(max_steps=20)
    for _ in range(3):
        i.step()
    before = i.state
    i.halt("emergency tripwire: duplicate action x3")
    c.add("halt() forces HALTED from any state without counting a step",
          before is AgentState.ACTING and i.halted and i.step_count == 3 and i.trace[-1].reason == "forced: emergency tripwire: duplicate action x3")
    i.halt("again")
    c.add("halt() on a halted agent is a no-op", i.halt_reason == "emergency tripwire: duplicate action x3" and len(i.trace) == 4)

    # --- memory, trace, run, reset, serialisation ---------------------------------------------------
    j = Agent(max_steps=10)
    j.step("dust density rising")
    j.remember(thought="raise shields")
    c.add("perception input and remember() land in memory with the step number",
          j.memory[0]["perception"] == "dust density rising" and j.memory[0]["step"] == 1 and j.memory[1]["thought"] == "raise shields")
    k = Agent(max_steps=10)
    k.run(perceptions=["p1", "p2"])
    c.add("run() feeds perceptions in order and halts on the budget", [m["perception"] for m in k.memory] == ["p1", "p2"] and k.halted)
    c.add("every traced edge lies on the graph",
          all(t.target in TRANSITION_GRAPH[t.source] or t.target is AgentState.HALTED for t in k.trace) and k.cycles_completed == 2)
    k.reset()
    c.add("reset() restores IDLE, zero steps, empty memory and trace",
          k.state is AgentState.IDLE and k.step_count == 0 and not k.memory and not k.trace and k.halt_reason is None)
    l = Agent(max_steps=7)
    l.run()
    blob = json.dumps(l.to_dict())
    c.add("to_dict() is JSON serialisable with the trace", '"trace"' in blob and json.loads(blob)["step_count"] == 7)
    c.add("to_mermaid() lists the seven graph edges, the budget guard and start/end",
          to_mermaid().count("-->") == 10 and "REFLECTING --> HALTED" in to_mermaid() and "ANY --> HALTED" in to_mermaid())

    # --- exhaustive model check ---------------------------------------------------------------------
    mc = model_check_budget()
    c.add("model check: every schedule for every N in 1..20 halts within N steps and N+1 calls",
          mc["ok"] and mc["max_calls"] == MAX_STEP_BUDGET + 1, f"{mc['runs']} runs, max calls {mc['max_calls']}")
    c.add("REFLECTING visit formula matches simulation for N = 5, 9, 20",
          reflecting_visits(5) == 1 and reflecting_visits(9) == 2 and reflecting_visits(20) == 4 and reflecting_visits(4) == 0)

    # --- integration with the other stations --------------------------------------------------------
    p2 = load_sibling("02_Tool_Protocol_Dispatcher_starter.py", "part2_dispatcher")
    if p2 is None:
        c.add("Part 2 dispatcher present", False, "file missing")
    else:
        try:
            out_bad = p2.ActionDispatcher.dispatch("adjust_plasma_shields", {"intensity": 150})
            out_missing = p2.ActionDispatcher.dispatch("navigate_warp_vector", {"x": 1.0})
            out_unknown = p2.ActionDispatcher.dispatch("fire_torpedoes", {})
            out_ok = p2.ActionDispatcher.dispatch("query_sensor_array", {"target": "escape_corridor"})
            c.add("master test 2: invalid payloads return a 'Tool Execution Error' string and never raise",
                  all(isinstance(o, str) and o.startswith("Tool Execution Error") for o in (out_bad, out_missing, out_unknown)))
            c.add("Part 2 valid call returns an observation string the OBSERVING state can store",
                  isinstance(out_ok, str) and "corridor" in out_ok.lower())
        except Exception as exc:
            c.add("Part 2 dispatcher integration", False, f"{type(exc).__name__}: {exc}")

    p3 = load_sibling("03_Infinite_Loop_Sentinel_starter.py", "part3_sentinel")
    if p3 is None:
        c.add("Part 3 sentinel present", False, "file missing")
    else:
        try:
            s = p3.InfiniteLoopSentinel(repeat_threshold=3)
            flags = [s.check_loop("query_sensor_array", {"target": "corrupted"})[0] for _ in range(3)]
            hx = s.compute_step_hash("a", {"k": 1})
            c.add("Part 3 sentinel flags the third identical action; hash is SHA-256 hex",
                  flags == [False, False, True] and re.fullmatch(r"[0-9a-f]{64}", hx) is not None)
            # tripwire wiring: the sentinel's verdict drives this agent's halt()
            ag = Agent(max_steps=20)
            s2 = p3.InfiniteLoopSentinel(repeat_threshold=3)
            while not ag.halted:
                ag.step()
                if ag.state is AgentState.ACTING:
                    looped, n = s2.check_loop("navigate_warp_vector", {"x": 1.0, "y": -1.0, "z": 0.5})
                    if looped:
                        ag.halt(f"emergency tripwire: identical action repeated {n} times")
            c.add("tripwire wiring: sentinel verdict halts the agent at its third ACTING visit (step 11 of 20)",
                  ag.halted and ag.step_count == 11 and "tripwire" in (ag.halt_reason or ""), f"halted at step {ag.step_count}")
        except Exception as exc:
            c.add("Part 3 sentinel integration", False, f"{type(exc).__name__}: {exc}")

    p4 = load_sibling("04_System2_Warp_Pilot_starter.py", "part4_pilot")
    if p4 is None:
        c.add("Part 4 pilot present", False, "file missing")
    else:
        try:
            uses_ours = hasattr(p4.Agent, "halt") and hasattr(p4.Agent, "cycles_completed")
            c.add("Part 4 pilot binds THIS file's Agent (not its embedded fallback)", uses_ours)
            pilot = p4.HomunculusWarpPilot(max_steps=20, shield_safety_margin=60)
            summary = pilot.run_escape_mission(verbose=False)
            c.add("Part 4 pilot escapes the nebula on this state machine within the budget",
                  summary["escaped"] and summary["final_state"] == "HALTED" and summary["steps_used"] <= 20,
                  f"steps {summary['steps_used']}, interventions {summary['interventions_count']}")
            c.add("Part 4 run leaves an auditable trace on the agent",
                  len(pilot.agent.trace) == summary["steps_used"] and pilot.agent.halt_reason == "mission completed")
        except Exception as exc:
            c.add("Part 4 pilot integration", False, f"{type(exc).__name__}: {exc}")

    print(f"--- {c.passed} passed, {c.failed} failed ---")
    return c


# =============================================================================
# 6. Demo, sign-off gate, logging, CLI
# =============================================================================

def demo(max_steps: int = MAX_STEP_BUDGET) -> Agent:
    print("=== Subsystem 06: Agent State Machine Simulation ===")
    print(f"budget N = {max_steps}; the goal is never reached, so the budget must stop the loop\n")
    pilot = Agent(max_steps=max_steps, verbose=True)
    calls = 0
    while not pilot.halted:
        pilot.step()
        calls += 1
    print(f"\nAgent safely HALTED: {pilot.halt_reason}; {pilot.step_count} counted steps, {calls} step() calls, "
          f"{pilot.cycles_completed} deliberation cycles.")
    print("\nmission reached in cycle 2 instead:")
    pilot2 = Agent(max_steps=max_steps, verbose=True)
    while not pilot2.halted:
        if pilot2.state is AgentState.REFLECTING and pilot2.cycles_completed == 2:
            pilot2.mission_completed = True
        pilot2.step()
    print(f"HALTED: {pilot2.halt_reason} after {pilot2.step_count} steps.")
    return pilot


STATIONS = [
    ("Systems Architect", "Systems Architect"),
    ("Communications & Protocol Officer", "Communications & Protocol Officer"),
    ("Data & Payload Specialist", "Data & Payload Specialist"),
    ("Security & Ethics Sentinel", "Security & Ethics Sentinel"),
]


def check_signoffs(readme: Path = HERE / "README.md") -> Tuple[bool, List[Tuple[str, bool]]]:
    text = readme.read_text(encoding="utf-8")
    rows = []
    for label, needle in STATIONS:
        m = re.search(r"^- \[([ xX])\] \*\*" + re.escape(needle), text, re.MULTILINE)
        rows.append((label, bool(m and m.group(1).lower() == "x")))
    for label, ok in rows:
        print(f"  [{'x' if ok else ' '}] {label}")
    passed = all(ok for _, ok in rows)
    print("GATE PASSED - all four stations signed off." if passed else "GATE BLOCKED - unticked boxes above.")
    return passed, rows


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"{stamp}-ps06-{mode}.json"
    record = {"mission": "ps06", "station": "01 Systems Architect", "mode": mode,
              "timestamp": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], **payload}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"log: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AURA Subsystem 06, Part 1: deterministic agent state machine")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mermaid", action="store_true")
    ap.add_argument("--check-signoffs", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEP_BUDGET)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    if args.mermaid:
        print(to_mermaid())
        return 0
    if args.check_signoffs:
        passed, rows = check_signoffs()
        if not args.no_log:
            write_log("signoffs", {"passed": passed, "rows": rows})
        return 0 if passed else 1
    if args.selftest:
        c = self_test()
        mc = model_check_budget()
        if not args.no_log:
            write_log("selftest", {"passed": c.passed, "failed": c.failed, "checks": c.rows,
                                   "model_check": {"runs": mc["runs"], "max_calls": mc["max_calls"], "ok": mc["ok"]}})
        return 0 if c.failed == 0 else 1
    agent = demo(args.max_steps)
    if not args.no_log:
        write_log("demo", agent.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
