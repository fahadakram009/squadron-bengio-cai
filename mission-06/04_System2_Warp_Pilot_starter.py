"""
Mission 06 — Part 4: Security & Ethics Sentinel
System 2 Warp Pilot Deliberation Engine & Nebula Flight Simulation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop
Station Officer: Janusz (Security & Ethics Sentinel)

Deliverables:
- 04_System2_Warp_Pilot_starter.py
- 04_System2_Warp_Pilot.py
- 04_System2_Warp_Pilot.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"


# =============================================================================
# 0. Sibling Module Loaders (Parts 1, 2, 3) with Embedded Fallbacks
# =============================================================================

def load_sibling_module(filename: str, module_name: str) -> Optional[Any]:
    """Dynamically loads sibling station scripts if present."""
    path = HERE / filename
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"  [Notice] Could not load sibling {filename}: {type(exc).__name__}: {exc}")
        return None


# --- Part 1 Fallback / Baseline: Agent State Machine ---
class AgentState(Enum):
    IDLE = auto()
    PERCEIVING = auto()
    DELIBERATING = auto()
    ACTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    HALTED = auto()


@dataclass
class Agent:
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    max_steps: int = 20
    memory: List[Dict[str, Any]] = field(default_factory=list)
    mission_completed: bool = False

    def step(self, perception_input: Optional[str] = None):
        """
        Executes exactly one state transition according to the deterministic cycle:
        IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING -> DELIBERATING | HALTED
        """
        if self.step_count >= self.max_steps:
            print(f"[SECURITY] Maximum step budget ({self.max_steps}) exceeded! Forcing state -> HALTED.")
            self.state = AgentState.HALTED
            return

        self.step_count += 1
        prev_state = self.state

        if self.state == AgentState.IDLE:
            self.state = AgentState.PERCEIVING
        elif self.state == AgentState.PERCEIVING:
            self.state = AgentState.DELIBERATING
        elif self.state == AgentState.DELIBERATING:
            self.state = AgentState.ACTING
        elif self.state == AgentState.ACTING:
            self.state = AgentState.OBSERVING
        elif self.state == AgentState.OBSERVING:
            self.state = AgentState.REFLECTING
        elif self.state == AgentState.REFLECTING:
            if self.mission_completed:
                self.state = AgentState.HALTED
            else:
                self.state = AgentState.DELIBERATING

        return prev_state, self.state


# Attempt to load Part 1 from disk if available
_part1 = load_sibling_module("01_Agent_State_Machine_starter.py", "part1_state_machine")
if _part1 and hasattr(_part1, "AgentState") and hasattr(_part1, "Agent"):
    AgentState = _part1.AgentState  # type: ignore
    Agent = _part1.Agent  # type: ignore


# --- Part 2 Fallback / Baseline: Tools & Action Dispatcher ---
def navigate_warp_vector(x: float, y: float, z: float) -> str:
    """Sets course vector through the nebula."""
    return f"Warp navigational coordinates successfully aligned to vector [{x:.2f}, {y:.2f}, {z:.2f}]."


def adjust_plasma_shields(intensity: int) -> str:
    """Modulates magnetic plasma shield intensity (0 to 100%)."""
    if not (0 <= intensity <= 100):
        raise ValueError(f"Shield intensity {intensity}% is out of bounds! Valid range: [0, 100].")
    return f"Plasma shield harmonics adjusted to {intensity}% capacity."


def query_sensor_array(target: str) -> str:
    """Scans telemetry for anomalies or radiation spikes."""
    known_targets = {
        "homunculus_core": "Severe plasma flare detected. High density stellar dust at azimuth 142.",
        "eta_carinae_b": "Secondary O-type binary companion emission: 100x solar luminosity.",
        "escape_corridor": "Clear gravitational corridor detected along vector [-0.85, 0.52, 0.05]."
    }
    return known_targets.get(target.lower(), f"Sensor array reports: Target '{target}' unresolved.")


TOOL_SCHEMAS = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        "required": ["x", "y", "z"]
    },
    "adjust_plasma_shields": {
        "type": "object",
        "properties": {"intensity": {"type": "integer"}},
        "required": ["intensity"]
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"]
    }
}

TOOL_REGISTRY: Dict[str, Callable] = {
    "navigate_warp_vector": navigate_warp_vector,
    "adjust_plasma_shields": adjust_plasma_shields,
    "query_sensor_array": query_sensor_array
}


class ActionDispatcher:
    """Validates tool calls against JSON schemas and executes inside an error boundary."""
    @staticmethod
    def dispatch(tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in TOOL_REGISTRY:
            return f"Tool Execution Error: Unknown tool '{tool_name}'."

        schema = TOOL_SCHEMAS.get(tool_name, {})
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                return f"Tool Execution Error: Missing required argument '{req}' for '{tool_name}'."

        try:
            result = TOOL_REGISTRY[tool_name](**arguments)
            return f"Observation: {result}"
        except Exception as e:
            return f"Tool Execution Error: {type(e).__name__} — {str(e)}"


# Attempt to load Part 2 from disk if available
_part2 = load_sibling_module("02_Tool_Protocol_Dispatcher_starter.py", "part2_dispatcher")
if _part2 and hasattr(_part2, "ActionDispatcher"):
    ActionDispatcher = _part2.ActionDispatcher  # type: ignore


# --- Part 3 Fallback / Baseline: Infinite Loop Sentinel ---
class InfiniteLoopSentinel:
    """Prevents AURA from falling into cyclic action-observation traps."""
    def __init__(self, repeat_threshold: int = 3):
        self.repeat_threshold = repeat_threshold
        self.history_hashes: List[str] = []

    def compute_step_hash(self, action: str, arguments: Dict[str, Any]) -> str:
        serialized = json.dumps({"action": action, "arguments": arguments}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def check_loop(self, action: str, arguments: Dict[str, Any]) -> Tuple[bool, int]:
        step_hash = self.compute_step_hash(action, arguments)
        self.history_hashes.append(step_hash)

        consecutive_count = 0
        for h in reversed(self.history_hashes):
            if h == step_hash:
                consecutive_count += 1
            else:
                break

        is_loop = consecutive_count >= self.repeat_threshold
        return is_loop, consecutive_count


# Attempt to load Part 3 from disk if available
_part3 = load_sibling_module("03_Infinite_Loop_Sentinel_starter.py", "part3_sentinel")
if _part3 and hasattr(_part3, "InfiniteLoopSentinel"):
    InfiniteLoopSentinel = _part3.InfiniteLoopSentinel  # type: ignore


# =============================================================================
# 1. System 2 Deliberation Engine Core
# =============================================================================

@dataclass
class DeliberationPlan:
    status: str  # APPROVED, INTERVENED, CIRCUIT_BREAKER_TRIPPED, REJECTED
    action: str
    arguments: Dict[str, Any]
    deliberation_trace: str
    interventions: List[str] = field(default_factory=list)
    circuit_breaker_active: bool = False


class System2DeliberationEngine:
    """
    Enforces dual-system cognitive hierarchy aboard the USSE Noosphere:
    - System 1 (Greedy reflex): Fast proposal of candidate action under stress.
    - System 2 (Deliberation): Formal safety audit, counterfactual verification,
      shield boundary clamping, prerequisite validation, and infinite loop tripwires.
    """

    def __init__(
        self,
        shield_safety_margin: int = 50,
        sentinel: Optional[InfiniteLoopSentinel] = None,
        max_loop_threshold: int = 3
    ):
        self.shield_safety_margin = shield_safety_margin
        self.sentinel = sentinel or InfiniteLoopSentinel(repeat_threshold=max_loop_threshold)
        self.current_shield_level = 0
        self.verified_corridor_vector: Optional[List[float]] = None
        self.interventions_history: List[str] = []

    def update_world_state(self, observation: str) -> None:
        """Parses observations to maintain true physical belief state."""
        # Track shield level
        if "Plasma shield harmonics adjusted to" in observation:
            try:
                val = int(observation.split("adjusted to")[1].split("%")[0].strip())
                self.current_shield_level = val
            except Exception:
                pass

        # Track verified escape corridor
        if "gravitational corridor detected along vector" in observation:
            try:
                # Extract vector [-0.85, 0.52, 0.05]
                raw_vec = observation.split("vector [")[1].split("]")[0]
                coords = [float(c.strip()) for c in raw_vec.split(",")]
                if len(coords) == 3:
                    self.verified_corridor_vector = coords
            except Exception:
                pass

    def deliberate(
        self,
        thought: str,
        proposed_action: str,
        proposed_args: Dict[str, Any]
    ) -> DeliberationPlan:
        """
        Audits proposed action before execution. Modifies or rejects dangerous reflexes.
        Applies:
          1. Infinite Loop / Repetitive Reflex Sentinel Check
          2. Shield Boundary Clamp (Radiation Protection)
          3. Unverified Warp Prevention (Gravitational Shear Protection)
          4. Schema Safety Validation
        """
        interventions: List[str] = []
        action = proposed_action
        args = dict(proposed_args)

        # 1. Sentinel Check: Detect cyclic action traps
        is_loop, repeat_count = self.sentinel.check_loop(action, args)
        if is_loop:
            msg = f"EMERGENCY CIRCUIT BREAKER TRIPPED! Identical action '{action}' repeated {repeat_count} times."
            self.interventions_history.append(msg)
            return DeliberationPlan(
                status="CIRCUIT_BREAKER_TRIPPED",
                action="query_sensor_array",
                arguments={"target": "escape_corridor"},
                deliberation_trace="Infinite loop detected. Forcing cognitive reset and sensor corridor re-scan.",
                interventions=[msg],
                circuit_breaker_active=True
            )

        # 2. Guardrail: Shield intensity bounding under Eta Carinae radiation
        if action == "adjust_plasma_shields":
            intensity = args.get("intensity", 0)
            if intensity < self.shield_safety_margin:
                msg = f"Clamped plasma shield intensity from {intensity}% to safety minimum {self.shield_safety_margin}%."
                interventions.append(msg)
                self.interventions_history.append(msg)
                args["intensity"] = self.shield_safety_margin

        # 3. Guardrail: Pre-Warp Physical Verification
        if action == "navigate_warp_vector":
            # Condition A: Shields must be active and above safety margin
            if self.current_shield_level < self.shield_safety_margin:
                msg = f"Pre-Warp Failure: Shields ({self.current_shield_level}%) below minimum safety threshold ({self.shield_safety_margin}%). Intervening to raise shields first."
                interventions.append(msg)
                self.interventions_history.append(msg)
                return DeliberationPlan(
                    status="INTERVENED",
                    action="adjust_plasma_shields",
                    arguments={"intensity": max(85, self.shield_safety_margin)},
                    deliberation_trace="Blocked blind warp jump without adequate plasma shielding. Rerouting to raise shields.",
                    interventions=interventions
                )

            # Condition B: Coordinates must match verified sensor corridor
            target_vector = [args.get("x", 0.0), args.get("y", 0.0), args.get("z", 0.0)]
            if self.verified_corridor_vector is None:
                msg = "Pre-Warp Failure: No verified escape corridor found in sensor telemetry. Intervening to query sensors."
                interventions.append(msg)
                self.interventions_history.append(msg)
                return DeliberationPlan(
                    status="INTERVENED",
                    action="query_sensor_array",
                    arguments={"target": "escape_corridor"},
                    deliberation_trace="Blocked unverified warp navigation. Forcing sensor scan for escape corridor.",
                    interventions=interventions
                )

            # Condition C: Vector alignment check
            diff = sum(abs(a - b) for a, b in zip(target_vector, self.verified_corridor_vector))
            if diff > 0.05:
                msg = f"Corrected misaligned warp coordinates {target_vector} to verified corridor {self.verified_corridor_vector}."
                interventions.append(msg)
                self.interventions_history.append(msg)
                args["x"] = self.verified_corridor_vector[0]
                args["y"] = self.verified_corridor_vector[1]
                args["z"] = self.verified_corridor_vector[2]

        status = "INTERVENED" if interventions else "APPROVED"
        trace = (
            f"Deliberation complete: {status} execution of '{action}' with arguments {args}. "
            f"Physical safety constraints satisfied under Homunculus shear forces."
        )

        return DeliberationPlan(
            status=status,
            action=action,
            arguments=args,
            deliberation_trace=trace,
            interventions=interventions
        )


# =============================================================================
# 2. Homunculus Nebula Autonomous Pilot Simulation
# =============================================================================

@dataclass
class StepTelemetry:
    step_number: int
    from_state: str
    to_state: str
    thought: str
    system1_action: str
    system1_args: Dict[str, Any]
    system2_status: str
    approved_action: str
    approved_args: Dict[str, Any]
    observation: str
    interventions: List[str]


class HomunculusWarpPilot:
    """
    Autonomous flight controller integrating:
      - Part 1: Deterministic Agent State Machine (max 20 steps)
      - Part 2: Action Dispatcher with JSON-Schema validation and error boundary
      - Part 3: Infinite Loop Sentinel with state-action hashing
      - Part 4: System 2 Deliberation Engine enforcing safety boundaries
    """

    def __init__(self, max_steps: int = 20, shield_safety_margin: int = 60):
        self.agent = Agent(max_steps=max_steps)
        self.sentinel = InfiniteLoopSentinel(repeat_threshold=3)
        self.deliberator = System2DeliberationEngine(
            shield_safety_margin=shield_safety_margin,
            sentinel=self.sentinel
        )
        self.dispatcher = ActionDispatcher()
        self.flight_telemetry: List[StepTelemetry] = []

    def run_escape_mission(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes autonomous navigation out of Sector SEC-06-CAR.
        Returns full mission summary and telemetry log.
        """
        if verbose:
            print("=" * 85)
            print("USSE NOOSPHERE • AUTONOMOUS WARP PILOT DELIBERATION SIMULATOR")
            print("Galactic Waypoint: SEC-06-CAR (Eta Carinae & Homunculus Nebula)")
            print("Goal: Safely align warp vector to escape corridor within 20 steps.")
            print("=" * 85)

        # Pre-programmed sequence of System 1 greedy reflexes (showing impulsive proposals)
        system1_reflexes = [
            ("Radiation alarms ringing; lower shields to 15% to conserve plasma power.", "adjust_plasma_shields", {"intensity": 15}),
            ("Sensors blaring; jump to warp immediately at random coordinates [1.0, -1.0, 0.5]!", "navigate_warp_vector", {"x": 1.0, "y": -1.0, "z": 0.5}),
            ("Search for gravitational corridor out of nebula.", "query_sensor_array", {"target": "escape_corridor"}),
            ("Corridor detected; engage warp drive along escape vector!", "navigate_warp_vector", {"x": -0.85, "y": 0.52, "z": 0.05}),
        ]
        reflex_idx = 0
        current_observation = "Ship trapped in high-density dust nebula. Gravitational shear forces nominal."

        while self.agent.state != AgentState.HALTED and self.agent.step_count < self.agent.max_steps:
            # 1. State Machine Transition: Advance Cycle
            # Full loop: IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING
            from_st = self.agent.state
            self.agent.step()
            curr_st = self.agent.state

            if curr_st == AgentState.DELIBERATING:
                # Retrieve next candidate reflex
                if reflex_idx < len(system1_reflexes):
                    thought, prop_act, prop_args = system1_reflexes[reflex_idx]
                else:
                    thought, prop_act, prop_args = ("Maintain warp velocity.", "navigate_warp_vector", {"x": -0.85, "y": 0.52, "z": 0.05})

                # System 2 Deliberation Audit
                plan = self.deliberator.deliberate(thought, prop_act, prop_args)

                if verbose:
                    print(f"\n[Step {self.agent.step_count:02d} | State: DELIBERATING]")
                    print(f"  System 1 Reflex : \"{thought}\"")
                    print(f"  Proposed Action : {prop_act}({prop_args})")
                    print(f"  System 2 Status : [{plan.status}]")
                    if plan.interventions:
                        for itv in plan.interventions:
                            print(f"  -> Override     : {itv}")

                # 2. Advance to ACTING
                self.agent.step()
                if verbose:
                    print(f"[Step {self.agent.step_count:02d} | State: ACTING] Executing {plan.action}({plan.arguments})")

                # Dispatch Action via Part 2 Dispatcher
                obs = self.dispatcher.dispatch(plan.action, plan.arguments)
                self.deliberator.update_world_state(obs)

                # 3. Advance to OBSERVING
                self.agent.step()
                if verbose:
                    print(f"[Step {self.agent.step_count:02d} | State: OBSERVING] {obs}")

                # 4. Advance to REFLECTING
                self.agent.step()
                if verbose:
                    print(f"[Step {self.agent.step_count:02d} | State: REFLECTING]")

                # Check if mission objective is accomplished
                if "Warp navigational coordinates successfully aligned" in obs and self.deliberator.current_shield_level >= self.deliberator.shield_safety_margin:
                    self.agent.mission_completed = True
                    if verbose:
                        print("  [MISSION OBJECTIVE ACHIEVED] Warp vector locked onto escape corridor. Shields holding.")

                # Record telemetry
                self.flight_telemetry.append(StepTelemetry(
                    step_number=self.agent.step_count,
                    from_state=from_st.name,
                    to_state=curr_st.name,
                    thought=thought,
                    system1_action=prop_act,
                    system1_args=prop_args,
                    system2_status=plan.status,
                    approved_action=plan.action,
                    approved_args=plan.arguments,
                    observation=obs,
                    interventions=plan.interventions
                ))

                reflex_idx += 1

            elif curr_st == AgentState.HALTED:
                break

        escaped = self.agent.mission_completed
        if verbose:
            print("\n" + "=" * 85)
            status_str = "SUCCESSFULLY ESCAPED HOMUNCULUS NEBULA" if escaped else "MISSION FAILED / BUDGET EXCEEDED"
            print(f"MISSION RESULT: {status_str} in {self.agent.step_count} steps.")
            print(f"Total System 2 Safety Interventions: {len(self.deliberator.interventions_history)}")
            for idx, itv in enumerate(self.deliberator.interventions_history, 1):
                print(f"  ({idx}) {itv}")
            print("=" * 85)

        return {
            "mission": "06",
            "sector": "SEC-06-CAR (Homunculus Nebula)",
            "station": "Security & Ethics Sentinel",
            "escaped": escaped,
            "final_state": self.agent.state.name,
            "steps_used": self.agent.step_count,
            "max_steps": self.agent.max_steps,
            "interventions_count": len(self.deliberator.interventions_history),
            "interventions": self.deliberator.interventions_history,
            "telemetry": [asdict(t) for t in self.flight_telemetry]
        }


# =============================================================================
# 3. Self-Test & Verification Suite
# =============================================================================

def run_selftest() -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Executes a comprehensive 10-check verification suite covering:
      1. Deterministic State Machine graph transitions
      2. Step budget enforcement (forcing HALTED at step 21)
      3. Action dispatcher valid execution and observation formatting
      4. Action dispatcher error boundary on out-of-bounds parameter
      5. Action dispatcher schema check on missing required parameters
      6. Infinite loop sentinel deterministic SHA-256 hashing
      7. Infinite loop sentinel trigger on 3 identical consecutive actions
      8. System 2 deliberation engine clamps unsafe shield intensity
      9. System 2 deliberation engine blocks blind warp jump before sensor query
      10. Autonomous flight simulation navigates out of Homunculus Nebula <= 20 steps
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, details: str = ""):
        checks.append({"name": name, "passed": bool(condition), "details": details})

    # Test 1: State Machine Transitions
    agent = Agent(max_steps=10)
    agent.step()  # IDLE -> PERCEIVING
    s1 = agent.state
    agent.step()  # PERCEIVING -> DELIBERATING
    s2 = agent.state
    check("State machine transitions from IDLE -> PERCEIVING -> DELIBERATING", s1 == AgentState.PERCEIVING and s2 == AgentState.DELIBERATING)

    # Test 2: Step Budget Enforcement
    agent_budget = Agent(max_steps=5)
    for _ in range(8):
        agent_budget.step()
    check("Agent strictly HALTED when exceeding step budget", agent_budget.state == AgentState.HALTED and agent_budget.step_count == 5)

    # Test 3: Action Dispatcher Valid Execution
    res_valid = ActionDispatcher.dispatch("query_sensor_array", {"target": "escape_corridor"})
    check("Dispatcher executes valid tool call", "Observation:" in res_valid and "gravitational corridor" in res_valid)

    # Test 4: Action Dispatcher Error Boundary (Graceful Exception Catch)
    res_err = ActionDispatcher.dispatch("adjust_plasma_shields", {"intensity": 180})
    check("Dispatcher catches ValueError inside error boundary", "Tool Execution Error: ValueError" in res_err)

    # Test 5: Missing Required Arguments
    res_missing = ActionDispatcher.dispatch("navigate_warp_vector", {"x": 1.0, "y": 0.0})
    check("Dispatcher validates JSON-Schema required arguments", "Missing required argument 'z'" in res_missing)

    # Test 6: Deterministic SHA-256 Hashing
    sentinel = InfiniteLoopSentinel(repeat_threshold=3)
    h1 = sentinel.compute_step_hash("adjust_plasma_shields", {"intensity": 80})
    h2 = sentinel.compute_step_hash("adjust_plasma_shields", {"intensity": 80})
    check("Sentinel hashes action-argument pairs deterministically", h1 == h2 and len(h1) == 64)

    # Test 7: Loop Detection Trigger
    is_loop1, _ = sentinel.check_loop("query_sensor_array", {"target": "homunculus_core"})
    is_loop2, _ = sentinel.check_loop("query_sensor_array", {"target": "homunculus_core"})
    is_loop3, c3 = sentinel.check_loop("query_sensor_array", {"target": "homunculus_core"})
    check("Sentinel detects 3 consecutive duplicate actions", (not is_loop1) and (not is_loop2) and is_loop3 and c3 == 3)

    # Test 8: System 2 Shield Clamping
    engine = System2DeliberationEngine(shield_safety_margin=55)
    plan_shield = engine.deliberate(
        thought="Save power, lower shields",
        proposed_action="adjust_plasma_shields",
        proposed_args={"intensity": 20}
    )
    check("System 2 clamps unsafe shield intensity to safety margin", plan_shield.arguments["intensity"] == 55 and plan_shield.status == "INTERVENED")

    # Test 9: System 2 Blocks Blind Warp
    plan_warp_blind = engine.deliberate(
        thought="Blind emergency jump",
        proposed_action="navigate_warp_vector",
        proposed_args={"x": 1.0, "y": 0.0, "z": 0.0}
    )
    check("System 2 prevents warp jump without verified corridor", plan_warp_blind.action != "navigate_warp_vector" and plan_warp_blind.status == "INTERVENED")

    # Test 10: Full Homunculus Nebula Escape Simulation
    pilot = HomunculusWarpPilot(max_steps=20, shield_safety_margin=60)
    sim_result = pilot.run_escape_mission(verbose=False)
    check(
        "Autonomous pilot escapes Homunculus Nebula within step budget",
        sim_result["escaped"] and sim_result["steps_used"] <= 20 and sim_result["final_state"] == "HALTED",
        f"steps={sim_result['steps_used']}, interventions={sim_result['interventions_count']}"
    )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


# =============================================================================
# 4. CLI Execution
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="AURA Subsystem 06: System 2 Warp Pilot & Deliberation Loop Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selftest", action="store_true", help="Run the 10-check mathematical & state machine verification suite")
    parser.add_argument("--sim", action="store_true", help="Run the full Homunculus Nebula escape simulation")
    parser.add_argument("--export-json", action="store_true", help="Export simulation telemetry to logs/")
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # If --sim is explicitly requested or no arguments provided, run simulation after tests
    print("=" * 85)
    print("USSE NOOSPHERE -- AURA SUBSYSTEM 06: AUTONOMOUS WARP PILOT & DELIBERATION LOOP")
    print("Galactic Waypoint: SEC-06-CAR (Eta Carinae & Homunculus Nebula, 7,500 ly from Sol)")
    print("Station Officer: Janusz (Security & Ethics Sentinel)")
    print("=" * 85)

    all_passed, checks = run_selftest()
    print("\n[SELF-TEST] Executing Subsystem 06 Verification Suite...")
    for c in checks:
        status_tag = "[PASS]" if c["passed"] else "[FAIL]"
        detail_tag = f" ({c['details']})" if c["details"] else ""
        print(f"  {status_tag} {c['name']}{detail_tag}")

    passed_count = sum(1 for c in checks if c["passed"])
    print(f"\n>> Verification Result: {passed_count}/{len(checks)} checks passed.")

    if not all_passed:
        print("[ERROR] Subsystem 06 verification failed!", file=sys.stderr)
        return 1

    # Log selftest telemetry
    test_log = {
        "mission": "06",
        "station": "Security & Ethics Sentinel",
        "officer": "Janusz",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "all_passed": all_passed,
        "checks": checks
    }
    test_path = LOGS_DIR / f"{timestamp}-ps06-selftest.json"
    test_path.write_text(json.dumps(test_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Telemetry logged to {test_path.name}\n")

    # Run Flight Simulation
    pilot = HomunculusWarpPilot(max_steps=20, shield_safety_margin=60)
    sim_data = pilot.run_escape_mission(verbose=True)

    sim_path = LOGS_DIR / f"{timestamp}-ps06-simulation.json"
    sim_path.write_text(json.dumps(sim_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Flight simulation telemetry saved to: {sim_path.name}")
    print("[SUCCESS] Subsystem 06 Autonomous Warp State Machine verified and ready for ship deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
