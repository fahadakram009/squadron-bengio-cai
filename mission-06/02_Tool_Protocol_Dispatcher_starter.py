"""
Mission 06 - Part 2: Communications & Protocol Officer (Madhu)
Tool Protocol Contracts, Schema Validation & Exception Boundaries

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula)
Subsystem 06: Autonomous Warp State Machine & Deliberation Loop

- tools: navigate_warp_vector, adjust_plasma_shields, query_sensor_array
- every call is checked against a JSON schema before the tool runs
- any error comes back as "Tool Execution Error: ..." (never a crash), goes into the agent's
  memory, and the agent can fix its call in the next step
"""

import json
import re
from typing import Any, Callable, Dict, List
import importlib.util
from pathlib import Path


# --- tools ---

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
        "escape_corridor": "Clear gravitational corridor detected along vector [-0.85, 0.52, 0.05].",
    }
    return known_targets.get(target.lower(), f"Sensor array reports: Target '{target}' unresolved.")


TOOL_REGISTRY: Dict[str, Callable] = {
    "navigate_warp_vector": navigate_warp_vector,
    "adjust_plasma_shields": adjust_plasma_shields,
    "query_sensor_array": query_sensor_array,
}

# --- JSON schemas ---

TOOL_SCHEMAS = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        "required": ["x", "y", "z"],
        "additionalProperties": False,
    },
    "adjust_plasma_shields": {
        "type": "object",
        "properties": {"intensity": {"type": "integer"}},     # the 0..100 range is checked by the tool
        "required": ["intensity"],
        "additionalProperties": False,
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {"target": {"type": "string", "minLength": 1}},
        "required": ["target"],
        "additionalProperties": False,
    },
}


def type_ok(value: Any, t: str) -> bool:
    # True/False are not numbers in JSON (in Python bool is a subclass of int)
    if isinstance(value, bool):
        return t == "boolean"
    if t == "number":
        return isinstance(value, (int, float))
    if t == "integer":
        return isinstance(value, int)
    if t == "string":
        return isinstance(value, str)
    if t == "object":
        return isinstance(value, dict)
    return False


def validate(arguments: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Returns a list of problems (empty list = valid)."""
    errors = []
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in arguments:
            errors.append(f"Missing required argument '{req}'")
    for key, value in arguments.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                errors.append(f"Unexpected argument '{key}'")
            continue
        rule = props[key]
        if not type_ok(value, rule["type"]):
            errors.append(f"'{key}' must be {rule['type']}, got {type(value).__name__} {value!r}")
        elif isinstance(value, str) and len(value) < rule.get("minLength", 0):
            errors.append(f"'{key}' must not be empty")
    return errors


# --- dispatcher with error boundary ---

ERROR = "Tool Execution Error: "


class ActionDispatcher:
    @staticmethod
    def dispatch(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Runs one tool call. Always returns a string, never raises."""
        try:
            if tool_name not in TOOL_REGISTRY:
                return f"{ERROR}UnknownTool: '{tool_name}'. Available: {', '.join(TOOL_REGISTRY)}."
            if not isinstance(arguments, dict):
                return f"{ERROR}SchemaError: arguments must be a JSON object."
            problems = validate(arguments, TOOL_SCHEMAS[tool_name])
            if problems:
                return f"{ERROR}SchemaError: " + "; ".join(problems) + f" for '{tool_name}'."
            result = TOOL_REGISTRY[tool_name](**arguments)
            return f"Observation: {result}"
        except Exception as e:                     # error boundary: the error goes back to the agent as text
            return f"{ERROR}{type(e).__name__}: {e}"

    @staticmethod
    def dispatch_json(payload: str) -> str:
        """Takes the model's raw output, e.g. '{"tool": "...", "arguments": {...}}'."""
        try:
            call = json.loads(payload)
        except json.JSONDecodeError as e:
            return f"{ERROR}JSONDecodeError: {e.msg} at column {e.colno}. Send valid JSON."
        if not isinstance(call, dict):
            return f"{ERROR}SchemaError: a tool call must be a JSON object."
        return ActionDispatcher.dispatch(call.get("tool"), call.get("arguments", {}))


def record_observation(memory: List[Dict[str, Any]], observation: str) -> None:
    """Puts the result into the agent's memory so the next deliberation step can read it."""
    memory.append({"role": "tool", "content": observation, "ok": not observation.startswith(ERROR)})


# --- self-correction demo (uses Part 1's state machine) ---

def load_part1():
    path = Path(__file__).resolve().parent / "01_Agent_State_Machine_starter.py"
    spec = importlib.util.spec_from_file_location("part1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent, mod.AgentState


def scripted_model(memory: List[Dict[str, Any]]) -> str:
    """Stand-in for the LLM: tries 150% shields, then reads the valid range from the error and retries."""
    if not memory:
        return json.dumps({"tool": "query_sensor_array", "arguments": {"target": "escape_corridor"}})
    last = memory[-1]["content"]
    if "corridor" in last:
        return json.dumps({"tool": "adjust_plasma_shields", "arguments": {"intensity": 150}})
    m = re.search(r"Valid range: \[(\d+), (\d+)\]", last)
    if m:
        return json.dumps({"tool": "adjust_plasma_shields", "arguments": {"intensity": int(m.group(2))}})
    return json.dumps({"tool": "navigate_warp_vector", "arguments": {"x": -0.85, "y": 0.52, "z": 0.05}})


def run_mission(verbose: bool = True) -> Dict[str, Any]:
    Agent, AgentState = load_part1()
    agent = Agent(max_steps=20)
    payload, observation = None, None
    while agent.state != AgentState.HALTED:
        if agent.state == AgentState.DELIBERATING:
            payload = scripted_model(agent.memory)
        elif agent.state == AgentState.ACTING:
            observation = ActionDispatcher.dispatch_json(payload)
            if verbose:
                print(f"  step {agent.step_count:02d}  {payload}\n           -> {observation}")
        elif agent.state == AgentState.OBSERVING:
            record_observation(agent.memory, observation)
        elif agent.state == AgentState.REFLECTING:
            agent.mission_completed = observation.startswith("Observation: Warp")
        agent.step()
    errors = sum(not m["ok"] for m in agent.memory)
    if verbose:
        print(f"  HALTED after {agent.step_count} steps, mission completed: {agent.mission_completed}, errors fed back: {errors}")
    return {"steps": agent.step_count, "completed": agent.mission_completed, "errors": errors}


# --- tests ---

def protocol_dispatch_test():
    """Master test 2 from the brief: an invalid JSON payload returns an error string and does not crash."""
    out = ActionDispatcher.dispatch_json('{"tool": "adjust_plasma_shields", "arguments": {"intensity": 85')
    assert isinstance(out, str) and out.startswith("Tool Execution Error: JSONDecodeError")


def run_tests() -> int:
    d = ActionDispatcher.dispatch
    tests = [
        ("valid call returns an observation", d("query_sensor_array", {"target": "escape_corridor"}).startswith("Observation:")),
        ("tool ValueError is caught (shields 150%)", d("adjust_plasma_shields", {"intensity": 150}).startswith(ERROR + "ValueError")),
        ("missing required argument rejected", "Missing required argument 'z'" in d("navigate_warp_vector", {"x": 1.0, "y": 0.0})),
        ("wrong type rejected ('85' as string)", "must be integer" in d("adjust_plasma_shields", {"intensity": "85"})),
        ("true is not accepted as a number", "must be integer" in d("adjust_plasma_shields", {"intensity": True})),
        ("extra argument rejected", "Unexpected argument 'warp_factor'" in d("navigate_warp_vector", {"x": 1, "y": 0, "z": 0, "warp_factor": 9})),
        ("unknown tool rejected", d("fire_torpedoes", {}).startswith(ERROR + "UnknownTool")),
    ]
    try:
        protocol_dispatch_test()
        tests.append(("Protocol Dispatch Test: broken JSON -> error string, no crash", True))
    except AssertionError:
        tests.append(("Protocol Dispatch Test: broken JSON -> error string, no crash", False))
    res = run_mission(verbose=False)
    tests.append(("agent self-corrects through Part 1's state machine (150% -> 100%)",
                  res["completed"] and res["errors"] == 1 and res["steps"] <= 20))
    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = sum(not ok for _, ok in tests)
    print(f"--- {len(tests) - failed} passed, {failed} failed ---\n")
    return failed


if __name__ == "__main__":
    print("=== Part 2 tests ===")
    failed = run_tests()
    print("=== Self-correction run ===")
    run_mission()
    raise SystemExit(1 if failed else 0)
