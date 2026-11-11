# Mission 06: The Autonomous Warp State Machine & Deliberation Loop

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 6 — The Agentic Shift: System 1 vs. System 2 Thinking  
**Waypoint:** Eta Carinae & Homunculus Nebula (`SEC-06-CAR`, 7,500 ly from Sol)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ / PyTorch 2.4+ / Dataclasses & Enums  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 6**.

```
mission-06/
├── README.md                              <- Mission overview & squad sign-off status
├── ps06_briefing.md                        <- Full mission specification & problem statement
├── 01_Agent_State_Machine_starter.py      <- Deterministic state machine & step budget (Systems Architect)
├── 02_Tool_Protocol_Dispatcher_starter.py  <- Schema validator & tool error boundary (Protocol Officer)
├── 03_Infinite_Loop_Sentinel_starter.py    <- Cycle detection & history hashing (Data Specialist)
└── 04_System2_Warp_Pilot_starter.py       <- Deliberation loop & telemetry simulation (Security Sentinel)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [x] **Systems Architect (`01_Agent_State_Machine_starter.py`)** - Fahad **:** Formal `AgentState` enum, `@dataclass Agent`, deterministic graph transitions, and hard step budget enforcement ($N \le 20$).
- [ ] **Communications & Protocol Officer (`02_Tool_Protocol_Dispatcher_starter.py`):** Ship navigation tools (`navigate_warp_vector`, `adjust_plasma_shields`, `query_sensor_array`), JSON-Schema validation, and exception boundary wrapping.
- [ ] **Data & Payload Specialist (`03_Infinite_Loop_Sentinel_starter.py`):** SHA-256 state history hashing and cycle detector preventing infinite action-observation loops.
- [x] **Security & Ethics Sentinel (`04_System2_Warp_Pilot_starter.py` / `04_System2_Warp_Pilot.py`)** - Janusz **:** Integrated System 2 deliberation engine successfully navigating the ship out of the Homunculus Nebula.

---

## Running the agent state machine (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-5; standard library only, nothing to install)
cd mission-06
python 01_Agent_State_Machine_starter.py                   # demo: budget-forced halt at N = 20, then a mission halt at step 10
python 01_Agent_State_Machine_starter.py --selftest        # 42 checks incl. exhaustive budget model check and Parts 2, 3, 4 integration
python 01_Agent_State_Machine_starter.py --mermaid         # the transition graph as a Mermaid stateDiagram
python 01_Agent_State_Machine_starter.py --check-signoffs  # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps06-<mode>.json`.

**What `01_Agent_State_Machine_starter.py` implements** (Enum + dataclasses, no framework):

1. `AgentState`: the seven briefing states in graph order; `HALTED` is the only terminal state.
2. `TRANSITION_GRAPH` as data and `next_state(state, mission_completed)`, a pure and total transition
   function. Every state has exactly one successor for a given mission flag; `REFLECTING` is the only
   branch point (`DELIBERATING` if the goal is pending, `HALTED` if reached); `HALTED` is absorbing.
3. `Agent` dataclass with the starter's public fields (`state`, `step_count`, `max_steps`, `memory`,
   `mission_completed`) plus an auditable `trace` of `Transition` records, `halt_reason`, and `halt(reason)`
   as the entry point for the Sentinel's emergency tripwire. `step()` executes exactly one transition.
4. Step budget enforced twice: `max_steps` outside `[1, 20]` is refused at construction (hard cap N <= 20),
   and a `step()` with no budget left forces `HALTED` without counting, so `step_count` can never exceed
   `max_steps`. With N = 20 and the goal never reached: 20 counted transitions, the 21st call halts.
5. Verification: `validate_graph()` (determinism, all states reachable from `IDLE`, `HALTED` reachable from
   every state, absorbing), and `model_check_budget()`, which runs the machine over **every** sequence of
   `REFLECTING` decisions for every N in 1..20 (124 runs) and proves it always halts within N counted steps
   and N + 1 calls.

| Scenario | Result |
|---|---|
| default budget, goal never reached, 21 `step()` calls (master test 1) | `HALTED`, `step_count` = 20, reason "step budget 20 exhausted" |
| budget 5, eight calls (Sentinel's check) | `HALTED`, `step_count` = 5 |
| goal set at the second `REFLECTING` | `HALTED` at step 10, reason "mission completed" |
| `halt("emergency tripwire")` from `ACTING` | `HALTED`, step not counted, forced transition in the trace |
| `Agent(max_steps=21)` | `ValueError` |

**Integration.** The Sentinel's `04_System2_Warp_Pilot_starter.py` loads `Agent` and `AgentState` from this
file at import and flies the escape mission on it (18 steps, 2 interventions, `HALTED` with reason
"mission completed"); this file's self-test loads the Protocol Officer's dispatcher (invalid payloads come
back as `Tool Execution Error` strings, never exceptions), the Payload Specialist's sentinel (third identical
action flagged, SHA-256 hashes) and wires that verdict to `halt()` as the tripwire.

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 6 (Mission 6) -> Problem Set ps06 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
