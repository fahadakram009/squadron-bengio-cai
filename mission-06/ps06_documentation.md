# Mission 6: The Autonomous Warp State Machine and Deliberation Loop

**Solution documentation, Squadron Bengio**

- Course: Conversational AI: History, Applications, Future (BTU Cottbus-Senftenberg, WS 2026/27)
- Problem set: ps06, Week 6 (The Agentic Shift: System 1 vs. System 2 Thinking)
- Squadron: Bengio (Moodle Group 11)
- Repository: https://github.com/fahadakram009/squadron-bengio-cai (folder `mission-06`)
- Date: 20 November 2026

## 1. Mission objective

In the stellar winds of Eta Carinae AURA's Subsystem 06 has collapsed into System 1 reflex processing: fast, greedy actions without lookahead, repeated until the plasma reserves run dry. The mission rebuilds the agent loop as a deterministic, haltable System 2 machine in plain Python, with no agent framework: a formal state machine with a hard step budget, a tool protocol that validates every call against a JSON schema and turns every failure into an observation the agent can learn from, a Markov decision process logger with discounted returns and an empirical transition model, and a deliberation engine that audits each reflex against physical safety invariants and trips a circuit breaker on repeated actions. Every deliverable is a script with its own self-test; nothing needs a GPU, a network or an LLM API.

## 2. Crew and stations

| Station | Officer | Deliverable | Status |
|---|---|---|---|
| Systems Architect (Lead Engineer) | Fahad Akram | 01_Agent_State_Machine_starter.py, README, logs | signed off |
| Communications and Protocol Officer | Madhurima Lakshmi Prabha | 02_Tool_Protocol_Dispatcher_starter.py | signed off |
| Data and Payload Specialist | Osaf Ahmad | 03_MDP_Trajectory_Logger.py | signed off |
| Security and Ethics Sentinel | Janusz Manczak | 04_System2_Warp_Pilot_starter.py, 04_System2_Warp_Pilot.md | signed off |

Each officer kept the station held since Mission 1. The stations are chained at run time: the Protocol Officer's self-correction demo and the Sentinel's flight simulation both load the Architect's `Agent` and `AgentState` from disk and drive the loop through them, the Sentinel's pilot also loads the dispatcher and the loop sentinel, and the Architect's self-test loads the other three scripts and checks the contracts they rely on.

## 3. Repository layout and how to run

| File | Content |
|---|---|
| README.md | mission overview, run instructions, Architect's results, sign-off checklist |
| ps06_briefing.md | the instructor's mission specification |
| 01_Agent_State_Machine_starter.py | Station 01: AgentState, transition graph, Agent dataclass, step budget, graph validation, exhaustive model check, 42-check self-test |
| 02_Tool_Protocol_Dispatcher_starter.py | Station 02: three ship tools, JSON-schema validator, dispatcher with error boundary, self-correction run, 9 checks |
| 03_MDP_Trajectory_Logger.py | Station 03: trajectory logger, discounted return, returns-to-go, transition frequency matrix, assert-based self-test |
| 03_Infinite_Loop_Sentinel_starter.py | the instructor's SHA-256 loop sentinel, used unchanged by the Sentinel's pilot |
| 04_System2_Warp_Pilot_starter.py, 04_System2_Warp_Pilot.md | Station 04: System 2 deliberation engine, Homunculus escape simulation, 10 checks, written audit |
| logs/*.json | committed demo, self-test and gate evidence from the Architect's script |

Running the scripts (standard library only; the instructor's starter names are kept where the grader's checklist uses them):

```
cd mission-06
python 01_Agent_State_Machine_starter.py --selftest        # 42 checks + integration
python 01_Agent_State_Machine_starter.py                   # demo trace
python 02_Tool_Protocol_Dispatcher_starter.py              # 9 checks + demo run
python 03_MDP_Trajectory_Logger.py                         # asserts + tables
python 04_System2_Warp_Pilot_starter.py                    # 10 checks + simulation
python 01_Agent_State_Machine_starter.py --check-signoffs  # gate: all four boxes ticked
```

## 4. Station 01: Agent state machine core and step budget (Fahad Akram)

The script keeps the starter's public surface, `Agent(max_steps=...)` with `state`, `step_count`, `max_steps`, `memory`, `mission_completed` and `step()`, because the other three stations read and write exactly those names. Behind it the machine is specified as data and verified rather than merely coded.

### 4.1 Implementation

`AgentState` is the seven-state enum of the briefing in graph order, and `HALTED` is the only terminal state. The transition graph is a dictionary from state to its tuple of successors, and `next_state(state, mission_completed)` is a pure, total function over it: every state has exactly one successor for a given mission flag, `REFLECTING` is the only branch point (back to `DELIBERATING` while the goal is pending, to `HALTED` when it is reached), and `HALTED` maps to itself. `Agent` is a standard dataclass. Its `step()` executes exactly one transition and returns the pair of previous and new state; every transition is appended to an auditable trace with its step number and reason, a perception passed to `step()` is stored in memory with the step it arrived at, and `halt(reason)` forces `HALTED` from any state as the entry point for the Sentinel's emergency tripwire.

The step budget is enforced twice. At construction a budget outside 1 to 20 is refused, so no agent can be given more than the briefing's ceiling. At run time a `step()` with no budget left forces `HALTED`, records the reason and does not count, so `step_count` can never exceed `max_steps`. With the default budget and a goal that is never reached the machine makes 20 counted transitions and the 21st call halts it, which is the briefing's first master test. A halted agent is absorbing: further calls change nothing.

### 4.2 Verification

`validate_graph()` checks the static properties: every state has an entry, `HALTED` has no successors, `REFLECTING` is the only branch, all seven states are reachable from `IDLE`, `HALTED` is reachable from every state, and `next_state` stays on graph edges. `model_check_budget()` is an exhaustive check of the dynamic property. The only free choice in the machine is the decision at each `REFLECTING` visit, and a run of at most N steps visits `REFLECTING` at steps 5, 9, 13 and 17, so enumerating all sequences of those decisions for every N from 1 to 20 covers every possible run. All 124 runs halt within N counted steps and N + 1 calls, and every traced edge lies on the graph.

| Scenario | Result |
|---|---|
| default budget, goal never reached, 21 calls (master test 1) | HALTED, 20 counted steps, reason "step budget 20 exhausted" |
| budget 5, eight calls (the Sentinel's own check) | HALTED, 5 counted steps |
| goal set at the second REFLECTING | HALTED at step 10, reason "mission completed" |
| halt("emergency tripwire") from ACTING at step 3 | HALTED, step not counted, forced transition in the trace |
| Agent(max_steps=21), Agent(max_steps=0), Agent(max_steps=5.5) | refused at construction |

The self-test passes 42 checks: the enum, the graph properties, single and chained transitions, the branch, the budget in both forms, absorption, the tripwire, memory and trace, serialisation, a Mermaid export of the graph, the model check, and the integration checks of section 8.

## 5. Station 02: Tool protocol and action dispatcher (Madhurima Lakshmi Prabha)

The three ship tools of the briefing are kept as in the starter, and the schemas are tightened: every schema sets `additionalProperties` to false, the sensor target must be non-empty, and the shield intensity is typed integer while the 0 to 100 range is left to the tool itself so that the error boundary is exercised. A small validator implements the subset of JSON Schema the contracts need, with one deliberate detail: a Python boolean is not accepted as a number, because `True` is an `int` subclass in Python but not a number in JSON. The validator returns a list of problems rather than raising, and `ActionDispatcher.dispatch` turns an unknown tool, a non-object argument payload, any schema problem and any exception from the tool into a string beginning with "Tool Execution Error". It never raises. `dispatch_json` takes the model's raw text, so a truncated or malformed JSON payload also comes back as an error observation naming the column, which is the briefing's second master test. `record_observation` writes the result into the agent's memory tagged with whether it succeeded.

The self-correction run closes the loop with the Architect's machine. A scripted stand-in for the model first queries the escape corridor, then asks for 150 percent shields; the dispatcher returns the `ValueError` text with the valid range, the model reads the range out of the observation on the next deliberation and retries with 100 percent, then aligns the warp vector. The agent halts after 18 steps with the mission completed and one error fed back. Her 9 checks pass: valid call, caught `ValueError`, missing argument, wrong type, boolean rejected as number, extra argument, unknown tool, broken JSON, and the self-correction run.

## 6. Station 03: MDP trajectory logger and empirical return evaluator (Osaf Ahmad)

The Payload Specialist delivered the briefing's Part 3. `TrajectoryLogger` records an episode as the sequence s0, a0, r0, s1, ..., sT: it starts from an initial state, `record(action, reward, next_state)` appends one step, and `finish()` seals the episode so that a later record is refused; a NaN reward is refused at the point of recording. `as_tau()` renders the flat tuple of the briefing. `discounted_return` computes G0 as the sum of gamma to the power t times r_t by Horner's rule, accumulating from the last reward backwards, which is linear in the episode length and numerically stable; gamma outside 0 to 1 is refused. `returns_to_go` gives G_t for every t with G_0 as its first entry.

The transition model is a maximum-likelihood estimate of P(s' given s, a) as the count of (s, a, s') over the count of (s, a), built from a dataset of historical jumps with sectors CALM, TURBULENT, SAFE and DESTROYED and actions WARP_SLOW, WARP_FAST and WARP_ESCAPE. The model reports the row for a state-action pair, the full matrix, a success rate over a set of absorbing success states, and a formatted table. On the shipped dataset the escape jump from a calm sector reaches SAFE with probability 0.70 and a fast jump from a turbulent sector is destroyed with probability 0.40. The self-test is assert-based in the style of the master suite: rewards 10, 0, −5, 20 at gamma 0.95 give G0 = 22.635, which is the briefing's third master test; empty episodes, gamma 0 and gamma 1 give the expected limits; the logger reproduces the tau tuple; rows of the matrix sum to one; an unseen pair has probability zero.

## 7. Station 04: System 2 deliberation engine and warp pilot (Janusz Manczak)

The Sentinel's script is the integration layer. It loads the Architect's state machine, the Protocol Officer's dispatcher and the loop sentinel from disk, with embedded copies as fallbacks, and wraps them in `HomunculusWarpPilot`. A scripted System 1 proposes four greedy reflexes in turn: lower shields to 15 percent, jump to warp at random coordinates, search for the corridor, engage warp along the corridor. `System2DeliberationEngine` audits each proposal against three invariants before it reaches the dispatcher: shield intensity must stay at or above a 60 percent safety margin and is clamped if not; a warp jump is blocked unless the escape corridor has been verified through the sensor array, in which case the engine substitutes the sensor query; and the warp coordinates must match the verified corridor within an L1 tolerance of 0.05. The loop sentinel hashes each action with its arguments by SHA-256 and the engine trips the circuit breaker on three identical consecutive actions.

The simulation escapes the nebula in 18 of 20 steps with two interventions, the shield clamp and the blind-warp reroute, ending in `HALTED` with the mission completed. His 10 checks pass, among them the state machine transitions, the budget halt at 5, the dispatcher's error boundary and schema check, the sentinel's deterministic hashing and triple-duplicate detection, the shield clamp, the blocked blind warp, and the full escape. His written audit explains the System 1 versus System 2 framing, the invariants, the state graph, the telemetry log of all 18 steps and a performance scorecard.

## 8. Integration across the four stations

| Interface | What is checked | Result |
|---|---|---|
| Station 02 loads Station 01 | the self-correction run drives the loop through the Architect's `Agent`, writing observations into its memory and setting the mission flag at REFLECTING | halts at 18 steps, mission completed |
| Station 04 loads Station 01 | the pilot binds the Architect's `Agent` and `AgentState` rather than its embedded fallback and flies the escape on them | escape in 18 steps, trace of 18 transitions, halt reason "mission completed" |
| Station 01 loads Station 02 | invalid payloads (out-of-range, missing argument, unknown tool) come back as "Tool Execution Error" strings, a valid call returns an observation | passes, nothing raises |
| Station 01 loads the loop sentinel | the third identical action is flagged and its hash is 64 hex characters; the verdict is wired to `halt()` and stops the agent at its third ACTING visit, step 11 | passes |
| Station 01 loads Station 04 | the pilot module loads, binds the Architect's class and completes the mission within the budget | passes |

Together the four stations are the deliberation loop of the subsystem's name: the state machine fixes the order and the budget, the dispatcher makes every action safe to attempt and every failure informative, the MDP logger scores an episode and estimates how the nebula responds to each action, and the System 2 engine with the sentinel decides which reflexes may execute and when the loop must stop.

## 9. Verification and sign-off

| Check | Result |
|---|---|
| Station 01 self-test | 42 of 42 pass with all teammates' committed files (log of 11 November) |
| Station 01 model check | 124 runs, every N in 1..20, all halt within budget |
| Station 02 self-test and run | 9 of 9 pass; self-correction run reproduced above |
| Station 03 self-test | all asserts pass; G0 = 22.635 on the master-test rewards |
| Station 04 self-test and simulation | 10 of 10 pass; escape in 18 steps |
| Sign-off gate | all four officers signed, `--check-signoffs` exits 0 (log of 20 November) |
| Scripts run with a plain python command | yes, Python 3.12, standard library only |
| Repository visibility | public |

## 10. Limitations and honest notes

- The briefing names two different Part 3 and Part 4 deliverables than the README checklist and starter files do (MDP logger and tortoise-and-hare sentinel in the briefing; loop sentinel and System 2 pilot in the README). The squad followed the briefing for Station 03 and the README for Station 04, so both the MDP logger and the deliberation pilot exist, while the instructor's loop sentinel starter is used unchanged.
- Nobody implemented Floyd's tortoise-and-hare cycle detection. The loop sentinel detects immediate repetition of one action-argument pair, which trips on the briefing's example of the same failed action three times, but would not detect a longer cycle such as s1, s2, s3 repeating. The Architect's trace and halt() provide the hook for such a detector; the algorithm itself is still open.
- The Payload Specialist's file was uploaded through the GitHub web interface, which cannot edit the README in the same step, so the Architect ticked his box after the upload landed and verified the file runs.
- The Protocol Officer's and Sentinel's loaders import the Architect's file without registering it in `sys.modules`. A dataclass with postponed annotations crashes under such a loader, so the Architect's file avoids the `from __future__ import annotations` import; the Architect's own loader registers modules first.
- The System 1 proposals in the pilot and the model in the dispatcher's demo are scripted stand-ins, not an LLM. The loop, the guards and the budget are what the mission tests; plugging in a real model only changes who proposes the action.
- The briefing's `ps06_solution.ipynb` and Canvas upload were not used; as in Missions 3 to 5 the squad keeps the instructor's `mission-06` folder, delivers scripts with self-tests, and submits the repository link on the course portal.

## 11. Links

- Squad repository: https://github.com/fahadakram009/squadron-bengio-cai
- Mission 6 folder: https://github.com/fahadakram009/squadron-bengio-cai/tree/main/mission-06
- Instructor repository: https://github.com/Unisvet/couse-c-ai
- Course site: https://dlearn-cai.web.app
