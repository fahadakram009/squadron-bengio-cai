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

- [ ] **Systems Architect (`01_Agent_State_Machine_starter.py`):** Formal `AgentState` enum, `@dataclass Agent`, deterministic graph transitions, and hard step budget enforcement ($N \le 20$).
- [ ] **Communications & Protocol Officer (`02_Tool_Protocol_Dispatcher_starter.py`):** Ship navigation tools (`navigate_warp_vector`, `adjust_plasma_shields`, `query_sensor_array`), JSON-Schema validation, and exception boundary wrapping.
- [ ] **Data & Payload Specialist (`03_Infinite_Loop_Sentinel_starter.py`):** SHA-256 state history hashing and cycle detector preventing infinite action-observation loops.
- [x] **Security & Ethics Sentinel (`04_System2_Warp_Pilot_starter.py` / `04_System2_Warp_Pilot.py`)** - Janusz **:** Integrated System 2 deliberation engine successfully navigating the ship out of the Homunculus Nebula.

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 6 (Mission 6) -> Problem Set ps06 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
