# Mission 01: The Architecture of Argument & Dialogic Protocols

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Squadron:** Bengio (Moodle Group 11, Delta Orbit)  
**Course:** Conversational AI — Winter Semester 2026/27  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Modern LLMs (Gemini, ChatGPT, Claude)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 1**.

```
mission-01/
├── README.md                              <- Mission overview & squad sign-off status
├── ps01_briefing.md                        <- Full mission specification & problem statement
├── ps01_starter.ipynb                      <- Starter Jupyter notebook for VS Code + Antigravity
├── 01_Alignment_Brief_template.md          <- Deliverable template (Historical Archivist)
├── 02_Socratic_Protocol_Map_template.md    <- Deliverable template (Protocol Specialist)
├── 03_Galilean_Sandbox_template.md         <- Deliverable template (Ethical Navigator)
└── 04_dialogic_engine_starter.py           <- Starter Python script (Core Architect)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [x] **Historical Archivist (`01_Alignment_Brief.md`)** - Osaf Ahmad **:** Dialogic alignment report comparing Socratic Discovery vs. Galilean Persuasion and grounding system prompts.
- [ ] **Protocol Specialist (`02_Socratic_Protocol_Map.md`)** - Madhu **:** Socratic elenchus state-machine flowchart and pseudocode logic.
- [x] **Ethical Navigator (`03_Galilean_Sandbox.md`)** - Janusz **:** 3-agent Galilean debate sandbox (Salviati, Simplicio, Sagredo) and AI ethics thought experiment.
- [x] **Core Architect (`04_dialogic_engine_starter.py`)** - Fahad **:** Executable Python script/notebook in VS Code + Antigravity connecting to modern LLMs (Gemini, ChatGPT, Claude).

---

## Running the engine (Core Architect)

```bash
# from the repo root
uv venv --python 3.12 .venv && source .venv/bin/activate      # or: python3 -m venv .venv
pip install -r mission-01/requirements.txt
cp .env.example .env                                          # add ONE provider key
cd mission-01
python 04_dialogic_engine_starter.py --selftest               # offline verification (mock provider)
python 04_dialogic_engine_starter.py                          # live: elenchus + Galilean debate + sycophancy probe
python 04_dialogic_engine_starter.py --mode galilean          # loads the dilemma + 3 persona system prompts from 03_Galilean_Sandbox.md
python 04_dialogic_engine_starter.py --check-signoffs         # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps01-<mode>.jsonl` (every JSON-RPC state message and LLM call with
token usage) and a `.result.json` next to it. Default LLM: Claude `claude-opus-5-5`; switch with
`LLM_PROVIDER=gemini|openai` in `.env`.

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 1 (Mission 1) -> Problem Set ps01 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
