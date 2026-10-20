# Mission 02: FST Dialogue Systems, Lovelace's Objection & Noisy Channels

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Modern LLMs (Gemini, ChatGPT, Claude)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 2**.

```
mission-02/
├── README.md                              <- Mission overview & squad sign-off status
├── ps02_briefing.md                        <- Full mission specification & problem statement
├── ps02_starter.ipynb                      <- Starter Jupyter notebook for VS Code + Antigravity
├── 01_Lovelace_Inquiry_Brief_template.md   <- Deliverable template (Historical Archivist)
├── 02_FST_and_Noisy_Protocol_template.md   <- Deliverable template (Protocol Specialist)
├── 03_Entropy_Hazard_Audit_template.md     <- Deliverable template (Ethical Navigator)
└── 04_fst_noisy_simulator_starter.py      <- Starter Python script (Core Architect)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [x] **Historical Archivist (`01_Lovelace_Inquiry_Brief.md`)** - Osaf Ahmad **:** 5-question Anti-Turing interrogation protocol & modern LLM evaluation against Lovelace's objection.
- [x] **Protocol Specialist (`02_FST_and_Noisy_Protocol.md`)** - Madhu **:** `FSTDialogueManager` class implementation (5 required states & slot memory) + Noisy Channel Protocol (7-word limit + noise tokens).
- [x] **Ethical Navigator (`03_Entropy_Hazard_Audit.md`)** - Janusz **:** FST stress tests documenting state breakage + Shannon information entropy & hazard audit under noise.
- [x] **Core Architect (`04_fst_noisy_simulator_starter.py`)** - Fahad **:** Integrated Python script/notebook in VS Code + Antigravity connecting to modern LLMs (Gemini, ChatGPT, Claude).

---

## Running the simulator (Core Architect)

```bash
# from the repo root (same .venv and .env as Mission 1)
source .venv/bin/activate && pip install -r mission-02/requirements.txt
cd mission-02
python 04_fst_noisy_simulator_starter.py --selftest              # offline verification, 21 checks
python 04_fst_noisy_simulator_starter.py --mode stress           # FST brittleness benchmark, no LLM needed
python 04_fst_noisy_simulator_starter.py --mode stress --strict  # portal-style catastrophic failure
python 04_fst_noisy_simulator_starter.py                         # live: FST demo + stress tests + channel demo + 10-turn noisy interrogation
python 04_fst_noisy_simulator_starter.py --check-signoffs        # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps02-<mode>.jsonl` (FST state trace, every LLM call with token usage) plus a
`.result.json`; a live interrogation also writes `05_Noisy_Interrogation_Transcript.md` (the squad's Part 5 deliverable).

**Integration of the four stations.** The simulator reads the other officers' files at run time:

| Station file | What the simulator takes from it | Fallback if absent |
|---|---|---|
| `01_Lovelace_Inquiry_Brief.md` (or `.pdf`) | the five Anti-Turing prompts (`Prompt: "..."` blocks, or numbered/quoted questions) open the interrogation | built-in question list |
| `02_FST_and_Noisy_Protocol.md` | the five states, slot names and the 7-word / noise-token rule the code implements | - |
| `03_Entropy_Hazard_Audit.md` | the five non-canonical utterances (quoted under a heading containing "stress") used for the brittleness benchmark | built-in stress set |

**Live run evidence.** `05_Noisy_Interrogation_Transcript.md` and `logs/20261019-234831-ps02-all.*` come from a real run
after the Protocol Specialist's fixes, opened by the Archivist's five prompts from `01_Lovelace_Inquiry_Brief.pdf` (provider
`gemini`, free-tier key: gemini-3.8-flash, then 3.5-flash and 3.1-flash-lite as daily quotas ran out, 21 LLM calls): all ten
transmissions obey the 7-word / noise-token channel, source entropy 8.12 vs transmitted 5.85 bits/word, lexical retention 0.11
but receiver-judged fidelity 0.91 and human-likeness 0.69 (0 of 10 turns judged HUMAN); the FST breaks on 4 of the
Navigator's 5 non-canonical utterances. Logs are otherwise git-ignored; these are committed deliberately.

Metrics per turn and per run: Shannon entropy H of source and transmitted word streams (bits/word), noise-token rate p,
erasure-channel capacity bound C = 1 - p, lexical retention of source content words, receiver-judged fidelity and
human-likeness, and the FST's state-sequence entropy (2.0 bits for a canonical run, higher once FALLBACK appears).

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 2 (Mission 2) -> Problem Set ps02 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
