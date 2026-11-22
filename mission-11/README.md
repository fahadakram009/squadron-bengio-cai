# Mission 11: Andromeda Swarm Nexus — Multi-Agent Topologies, Handoff Contracts & Deadlock Defense

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 11 — Multi-Agent Systems (MAS) & Framework Topologies  
**Waypoint:** Sector `SEC-11-M31`: Andromeda Galaxy Core (`Messier 31` / `NGC 224` — Double Nucleus `P1` & `P2`, `2.50 Million ly` from Sol)  
**AURA Subsystem Repair:** Subsystem 11: Andromeda Swarm Fleet Coordination Nexus (`+7.1%` $\rightarrow$ `78.1%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Google Antigravity (`agy` / `run_workflow`) | Python 3.11+ & Google ADK 2.0 (`google-adk` / `adk web`)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter/reference modules, verification harnesses, live multi-agent studio templates (Google ADK 2.0 & Antigravity), and the final documentation template for **Mission 11**:

```text
mission-11/
├── README.md                         <- Mission overview, live studio quick-start & Moodle PDF guide
├── ps11_briefing.md                  <- Full mission specification, mathematical equations & instructions
├── swarm_state_graph_architect.py    <- Part 1: Systems Architect (Hybrid State-Graph, Reducers & Amdahl Speedup)
├── handoff_contract_protocol.py      <- Part 2: Protocol Officer (Typed Handoff Envelopes & Least-Context Gate)
├── blackboard_debate_consensus.py    <- Part 3: Data Specialist (Versioned OCC Blackboard & Condorcet Quorum)
├── deadlock_byzantine_sentinel.py    <- Part 4: Security Sentinel (3-Color DFS Cycle Detector & Byzantine Firewall)
├── andromeda_crew/                   <- Part 5 (Track A): Google ADK 2.0 (`adk web`) Multi-Agent Studio Package
│   ├── __init__.py
│   └── agent.py
├── andromeda_branch_swarm.py         <- Part 5 (Track B): Google Antigravity `run_workflow` Branch Swarm Script
└── mission-11-doc.md                 <- Part 5: Final Squadron Documentation Template (Convert to PDF for Moodle!)
```

---

## Station Role Deliverables Checklist

Before compiling your final documentation and submitting to Moodle, ensure all four station officers have implemented and verified their modules and completed one Live Studio Showcase track:

- [ ] **Part 1 — Systems Architect (`swarm_state_graph_architect.py`):** Implement `SwarmStateGraph` (`OVERWRITE`, `APPEND_LIST`, `SUM_NUMERIC`, and `MAX_CONFIDENCE` reducer channels, sequential/conditional/parallel fan-out execution with `max_steps` protection, and critical-path Agentic Amdahl speedup $S = T_{\text{seq}} / T_{\text{crit}}$).
- [ ] **Part 2 — Communications & Protocol Officer (`handoff_contract_protocol.py`):** Implement `HandoffEnvelope` and `ContextScopingGate` (Least-Context projection onto `allowed_context_keys`, `SUBROUTINE_TOOL` caller return-stack vs. `PEER_TRANSFER` active-agent handoff, `required_output_fields` schema validation, `HOP_LIMIT_EXCEEDED` protection, and context compression ratio $\eta_{\text{comp}}$).
- [ ] **Part 3 — Data & Payload Specialist (`blackboard_debate_consensus.py`):** Implement `VersionedBlackboard` (role-scoped ACLs and Optimistic Concurrency Control raising `CONCURRENCY_CONFLICT` on stale `expected_version`) and `DebateConsensusAggregator` (exact Condorcet majority probability $P_{\text{maj}}(N, p)$, Log-Odds reliability-weighted quorum $w_i = \ln(p_i/(1-p_i))\cdot c_i$, and correlated echo-chamber detection at $\rho \ge 0.50$).
- [ ] **Part 4 — Security & Ethics Sentinel (`deadlock_byzantine_sentinel.py`):** Implement `WaitForGraphMonitor` ($\mathcal{O}(|\mathcal{V}| + |\mathcal{E}_W|)$ 3-Color DFS `WHITE`/`GRAY`/`BLACK` cycle extraction accepting diamond DAGs while catching circular wait cycles), `SupervisorDeadlockArbiter` (deterministic priority tie-breaking), and `ByzantineSwarmFirewall` (robust MAD z-score $> 3.5$ quarantine + swarm token budget circuit breaker).
- [ ] **Part 5 — All 4 Squadron Officers (Live Studio Showcase + `mission-11-doc.md` $\rightarrow$ `mission-11-doc.pdf`):** Run **Track A** (Google ADK 2.0 `adk web` on `andromeda_crew/`) **OR Track B** (Google Antigravity `run_workflow` / `/teamwork` on `andromeda_branch_swarm.py`), embed your live execution trace screenshot in `mission-11-doc.md`, export to **`mission-11-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Quick-Start: Part 5 Live Multi-Agent Studio Showcase

### Track A: Google ADK 2.0 (`adk web` Visual Trace Studio)
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1  |  macOS/Linux: source .venv/bin/activate
pip install google-adk
# Set your free Google AI Studio API key:
# PowerShell: $env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# Bash/Zsh:   export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

cd mission-11
adk web
```
Open `http://localhost:8000`, select **`andromeda_crew`**, submit an `SEC-11-M31` orbital navigation prompt (e.g., *"Compute and safety-audit a relativistic clearance trajectory between Andromeda nuclei P1 and P2 around M31*"*), and capture a screenshot of the **Events / Graph Trace** panel.

### Track B: Google Antigravity (`run_workflow` Branch Swarm or `/teamwork`)
Open this `mission-11/` workspace inside **Google Antigravity (`agy`)** and execute `andromeda_branch_swarm.py` via the `run_workflow` tool (or trigger `/teamwork` across the 4 station modules). Capture a screenshot of the **Antigravity Subagents Auxiliary Pane** displaying parallel branch-isolated verifier cards and schema-validated outputs.

---

## Post-Implementation Documentation & Moodle Submission Workflow

### Step 1: Complete `mission-11-doc.md`
Open [`mission-11-doc.md`](./mission-11-doc.md) in this directory and complete:
1. **Squadron Roster & Station Sign-Off Table**
2. **Integrated Multi-Agent Topology & Deadlock Defense Diagram**
3. **Station-by-Station Experimental Reports (Sections A–D) & Live Studio Showcase (Section E)**
4. **End-to-End Verification Terminal Output** (from running all 4 Python modules)
5. **Squadron Seminar Reflection** (Least-Context Principle, State-Graph Reducers, and Correlated Echo Chambers)

### Step 2: Convert `mission-11-doc.md` to `mission-11-doc.pdf`
- **Option A — VS Code (Recommended):** Install *Markdown PDF* (`yzane.markdown-pdf`), open `mission-11-doc.md`, right-click, and select **Markdown PDF: Export (pdf)**.
- **Option B — Pandoc + Typst CLI:**
  ```bash
  pandoc mission-11-doc.md -o mission-11-doc.pdf --pdf-engine=typst
  ```
- **Option C — Browser / GitHub Preview Print-to-PDF:** Open the rendered Markdown preview and choose **Print $\rightarrow$ Save as PDF**.

### Step 3: Submit in Moodle & USSE Noosphere Portal
1. **Moodle Submission (Required):** Upload **`mission-11-doc.pdf`** to the **Week 11 Assignment Drop-box in Moodle** before `2027-01-10`.
2. **USSE Noosphere Portal:** Push your completed repository to GitHub and transmit your public repository link under **Week 11 (Mission 11) $\rightarrow$ 4. Transmit Solution**.
