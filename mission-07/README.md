# Mission 07: Sagittarius A* — The Gravitational Search Horizon & Process Reward Evaluator

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 7 — Test-Time Compute & Reasoning Search Paradigms  
**Waypoint:** Sagittarius A* Supermassive Black Hole (`SEC-07-SGR`, 26,670 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator (`+7.1%` $\rightarrow$ `49.7%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ (Zero External Framework Bloat)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter code, verification harnesses, and the final documentation template for **Mission 07**:

```text
mission-07/
├── README.md              <- Mission overview, checklist & Moodle PDF submission guide
├── ps07_briefing.md       <- Full mission specification, mathematical equations & instructions
├── tot_solver.py          <- Part 1: Systems Architect (Tree-of-Thoughts BFS Beam & DFS Backtracking)
├── mcts_protocol.py       <- Part 2: Protocol Officer (4-Phase MCTS & PUCT Selection Engine)
├── prm_evaluator.py       <- Part 3: Data Specialist (Process Reward Evaluator & Math-Shepherd Rollouts)
├── horizon_pruner.py      <- Part 4: Security Sentinel (Event Horizon Pruner & Reward-Hacking Defense)
└── mission-07-doc.md      <- Part 5: Final Squadron Documentation Template (Convert to PDF for Moodle!)
```

---

## Station Role Deliverables Checklist

Before compiling your final documentation and submitting to Moodle, ensure all four station officers have implemented and verified their modules:

- [ ] **Part 1 — Systems Architect (`tot_solver.py`):** Implement `ThoughtNode` and `TreeOfThoughtsSolver` supporting both Breadth-First Beam Search (`bfs_beam_search`, maintaining beam width $B$) and Depth-First Backtracking Search (`dfs_backtrack_search`, pruning below threshold $v_{\text{th}}$) under a hard node-expansion compute budget `max_expansions`.
- [ ] **Part 2 — Communications & Protocol Officer (`mcts_protocol.py`):** Implement `MCTSNode` and the 4-phase `MCTSEngine` (`puct_score` using $\text{PUCT}(s,a) = Q(s,a) + c_{\text{puct}} P(a\mid s)\frac{\sqrt{N(s)}}{1+N(s,a)}$, `select_best_child`, `expand`, `simulate_rollout`, and `backpropagate`).
- [ ] **Part 3 — Data & Payload Specialist (`prm_evaluator.py`):** Implement `ProcessRewardEvaluator` supporting step-level scoring, trajectory reductions (`min`, `prod`, `geometric_mean`), comparison against `OutcomeRewardEvaluator` on false-positive cancellation traps, and *Math-Shepherd* Monte Carlo soft/hard step label estimation.
- [ ] **Part 4 — Security & Ethics Sentinel (`horizon_pruner.py`):** Implement `EventHorizonPruner` to sever hallucinated branches when $r_{\text{PRM}}(z_t) < \tau_{\text{horizon}}$ and detect/penalize adversarial PRM reward-hacking payloads (sycophantic filler phrases & artificial step-splitting inflation).
- [ ] **Part 5 — All 4 Squadron Officers (`mission-07-doc.md` $\rightarrow$ `mission-07-doc.pdf`):** Complete `mission-07-doc.md` after all 4 modules pass verification, export it to **`mission-07-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Post-Implementation Documentation & Moodle Submission Workflow (Part 5)

Once all four station officers have completed and tested Parts 1–4:

### Step 1: Complete `mission-07-doc.md`
Open [`mission-07-doc.md`](./mission-07-doc.md) in this directory and fill in all 5 sections together as a squadron:
1. **Squadron Roster & Station Sign-Off Table** (Station Roles, Cadet Full Names, Deliverable Files, and Verification Status).
2. **Integrated Architecture Diagram** (Review/customize the ASCII or Mermaid diagram showing how `tot_solver.py`, `mcts_protocol.py`, `prm_evaluator.py`, and `horizon_pruner.py` interact).
3. **Station-by-Station Experimental Reports (Sections A–D)** (Record your empirical tables, PUCT traces, ORM vs. PRM comparisons, and Event Horizon pruning logs).
4. **End-to-End Integration Verification Output** (Paste the terminal output from running all 4 modules).
5. **Squadron Reflection on Test-Time Compute Scaling** (1-paragraph synthesis answering the Week 7 central question).

### Step 2: Convert `mission-07-doc.md` to `mission-07-doc.pdf`
Choose any of the following methods to generate `mission-07-doc.pdf`:
- **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-07-doc.md`, right-click inside the editor, and choose **Markdown PDF: Export (pdf)**.
- **Option B — Pandoc CLI:**
  ```bash
  pandoc mission-07-doc.md -o mission-07-doc.pdf --pdf-engine=xelatex
  ```
- **Option C — Browser / GitHub Preview Print-to-PDF:** Open the rendered Markdown preview in VS Code (`Ctrl+Shift+V`) or on GitHub and select **Print $\rightarrow$ Save as PDF**.

### Step 3: Submit in Moodle & USSE Noosphere Portal
1. **Moodle Submission (Required):** Upload **`mission-07-doc.pdf`** (which contains your active GitHub repository URL and all 4 role reports) to the **Week 07 Assignment Drop-box in Moodle** before the deadline.
2. **USSE Noosphere Portal:** Push your completed code and `mission-07-doc.md` / `mission-07-doc.pdf` to your public squad GitHub repository, then paste the repository URL under **Week 7 (Mission 7) $\rightarrow$ 4. Transmit Solution**.
