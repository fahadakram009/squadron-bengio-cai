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

- [x] **Part 1 — Systems Architect (`tot_solver.py`)** - Fahad **:** Implement `ThoughtNode` and `TreeOfThoughtsSolver` supporting both Breadth-First Beam Search (`bfs_beam_search`, maintaining beam width $B$) and Depth-First Backtracking Search (`dfs_backtrack_search`, pruning below threshold $v_{\text{th}}$) under a hard node-expansion compute budget `max_expansions`.
- [x] **Part 2 — Communications & Protocol Officer (`mcts_protocol.py`):** - Madhu **:** Implement `MCTSNode` and the 4-phase `MCTSEngine` (`puct_score` using $\text{PUCT}(s,a) = Q(s,a) + c_{\text{puct}} P(a\mid s)\frac{\sqrt{N(s)}}{1+N(s,a)}$, `select_best_child`, `expand`, `simulate_rollout`, and `backpropagate`).
- [x] **Part 3 — Data & Payload Specialist (`prm_evaluator.py`)** - Osaf's station, implemented by Fahad **:** Implement `ProcessRewardEvaluator` supporting step-level scoring, trajectory reductions (`min`, `prod`, `geometric_mean`), comparison against `OutcomeRewardEvaluator` on false-positive cancellation traps, and *Math-Shepherd* Monte Carlo soft/hard step label estimation.
- [x] **Part 4 — Security & Ethics Sentinel (`horizon_pruner.py`):** - Janusz **:** Implement `EventHorizonPruner` to sever hallucinated branches when $r_{\text{PRM}}(z_t) < \tau_{\text{horizon}}$ and detect/penalize adversarial PRM reward-hacking payloads (sycophantic filler phrases & artificial step-splitting inflation).
- [x] **Part 5 — All 4 Squadron Officers (`mission-07-doc.md` $\rightarrow$ `mission-07-doc.pdf`)** - completed 2026-11-22 **:** Complete `mission-07-doc.md` after all 4 modules pass verification, export it to **`mission-07-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Running the Tree-of-Thoughts solver (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-6; standard library only, nothing to install)
cd mission-07
python tot_solver.py                   # instructor harness (mock 24 km/s tree) + short Game-of-24 benchmark
python tot_solver.py --selftest        # 39 checks: node, budget, beam, depth, pruning, backtracking, Game of 24, Parts 2/3/4
python tot_solver.py --bench           # full benchmark: BFS B = 1/2/3/5 and DFS over budgets 48..192 and evaluator noise 0 / 0.25 / 0.5
python tot_solver.py --check-signoffs  # gate: Parts 1-4 ticked above
python prm_evaluator.py                # Part 3 harness + Section C tables (length penalty, ORM vs PRM, Math-Shepherd vs M)
python prm_evaluator.py --selftest     # 28 checks incl. Part 1 and Part 4 integration
```

Each run writes `logs/<timestamp>-ps07-<mode>.json`.

**What `tot_solver.py` implements** (the instructor's contract is kept: `generator_fn(node) -> List[str]`,
`evaluator_fn(thought) -> float`, one expansion per generated-and-evaluated child, `cumulative_score` =
min-link by default, `prod` optional):

1. `ThoughtNode` with `add_child`, `get_trajectory`, `get_scored_trajectory`, `is_pruned` / `is_terminal` flags.
2. `TreeOfThoughtsSolver.bfs_beam_search`: expand every beam node, keep the best `beam_width` survivors per
   depth; `dfs_backtrack_search`: descend into the best-scoring child, backtrack chronologically on dead ends.
   Both never exceed `max_expansions`, flag children below `prune_threshold` without expanding them, and keep
   an event trace (expand / prune / keep / drop / backtrack / terminal / budget) plus `stats()`.
3. A real problem: the Game of 24 with an exact arithmetic generator, a brute-force solvability oracle, and an
   evaluator with tunable Gaussian noise that stands in for an imperfect PRM.

| Game of 24, 40 puzzles, evaluator noise 0.5 | budget 48 | budget 64 | budget 96 | budget 192 |
|---|---|---|---|---|
| BFS B = 1 (greedy) | 0.20 | 0.20 | 0.20 | 0.20 |
| BFS B = 2 | 0.07 | 0.30 | 0.42 | 0.42 |
| BFS B = 3 | 0.00 | 0.12 | 0.47 | 0.47 |
| BFS B = 5 | 0.00 | 0.00 | 0.25 | 0.47 |
| DFS with backtracking | 0.40 | 0.47 | 0.50 | 0.50 |

With an exact evaluator every configuration reaches 1.00 at budget 192 and the greedy beam is the cheapest
(43 expansions). With a noisy evaluator the greedy beam locks onto a lucky wrong step and stays at 0.20
however much budget it gets; a beam of 3 recovers to 0.47 at about 79 expansions; DFS reaches 0.50 at about
47 expansions because it only pays for siblings when a branch dies. Token cost is proportional to expansions
and the peak KV cache of BFS grows linearly with B (B live contexts), while DFS keeps one context.

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
