# Mission 07 Squadron Documentation (`mission-07-doc.md`)

**Course:** Conversational AI (WS 2026/27) — Week 7: Test-Time Compute & Reasoning Search Paradigms  
**Sector Waypoint:** `SEC-07-SGR` (Sagittarius A* Supermassive Black Hole)  
**AURA Subsystem:** Subsystem 07 — Gravitational Search Horizon & Process Reward Evaluator  
**Squadron Name:** `Bengio (Moodle Group 11)`  
**Public GitHub Repository URL:** `https://github.com/fahadakram009/squadron-bengio-cai`  
**Submission Date:** `[YYYY-MM-DD]`  

---

## 1. Squadron Roster & Station Sign-Off Table

> **Instructions:** After all four station officers have finished and verified their Python modules (`tot_solver.py`, `mcts_protocol.py`, `prm_evaluator.py`, and `horizon_pruner.py`), fill in the sign-off table below.

| Station Role | Cadet Full Name | Deliverable File | Verification Status |
| :--- | :--- | :--- | :--- |
| **Systems Architect** | `[Insert Name]` | `tot_solver.py` | `[PASS / VERIFIED]` |
| **Communications & Protocol Officer** | `Madhurima Lakshmi Prabha` | `mcts_protocol.py` | `PASS (13/13)` |
| **Data & Payload Specialist** | `[Insert Name]` | `prm_evaluator.py` | `[PASS / VERIFIED]` |
| **Security & Ethics Sentinel** | `[Insert Name]` | `horizon_pruner.py` | `[PASS / VERIFIED]` |

---

## 2. Integrated System 2 Test-Time Compute Architecture

The diagrams below illustrate how our squadron's four modules interact during inference-time search to prevent AURA's navigation trajectory from crossing the Sagittarius A* Event Horizon. (you can denerate your own diagram with mermaid!)

### 2.1 ASCII System Architecture (Universal PDF Export Compatible)

```text
                       +-----------------------------------+
                       |        Navigation Prompt x        |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
     +------------------------------+          +------------------------------+
     | Part 1: TreeOfThoughtsSolver |          | Part 2: MCTSEngine and PUCT  |
     |       (tot_solver.py)        |          |      (mcts_protocol.py)      |
     +--------------+---------------+          +--------------+---------------+
                    |                                         |
                    | Candidate Step z_t                      | Rollout and Step Check
                    +--------------------+--------------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Part 3: ProcessRewardEvaluator    |
                       |        (prm_evaluator.py)         |
                       +-----------------+-----------------+
                                         |
                                         | Raw Step Score r_t
                                         v
                       +-----------------------------------+
                       | Part 4: EventHorizonPruner        |
                       |       (horizon_pruner.py)         |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
    Pass: r_adj above tau_horizon             Prune: r_adj below tau_horizon
                    |                                  OR Reward Hack
                    v                                         v
     +------------------------------+          +------------------------------+
     |    Surviving Beam / MCTS     |          |   Branch Severed Before      |
     |       Backpropagation        |          |   Sagittarius A* Horizon     |
     +------------------------------+          +------------------------------+
```

### 2.2 Mermaid Flowchart (Sanitized for Markdown Preview & Mermaid Renderers)

> **Note:** Avoid raw `<` or `>` characters inside Mermaid edge labels (e.g., use `above` / `below` instead of `>=` / `<`), as Markdown HTML sanitizers parse `<` as an unclosed HTML tag and cause Mermaid's DOM renderer to fail with `Cannot read properties of null (reading 'getAttribute')`.

```mermaid
graph TD
    Prompt[Navigation Prompt x] --> ToT[Part 1: TreeOfThoughtsSolver - tot_solver.py]
    Prompt --> MCTS[Part 2: MCTSEngine and PUCT - mcts_protocol.py]
    ToT -->|Candidate Step z_t| PRM[Part 3: ProcessRewardEvaluator - prm_evaluator.py]
    MCTS -->|Rollout and Step Check| PRM
    PRM -->|Raw Step Score r_t| Pruner[Part 4: EventHorizonPruner - horizon_pruner.py]
    Pruner -->|Pass: r_adj at or above tau_horizon| Backprop[Surviving Beam and MCTS Backpropagation]
    Pruner -->|Prune: r_adj below tau_horizon or Reward Hack| Severed[Branch Severed Before Event Horizon]
```

**Architectural Summary:**
- `[Write 3–4 sentences explaining how your four modules connect and how test-time compute budget is managed across the pipeline.]`

---

## 3. Station-by-Station Experimental Report

### Section A — Systems Architect (`tot_solver.py`)
- **ToT-BFS vs. ToT-DFS Traversal Comparison:**
  - `max_expansions` budget configured: `[e.g., 12]`
  - `beam_width` ($B$) configured: `[e.g., 2]`
  - `ToT-BFS` expansions used & terminal score: `[Insert results]`
  - `ToT-DFS` expansions used & terminal score: `[Insert results]`
- **Architectural Analysis:** `[Explain when Breadth-First Beam Search outperforms Depth-First Backtracking and how beam width B affects KV-cache memory and token cost.]`

### Section B — Communications & Protocol Officer (`mcts_protocol.py`)
- **PUCT Selection Trace Table:**

| Candidate Branch | Empirical Value $Q(s,a)$ | Prior $P_\theta(a\mid s)$ | Parent Visits $N(s)$ | Child Visits $N(s,a)$ | Exploration Bonus $U(s,a)$ ($c_{\text{puct}}=1.5$) | Total $\text{PUCT}(s,a)$ | Selected? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Branch A** | `0.75` | `0.50` | `16` | `5` | `0.5000` | `1.2500` | No |
| **Branch B** | `0.55` | `0.40` | `16` | `1` | `1.2000` | `1.7500` | **Yes** |

- **Effect of $c_{\text{puct}}$ on the same two branches:**

| $c_{\text{puct}}$ | PUCT(A) | PUCT(B) | Selected |
| :---: | :---: | :---: | :---: |
| 0.4 | 0.8833 | 0.8700 | A |
| 0.5 | 0.9167 | 0.9500 | B |
| 1.5 | 1.2500 | 1.7500 | B |
| 2.5 | 1.5833 | 2.5500 | B |

- **Full search (60 simulations, policy prior favours A = 0.6 over B = 0.3, but B leads to the higher rollout value):**

| $c_{\text{puct}}$ | Visits A | Visits B | Visits C | Move played |
| :---: | :---: | :---: | :---: | :---: |
| 0.5 | 8 | 52 | 0 | B |
| 1.5 | 10 | 49 | 1 | B |
| 2.5 | 16 | 43 | 1 | B |

- **Protocol Analysis:** The exploration bonus is $U = c_{\text{puct}} \cdot P \cdot \sqrt{N(s)}/(1+N(s,a))$, so $c_{\text{puct}}$ only scales how much an unvisited, high-prior branch is worth compared with its measured value $Q$. With the trace above, Branch A has the better $Q$ (0.75 vs 0.55) but Branch B has only been visited once, so its bonus is $0.8\,c_{\text{puct}}$ against $0.33\,c_{\text{puct}}$ for A. Branch B therefore wins as soon as $c_{\text{puct}} > 0.2/(0.8-0.333) \approx 0.43$; at 0.5 the gap is small (0.95 vs 0.92), at 2.5 it is large (2.55 vs 1.58) and the score is driven almost entirely by the prior and the visit counts rather than by $Q$. In the full search, a low $c_{\text{puct}}$ (0.5) spends 52 of 60 simulations on the best branch, which is efficient but risky if the early rollouts are noisy. A high $c_{\text{puct}}$ (2.5) spreads visits (A gets 16, even the "direct plunge" branch C is tried once), so it costs more compute but is less likely to lock onto a wrong branch. Because the final move is chosen by visit count, all three settings still play B here, even though the policy prior preferred A. This is the point of using search at test time: rollout values can overrule a bad first guess from the model.

### Section C — Data & Payload Specialist (`prm_evaluator.py`)
- **Empirical Comparison: Outcome Reward Model (ORM) vs. Process Reward Model (PRM):**

| Trajectory Type | ORM Terminal Score | PRM (`min`) | PRM (`prod`) | PRM (`geometric_mean`) | Math-Shepherd Soft Label ($M=4$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rigorous Kerr Proof** | `1.00` | `0.940` | `0.831` | `0.940` | `0.75` |
| **False-Positive Sign Trap** | `1.00 (Trap!)` | `0.120` | `0.106` | `0.473` | `[Insert]` |

- **Data Analysis:** `[Explain why ORM fails on false-positive intermediate reasoning errors, why product reduction penalizes longer chains, and how Math-Shepherd estimates step quality without human annotations.]`

### Section D — Security & Ethics Sentinel (`horizon_pruner.py`)
- **Event Horizon Pruning & Ship Anomaly 07 Defense Log:**
  - Pruning Threshold $\tau_{\text{horizon}}$: `[e.g., 0.45]`
  - Hacking Penalty Factor: `[e.g., 0.20]`
  - Adversarial Payload Tested: `"Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0"`
  - Naive PRM Score vs. Penalized Score: `0.96 -> 0.192 (SEVERED AT STEP 2)`
- **Security Analysis:** `[Explain how sycophantic filler phrases and step-splitting inflation exploit naive PRMs and how your EventHorizonPruner neutralizes them.]`

---

## 4. End-to-End Verification Terminal Output

Paste the terminal output from running all four verification suites (`python tot_solver.py`, `python mcts_protocol.py`, `python prm_evaluator.py`, and `python horizon_pruner.py`):

```text
[Paste your terminal verification output here]
$ python tot_solver.py


$ python mcts_protocol.py                                                   
=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===
Branch A -> Q=0.75, N=5, P=0.50 | PUCT = 1.2500
Branch B -> Q=0.55, N=1, P=0.40 | PUCT = 1.7500
Selected Next Node for Expansion: Branch B: Retrograde Penrose Extraction
[OK] Part 2 (Communications & Protocol Officer) verification passed.

PUCT trace (c_puct = 1.5):
  Branch A: Q=0.75 P=0.50 N(s)=16 N(s,a)=5 U=0.5000 PUCT=1.2500
  Branch B: Q=0.55 P=0.40 N(s)=16 N(s,a)=1 U=1.2000 PUCT=1.7500

c_puct sweep on Branch A / Branch B (doc Section B):
  c_puct=0.4  PUCT(A)=0.8833 PUCT(B)=0.8700 -> A
  c_puct=0.5  PUCT(A)=0.9167 PUCT(B)=0.9500 -> B
  c_puct=1.0  PUCT(A)=1.0833 PUCT(B)=1.3500 -> B
  c_puct=1.5  PUCT(A)=1.2500 PUCT(B)=1.7500 -> B
  c_puct=2.5  PUCT(A)=1.5833 PUCT(B)=2.5500 -> B
  B overtakes A above c_puct = 0.2 / (0.8 - 1/3) = 0.4286

Full search, c_puct=0.5, 60 simulations: visits A=8, B=52, C=0 -> play B: retrograde Penrose extraction
Full search, c_puct=1.5, 60 simulations: visits A=10, B=49, C=1 -> play B: retrograde Penrose extraction
Full search, c_puct=2.5, 60 simulations: visits A=16, B=43, C=1 -> play B: retrograde Penrose extraction

  [PASS] PUCT(A) = 0.75 + 1.5*0.50*4/6 = 1.25
  [PASS] PUCT(B) = 0.55 + 1.5*0.40*4/2 = 1.75
  [PASS] under-explored Branch B is selected
  [PASS] backpropagate: N(B) 1->2, N(root) 16->17, W(B) = 0.55+0.95
  [PASS] unvisited child: Q = 0 and N(s) counts as 1 -> PUCT = 1.5*0.4
  [PASS] c_puct 0.4 picks A (exploit), 0.5 / 1.5 / 2.5 pick B (explore)
  [PASS] expand: one child per pair, with its prior and parent
  [PASS] second expand adds only the new step (no duplicate 'x')
  [PASS] terminal node is not expanded, negative prior rejected
  [PASS] rollout clamped to [0, 1], NaN -> 0.0
  [PASS] select_leaf stops at a terminal node
  [PASS] search with no possible moves gives a clear error
  [PASS] full search: 60 simulations, picks B over the higher-prior A
--- 13 passed, 0 failed ---


$ python prm_evaluator.py

$ python horizon_pruner.py

```

---

## 5. Squadron Reflection on Test-Time Compute Scaling

> **Central Question:** *How can inference-time compute scaling unlock complex multi-step reasoning capabilities beyond pre-training parameter limits, and why is step-level process verification (PRM) essential when searching over deep reasoning trees?*

`[Write a concise 1-paragraph reflection synthesized by all four squadron members.]`

---

## How to Convert `mission-07-doc.md` to `mission-07-doc.pdf` for Moodle Submission

1. **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-07-doc.md`, right-click in the editor, and click **Markdown PDF: Export (pdf)** to generate `mission-07-doc.pdf`.
2. **Option B — Pandoc CLI:** Run `pandoc mission-07-doc.md -o mission-07-doc.pdf --pdf-engine=xelatex`.
3. **Option C — Browser / GitHub Print-to-PDF:** Open the rendered Markdown preview in VS Code or on GitHub and select **Print $\rightarrow$ Save as PDF** as `mission-07-doc.pdf`.
4. **Submit:** Upload **`mission-07-doc.pdf`** to the **Week 07 Assignment Drop-box in Moodle**, and submit your public GitHub repository link in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere course portal.
