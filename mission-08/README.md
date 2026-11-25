# Mission 08: Cygnus X-1 — The ReWOO-Reflexion Surgeon & MemGPT Virtual Pager

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 8 — Core Agentic Design Patterns & State Memory Systems  
**Waypoint:** Cygnus X-1 Stellar-Mass Black Hole & Veil Nebula (`SEC-08-CYG`, 7,300 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 08: ReWOO DAG Planner, Reflexion Surgeon & MemGPT Virtual Pager (`+7.1%` $\rightarrow$ `56.8%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ (Zero External Framework Bloat)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter code, verification harnesses, and the final documentation template for **Mission 08**:

```text
mission-08/
├── README.md              <- Mission overview, checklist & Moodle PDF submission guide
├── ps08_briefing.md       <- Full mission specification, mathematical equations & instructions
├── rewoo_planner.py       <- Part 1: Systems Architect (ReWOO Decoupled DAG Planner & Token Cost Profiler)
├── reflexion_surgeon.py   <- Part 2: Protocol Officer (Reflexion 3-Actor Verbal RL & Traceback Surgeon)
├── memgpt_pager.py        <- Part 3: Data Specialist (MemGPT Virtual Context Pager & Tri-Factor Retrieval)
├── sandbox_guardian.py    <- Part 4: Security Sentinel (AST Static Sandbox Verifier & MemoryGraft Defense)
└── mission-08-doc.md      <- Part 5: Final Squadron Documentation Template (Convert to PDF for Moodle!)
```

---

## Station Role Deliverables Checklist

Before compiling your final documentation and submitting to Moodle, ensure all four station officers have implemented and verified their modules:

- [x] **Part 1 — Systems Architect (`rewoo_planner.py`)** - Fahad **:** Implement `PlanStep` and `ReWOOEngine` (`parse_blueprint`, `build_dependency_levels`, `execute_dag`, and `compare_token_cost` comparing $O(K^2)$ ReAct cumulative input tokens against $O(K)$ ReWOO).
- [x] **Part 2 — Communications & Protocol Officer (`reflexion_surgeon.py`)** - Madhu **:** Implement `TracebackParser` and the 3-Actor `ReflexionSurgeon` (`parse_traceback`, `generate_verbal_reflection` with sliding episodic buffer $\Omega = 3$, `is_duplicate_patch` via SHA-256 hashing, and `run_repair_loop`).
- [ ] **Part 3 — Data & Payload Specialist (`memgpt_pager.py`):** Implement `ArchivalMemoryNode` and `MemGPTVirtualPager` (`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `check_memory_pressure` at $\tau_{\text{warn}} = 0.70$, and Generative Agents tri-factor retrieval scoring $S(m, q) = \alpha_{\text{rec}}\gamma^{\Delta t} + \alpha_{\text{imp}}\hat{I}(m) + \alpha_{\text{rel}}\cos(\mathbf{e}_m, \mathbf{e}_q)$).
- [ ] **Part 4 — Security & Ethics Sentinel (`sandbox_guardian.py`):** Implement `SandboxGuardian(ast.NodeVisitor)` (`inspect_code` blocking forbidden imports, builtins, and dunder attribute escapes, plus `audit_episodic_write` defending against MemoryGraft `arXiv:2512.16962` and OEP `arXiv:2605.18930` experience poisoning).
- [ ] **Part 5 — All 4 Squadron Officers (`mission-08-doc.md` $\rightarrow$ `mission-08-doc.pdf`):** Complete `mission-08-doc.md` after all 4 modules pass verification, export it to **`mission-08-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Running the ReWOO planner (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-7; standard library only, nothing to install)
cd mission-08
python rewoo_planner.py                   # instructor harness + parallel timing + fault/re-planning traces + cost tables
python rewoo_planner.py --selftest        # 40 checks: parsing, levels, substitution, worker, faults, solver, tokens, Parts 2/3/4
python rewoo_planner.py --check-signoffs  # gate: Parts 1-4 ticked above
```

Each run writes `logs/<timestamp>-ps08-<mode>.json`.

**What `rewoo_planner.py` implements** (the instructor's interface is kept: `parse_blueprint`,
`to_plan_steps`, `build_dependency_levels`, `execute_dag(steps, fallback_replan_fn)`, `compare_token_cost`):

1. **Planner output parsing** with validation: multi-line `Plan:` text, nested brackets in arguments, and
   errors with line numbers for malformed steps, duplicate variables, self-references and references to
   variables that do not exist.
2. **Topological levels** (Kahn by frontier) with cycle and unknown-variable detection; forward references
   are allowed since order follows dependencies, not line order.
3. **Worker** that substitutes whole variables only (`#E1` never touches `#E10`) and runs each level in
   parallel with a thread pool, keeping a per-step trace (level, tool, resolved argument, output, status,
   duration). With 50 ms sensor latency the 6-step / 3-level DAG runs in 159 ms against 323 ms serial.
4. **Runtime faults**: an `ERR_` output marks the step; without a fallback its dependants are skipped as
   `ERR_UPSTREAM`; with `fallback_replan_fn` the engine takes a replacement value or, when the fallback
   returns blueprint lines, splices the new steps into the DAG and resolves the failed variable to the
   fragment's last step (dynamic re-planning: `#E1` fails, `#E10 -> #E11` are added, `#E3` waits for `#E11`).
5. **Solver stage**: one final prompt with the plan and every resolved `#E`, and `solve()` with a pluggable
   solver function.
6. **Token profiler**: the closed forms of the briefing plus a mock model that counts the tokens of every
   prompt it receives in a simulated ReAct loop and ReWOO run; the measurement equals the formula exactly.

| K | ReAct tokens | ReWOO tokens | savings |
|---|---|---|---|
| 1 | 1,200 | 2,800 | -133 % |
| 2 | 2,800 | 3,200 | -14 % |
| 3 | 4,800 | 3,600 | 25 % |
| 5 | 10,000 | 4,400 | 56 % |
| 8 | 20,800 | 5,600 | 73 % |
| 16 | 67,200 | 8,800 | 87 % |

ReAct is quadratic (constant second difference 400 tokens), ReWOO linear. ReWOO pays two base prompts
(planner and solver) so it is more expensive for one or two tool calls and wins from K = 3 whatever the
token sizes.

---

## Post-Implementation Documentation & Moodle Submission Workflow (Part 5)

Once all four station officers have completed and tested Parts 1–4:

### Step 1: Complete `mission-08-doc.md`
Open [`mission-08-doc.md`](./mission-08-doc.md) in this directory and fill in all 5 sections together as a squadron:
1. **Squadron Roster & Station Sign-Off Table** (Station Roles, Cadet Full Names, Deliverable Files, and Verification Status).
2. **Integrated Architecture Diagram** (Review/customize the ASCII or Mermaid diagram showing how `rewoo_planner.py`, `reflexion_surgeon.py`, `memgpt_pager.py`, and `sandbox_guardian.py` interact).
3. **Station-by-Station Experimental Reports (Sections A–D)** (Record your ReAct vs. ReWOO token tables, Reflexion repair trajectories, MemGPT pressure interrupt & $S(m,q)$ tables, and AST / MemoryGraft security audit logs).
4. **End-to-End Integration Verification Output** (Paste the terminal output from running all 4 modules).
5. **Squadron Reflection on Agentic Execution & State Memory Systems** (1–2 paragraph synthesis answering the Week 8 central question).

### Step 2: Convert `mission-08-doc.md` to `mission-08-doc.pdf`
Choose any of the following methods to generate `mission-08-doc.pdf`:
- **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-08-doc.md`, right-click inside the editor, and choose **Markdown PDF: Export (pdf)**.
- **Option B — Pandoc CLI:**
  ```bash
  pandoc mission-08-doc.md -o mission-08-doc.pdf --pdf-engine=xelatex
  ```
- **Option C — Browser / GitHub Preview Print-to-PDF:** Open the rendered Markdown preview in VS Code (`Ctrl+Shift+V`) or on GitHub and select **Print $\rightarrow$ Save as PDF**.

### Step 3: Submit in Moodle & USSE Noosphere Portal
1. **Moodle Submission (Required):** Upload **`mission-08-doc.pdf`** (which contains your active GitHub repository URL and all 4 role reports) to the **Week 08 Assignment Drop-box in Moodle** before the deadline.
2. **USSE Noosphere Portal:** Push your completed code and `mission-08-doc.md` / `mission-08-doc.pdf` to your public squad GitHub repository, then paste the repository URL under **Week 8 (Mission 8) $\rightarrow$ 4. Transmit Solution**.
