# Mission 09: The Tarantula Relay & The TorchTPU Epistemic Router

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 9 — Agentic RAG, Corrective & Self-Reflective Loops, and Hardware Acceleration on Free Google TPUs (`torch_tpu`)  
**Waypoint:** Sector `SEC-09-LMC`: Large Magellanic Cloud — Tarantula Nebula (`30 Doradus` / `NGC 2070` & `SN 1987A`, 160,000 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 09: Intergalactic Relay Router & TorchTPU Adaptive Indexer (`+7.1%` $\rightarrow$ `63.9%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ & PyTorch + Open-Source `torch_tpu` (Colab Free `TPU v5e-1` / CPU / CUDA fallback)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter code, verification harnesses, and the final documentation template for **Mission 09**:

```text
mission-09/
├── README.md                   <- Mission overview, checklist & Moodle PDF submission guide
├── ps09_briefing.md            <- Full mission specification, mathematical equations & instructions
├── adaptive_rag_tpu_router.py  <- Part 1: Systems Architect (Adaptive RAG Complexity Router & TorchTPU Dispatcher)
├── crag_selfrag_protocol.py    <- Part 2: Protocol Officer (Corrective RAG Strip Filter & Self-RAG Scorer)
├── torchtpu_hbm_indexer.py     <- Part 3: Data Specialist (TorchTPU bfloat16 ColBERT MaxSim & Contradiction Resolver)
├── route_hijack_sentinel.py    <- Part 4: Security Sentinel (Route-Hijack Firewall & Jaccard Sponge-Loop Breaker)
└── mission-09-doc.md           <- Part 5: Final Squadron Documentation Template (Convert to PDF for Moodle!)
```

---

## Station Role Deliverables Checklist

Before compiling your final documentation and submitting to Moodle, ensure all four station officers have implemented and verified their modules:

- [ ] **Part 1 — Systems Architect (`adaptive_rag_tpu_router.py`):** Implement `get_compute_device()` and `AdaptiveRAGRouter` (`route_query` dispatching across Route A / Route B / Route C, `decompose_multihop_query`, and latency/token budgeting on `torch.device("tpu")` with CPU fallback).
- [ ] **Part 2 — Communications & Protocol Officer (`crag_selfrag_protocol.py`):** Implement `CRAGEvaluator` and `SelfRAGCritiqueScorer` (`evaluate_and_action` with tri-band confidence gating at $\alpha_{\text{upper}} = 0.72$ and $\alpha_{\text{lower}} = 0.38$, `decompose_and_filter_strips`, LMC Relay query reformulation, and beam ranking over `[ISREL]`, `[ISSUP]`, and `[ISUSE]`).
- [ ] **Part 3 — Data & Payload Specialist (`torchtpu_hbm_indexer.py`):** Implement `TorchTPUHBMIndexer` and `EpistemicContradictionResolver` (`compute_maxsim` in `torch.bfloat16` via `torch.einsum("id,mjd->mij", ...)` with padding masks, and temporal/authority contradiction resolution reconciling Sanduleak -69° 202 vs. SN 1987A).
- [ ] **Part 4 — Security & Ethics Sentinel (`route_hijack_sentinel.py`):** Implement `RouteHijackSentinel` (`inspect_and_sanitize_relay_chunk` stripping Route-Hijack and critique-token spoofing injection patterns, and `check_loop_sponge_guard` enforcing $K_{\max} = 3$ and Jaccard cycle detection $J(q_t, q_k) \ge 0.85$).
- [ ] **Part 5 — All 4 Squadron Officers (`mission-09-doc.md` $\rightarrow$ `mission-09-doc.pdf`):** Complete `mission-09-doc.md` after all 4 modules pass verification, export it to **`mission-09-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Free Google TPU Workflow Guide

For Mission 09, cadets can run hardware-accelerated tensor operations for free:
1. **Option 1 (Primary Recommended — Google Colab Free Tier `TPU v5e-1`):** Change runtime type to **TPU v5e-1** (16 GB HBM), install `torch` and open-source `torch_tpu`, and verify eager execution with `torch.device("tpu")`.
2. **Option 2 (Kaggle Notebooks Free Tier — `TPU v3-8` / `TPU v5e-8`):** Multi-chip distributed tensor experiments.
3. **Option 3 (Local / CI Zero-Friction Fallback):** Scripts automatically fall back to `cuda` or `cpu` (with `bfloat16` emulation) when running outside a TPU environment.

---

## Post-Implementation Documentation & Moodle Submission Workflow (Part 5)

Once all four station officers have completed and tested Parts 1–4:

### Step 1: Complete `mission-09-doc.md`
Open [`mission-09-doc.md`](./mission-09-doc.md) in this directory and fill in all sections together as a squadron:
1. **Squadron Roster & Station Sign-Off Table** (Station Roles, Cadet Full Names, Deliverable Files, and Verification Status).
2. **Integrated Architecture Diagram** (ASCII and Mermaid diagrams depicting how router, CRAG, HBM indexer, and security sentinel interact).
3. **Station-by-Station Experimental Reports (Sections A–D)** (Routing distribution & TTFT, CRAG strip filtering & critique scores, `bfloat16` ColBERT MaxSim rankings & contradiction resolution, and route-hijack audit logs).
4. **End-to-End Integration Verification Output** (Paste terminal output from running all 4 modules).
5. **Squadron Reflection on Agentic RAG & Hardware-Accelerated Retrieval** (1–2 paragraph synthesis answering the Week 9 central question).

### Step 2: Convert `mission-09-doc.md` to `mission-09-doc.pdf`
Choose any of the following methods to generate `mission-09-doc.pdf`:
- **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-09-doc.md`, right-click inside the editor, and choose **Markdown PDF: Export (pdf)**.
- **Option B — Pandoc CLI:**
  ```bash
  pandoc mission-09-doc.md -o mission-09-doc.pdf --pdf-engine=typst
  ```
- **Option C — Browser / GitHub Preview Print-to-PDF:** Open the rendered Markdown preview in VS Code (`Ctrl+Shift+V`) or on GitHub and select **Print $\rightarrow$ Save as PDF**.

### Step 3: Submit in Moodle & USSE Noosphere Portal
1. **Moodle Submission (Required):** Upload **`mission-09-doc.pdf`** (which contains your active GitHub repository URL and all 4 role reports) to the **Week 09 Assignment Drop-box in Moodle** before the deadline.
2. **USSE Noosphere Portal:** Push your completed code and `mission-09-doc.md` / `mission-09-doc.pdf` to your public squad GitHub repository, then paste the repository URL under **Week 9 (Mission 9) $\rightarrow$ 4. Transmit Solution**.
