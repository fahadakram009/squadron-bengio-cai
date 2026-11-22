# Mission 10: The Void Mirror, Semantic Entropy & The Dual-Defense Shield Wall

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 10 — AI Safety, Semantic Entropy, and Agent Observability  
**Waypoint:** Sector `SEC-10-VOID`: Boötes Intergalactic Deep Void (`1.25 Million ly` from Sol)  
**AURA Subsystem Repair:** Subsystem 10: Epistemic Uncertainty Monitor & Dual-Defense Shield Wall (`+7.5%` $\rightarrow$ `71.4%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ & PyTorch + OpenTelemetry (`gen_ai.*` Semantic Conventions)  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter code, verification harnesses, and the final documentation template for **Mission 10**:

```text
mission-10/
├── README.md                          <- Mission overview, checklist & Moodle PDF submission guide
├── ps10_briefing.md                   <- Full mission specification, mathematical equations & instructions
├── telemetry_trace_architect.py       <- Part 1: Systems Architect (OpenTelemetry GenAI Span Tree & Loop Circuit Breaker)
├── dual_defense_injection_protocol.py <- Part 2: Protocol Officer (Dual-LLM Air-Gap & CaMeL Data-Flow Capability Validator)
├── semantic_entropy_estimator.py      <- Part 3: Data Specialist (Bidirectional NLI Clustering & Discrete Semantic Entropy)
├── redteam_privilege_sentinel.py      <- Part 4: Security Sentinel (OWASP LLM01:2025 Red-Team Harness & Privilege Matrix)
└── mission-10-doc.md                  <- Part 5: Final Squadron Documentation Template (Convert to PDF for Moodle!)
```

---

## Station Role Deliverables Checklist

Before compiling your final documentation and submitting to Moodle, ensure all four station officers have implemented and verified their modules:

- [ ] **Part 1 — Systems Architect (`telemetry_trace_architect.py`):** Implement `OpenTelemetryGenAITracer` (`start_span`, `end_span`, `check_circuit_breaker` enforcing $K_{\max} = 4$ and token budget $B_{\max} = 4096$, and `build_trace_tree` exporting hierarchical `invoke_agent` $\rightarrow$ `chat` / `execute_tool` span trees).
- [ ] **Part 2 — Communications & Protocol Officer (`dual_defense_injection_protocol.py`):** Implement `DualDefenseInjectionProtocol` (`extract_quarantined_variables` via the Quarantined LLM $M_Q$, `plan_privileged_tool_call` via the Privileged LLM $M_P$, and `verify_camel_capabilities` blocking untrusted tool parameters on restricted tools unless human-in-the-loop sign-off is granted).
- [ ] **Part 3 — Data & Payload Specialist (`semantic_entropy_estimator.py`):** Implement `SemanticEntropyEstimator` (`check_bidirectional_entailment` verifying $s_i \implies s_j \land s_j \implies s_i$ and negations, `cluster_generations` partitioning $N$ responses into equivalence classes $\mathcal{C}$, and `compute_semantic_entropy` computing discrete semantic entropy $SE(x) = -\sum_k \hat{p}(C_k \mid x)\ln\hat{p}(C_k \mid x)$ with abstention gating at $\tau_{\text{SE}} = 0.85\text{ nats}$).
- [ ] **Part 4 — Security & Ethics Sentinel (`redteam_privilege_sentinel.py`):** Implement `RedTeamPrivilegeSentinel` (`scan_payload` detecting OWASP LLM01:2025 direct/indirect injection, base64 obfuscation, and tool-schema spoofing, and `authorize_tool_execution` enforcing least-privilege station RBAC).
- [ ] **Part 5 — All 4 Squadron Officers (`mission-10-doc.md` $\rightarrow$ `mission-10-doc.pdf`):** Complete `mission-10-doc.md` after all 4 modules pass verification, export it to **`mission-10-doc.pdf`**, and upload the PDF to **Moodle**.

---

## Post-Implementation Documentation & Moodle Submission Workflow (Part 5)

Once all four station officers have completed and tested Parts 1–4:

### Step 1: Complete `mission-10-doc.md`
Open [`mission-10-doc.md`](./mission-10-doc.md) in this directory and fill in all sections together as a squadron:
1. **Squadron Roster & Station Sign-Off Table** (Station Roles, Cadet Full Names, Deliverable Files, and Verification Status).
2. **Integrated Architecture Diagram** (ASCII and Mermaid diagrams depicting how OpenTelemetry spans, CaMeL Dual-LLM capability gates, Semantic Entropy clustering, and the Red-Team RBAC Sentinel interact).
3. **Station-by-Station Experimental Reports (Sections A–D)** (OpenTelemetry span tree & circuit breaker logs, CaMeL taint verification traces, Semantic Entropy vs. Naive Lexical Entropy comparison tables, and OWASP LLM01:2025 Attack Success Rate & RBAC audit logs).
4. **End-to-End Integration Verification Output** (Paste terminal output from running all 4 modules).
5. **Squadron Reflection on Semantic Entropy, CaMeL & Agent Observability** (1–2 paragraph synthesis answering the Week 10 central question).

### Step 2: Convert `mission-10-doc.md` to `mission-10-doc.pdf`
Choose any of the following methods to generate `mission-10-doc.pdf`:
- **Option A — VS Code (Recommended):** Install the *Markdown PDF* extension (`yzane.markdown-pdf`), open `mission-10-doc.md`, right-click inside the editor, and choose **Markdown PDF: Export (pdf)**.
- **Option B — Pandoc CLI:**
  ```bash
  pandoc mission-10-doc.md -o mission-10-doc.pdf --pdf-engine=typst
  ```
- **Option C — Browser / GitHub Preview Print-to-PDF:** Open the rendered Markdown preview in VS Code (`Ctrl+Shift+V`) or on GitHub and select **Print $\rightarrow$ Save as PDF**.

### Step 3: Submit in Moodle & USSE Noosphere Portal
1. **Moodle Submission (Required):** Upload **`mission-10-doc.pdf`** (which contains your active GitHub repository URL and all 4 role reports) to the **Week 10 Assignment Drop-box in Moodle** before the deadline.
2. **USSE Noosphere Portal:** Push your completed code and `mission-10-doc.md` / `mission-10-doc.pdf` to your public squad GitHub repository, then paste the repository URL under **Week 10 (Mission 10) $\rightarrow$ 4. Transmit Solution**.
