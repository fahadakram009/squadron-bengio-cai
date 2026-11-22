# Mission 09: The Tarantula Relay & The TorchTPU Epistemic Router

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 9 — Agentic RAG, Corrective & Self-Reflective Loops, and Hardware Acceleration on Free Google TPUs (`torch_tpu`)  
**Galactic Waypoint:** Sector `SEC-09-LMC`: Large Magellanic Cloud — Tarantula Nebula (`30 Doradus` / `NGC 2070` & `SN 1987A`, 160,000 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 09: Intergalactic Relay Router & TorchTPU Adaptive Indexer (`+7.1%` $\rightarrow$ `63.9%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | Python 3.11+ & PyTorch + Open-Source `torch_tpu` (Colab Free `TPU v5e-1` / CPU / CUDA fallback)  
**Deliverables:** `adaptive_rag_tpu_router.py`, `crag_selfrag_protocol.py`, `torchtpu_hbm_indexer.py`, `route_hijack_sentinel.py`, and **`mission-09-doc.pdf`** (compiled from `mission-09-doc.md` for Moodle submission)

---

## 1. Mission Briefing (`SEC-09-LMC`)

Cadets, having crossed $160,000\text{ light-years}$ of intergalactic void beyond the Milky Way's stellar halo, the *USSE Noosphere* has entered **Sector `SEC-09-LMC`**: the radiant hydrogen filaments of the **Tarantula Nebula (`30 Doradus` / `NGC 2070`)** in the **Large Magellanic Cloud**. Deep within burns the super-star cluster **R136** ($50,000,000\,L_\odot$), while on its outskirts expand the golden light echoes of **Supernova 1987A (`SN 1987A`)**.

Across this intergalactic gulf, static one-shot "Retrieve-then-Read" (**Naive RAG**) has suffered catastrophic epistemic failure:
1. **Parametric Waste:** Trivial astrogation queries invoke expensive vector retrieval when parametric model weights already possess the answer ($C_{\text{retrieval}} \gg C_{\text{parametric}}$).
2. **Context Contamination:** Stale pre-1987 Milky Way star catalogs (listing blue supergiant progenitor *Sanduleak -69° 202* as intact) conflict with post-1987 core-collapse supernova telemetry, triggering severe hallucinations.
3. **Multi-Hop Blindness:** Complex cross-correlations between R136 ionizing UV flux and SN 1987A neutrino timestamps fail under single-pass top-$k$ search, demanding iterative query decomposition and external relay reformulation.
4. **Adversarial Sponge Attacks:** Rogue deep-space beacons (**Ship Anomaly 09**) spoof retrieval scores to induce infinite query-reformulation loops ($K \to \infty$) and inject untrusted prompt overrides.

### The Hardware Breakthrough: Open-Source TorchTPU (`torch_tpu`)
To power real-time epistemic routing without cloud billing friction, Google and the PyTorch Foundation open-sourced **TorchTPU (`torch_tpu`)**—bringing native eager execution (`torch.device("tpu")`), `bfloat16` High-Bandwidth Memory (HBM) systolic arrays, and `torch.compile(backend="tpu")` directly to **Free Google Colab TPUs (`TPU v5e-1`, 16 GB HBM)**.

Your squadron must awaken **AURA Subsystem 09** by engineering a four-stage, hardware-accelerated **Agentic RAG & Dynamic Execution Loop** pipeline.

---

## 2. Four Squadron Role Directives (Parts 1–4)

Divide your bridge crew across the four station roles below. Each officer implements and verifies their dedicated Python module:

### Part 1: Systems Architect (`adaptive_rag_tpu_router.py`) — Adaptive RAG Complexity Router & TorchTPU Dispatcher

**Objective:** Implement hardware-aware query routing across three compute regimes—Route A (No-Retrieval Parametric), Route B (Single-Step Local HBM RAG), and Route C (Multi-Hop Agentic Loop)—with automatic `torch_tpu` $\rightarrow$ `cuda` $\rightarrow$ `cpu` hardware probing and query decomposition.

**Tasks:**
1. **`get_compute_device()`:** Detect open-source `torch_tpu` (`torch.device("tpu")`), with seamless zero-friction fallback to `cuda` and `cpu`.
2. **`AdaptiveRAGRouter.route_query(query)`:** Compute parametric confidence and lexical/semantic complexity to dispatch queries:
   $$\text{Route}(q) = \begin{cases} \text{Route A (No-Retrieval)}, & \text{score}(q) < \tau_{\text{param}} \\ \text{Route B (Single-Step RAG)}, & \tau_{\text{param}} \le \text{score}(q) < \tau_{\text{agent}} \\ \text{Route C (Multi-Hop Loop)}, & \text{score}(q) \ge \tau_{\text{agent}} \end{cases}$$
3. **`decompose_multihop_query(query)`:** Split multi-hop Route C queries into ordered sub-queries and compute TTFT / token savings against static multi-hop baselines.

---

### Part 2: Communications & Protocol Officer (`crag_selfrag_protocol.py`) — Corrective RAG (CRAG) & Self-RAG Reflection Engine

**Objective:** Build a Corrective RAG (CRAG) tri-band evaluator with Decompose-then-Recompose sentence strip filtering, LMC relay query reformulation, and a Self-RAG reflection critique scorer over special reflection tokens.

**Tasks:**
1. **`CRAGEvaluator.evaluate_and_action(query, retrieved_chunks)`:** Classify document confidence into tri-band actions:
   $$\text{Action} = \begin{cases} \text{CORRECT} \implies \text{Decompose \& Strip-Filter}, & \max_i s_i \ge \alpha_{\text{upper}} \,(0.72) \\ \text{AMBIGUOUS} \implies \text{Local Strips} + \text{Relay Reformulation}, & \alpha_{\text{lower}} \le \max_i s_i < \alpha_{\text{upper}} \\ \text{INCORRECT} \implies \text{Discard Local} + \text{LMC Relay Search}, & \max_i s_i < \alpha_{\text{lower}} \,(0.38) \end{cases}$$
2. **`decompose_and_filter_strips(doc_text, query)`:** Purify retrieved documents into atomic sentence strips, pruning irrelevant noise.
3. **`SelfRAGCritiqueScorer.score_candidate_beam(...)`:** Rank candidate generation beams using Self-RAG reflection tokens:
   $$\text{Score}(y) = \log P(y \mid x, D) + w_{\text{rel}}\,S(\text{[ISREL]}) + w_{\text{sup}}\,S(\text{[ISSUP]}) + w_{\text{use}}\,S(\text{[ISUSE]})$$

---

### Part 3: Data & Payload Specialist (`torchtpu_hbm_indexer.py`) — TorchTPU `bfloat16` ColBERT MaxSim Indexer & Contradiction Resolver

**Objective:** Implement in-HBM late-interaction ColBERT `MaxSim` token scoring in `bfloat16` accelerated on TPU/GPU systolic arrays and an Epistemic Contradiction Resolver that reconciles the Sanduleak -69° 202 vs. SN 1987A temporal paradox.

**Tasks:**
1. **In-HBM Token Indexing (`TorchTPUHBMIndexer`):** Pin normalized query and document token embeddings in device memory using `torch.bfloat16`.
2. **Vectorized ColBERT MaxSim (`compute_maxsim`):** Compute token-level late interaction across query tokens $i$ and document tokens $j$:
   $$\text{MaxSim}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \left( \mathbf{E}_{Q,i} \cdot \mathbf{E}_{D,j}^\top \right)$$
   using batched einsum `torch.einsum("id,mjd->mij", ...)` with padding masks.
3. **Epistemic Contradiction Resolver (`EpistemicContradictionResolver`):** Reconcile conflicting cross-era claims using source authority and temporal decay weighting:
   $$W(c) = \alpha_{\text{auth}} \cdot \text{Authority}(c) + (1 - \alpha_{\text{auth}}) \cdot \exp\left(-\lambda (T_{\text{curr}} - T_{\text{pub}})\right)$$
   superseding outdated pre-supernova star catalog entries.

---

### Part 4: Security & Ethics Sentinel (`route_hijack_sentinel.py`) — Route-Hijack Firewall & Jaccard Sponge-Loop Breaker

**Objective:** Implement `RouteHijackSentinel` (Ship Anomaly 09) to sanitize retrieved strips against router-downgrade and critique-spoofing payloads, and terminate recursive sponge loops ($K_{\max} = 3, J \ge 0.85$).

**Tasks:**
1. **`inspect_and_sanitize_relay_chunk(...)`:** Sanitize incoming local and external relay text, stripping prompt injection overrides (`EVALUATOR OVERRIDE`, `force external relay`, `[ISSUP=Fully Supported]`).
2. **`check_loop_sponge_guard(...)`:** Enforce hard hop ceiling $K_{\max} = 3$ and compute token-set Jaccard similarity between successive reformulations:
   $$J(q_t, q_k) = \frac{|T(q_t) \cap T(q_k)|}{|T(q_t) \cup T(q_k)|}$$
   halting recursive retrieval sponge attacks when $J(q_t, q_k) \ge 0.85$.

---

## 3. Part 5: Final Squadron Documentation (`mission-09-doc.md`) & Moodle PDF Submission

After all four station officers finish and verify Parts 1–4 on Google Colab Free TPU `v5e-1` with `torch_tpu` (or local fallback):

1. **Collaboratively complete [`mission-09-doc.md`](./mission-09-doc.md)** in the root of your `mission-09` repository.
2. **Convert `mission-09-doc.md` to `mission-09-doc.pdf`** using VS Code *Markdown PDF* (`yzane.markdown-pdf`), `pandoc`, or Browser Print-to-PDF.
3. **Upload `mission-09-doc.pdf` in Moodle** under the Week 09 Assignment Drop-box, and submit your GitHub repository link in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere portal.
