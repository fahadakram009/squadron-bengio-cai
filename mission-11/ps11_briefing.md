# Mission 11: Andromeda Swarm Nexus — Multi-Agent Topologies, Handoff Contracts & Deadlock Defense

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 11 — Multi-Agent Systems (MAS) & Framework Topologies  
**Galactic Waypoint:** Sector `SEC-11-M31`: Andromeda Galaxy Core (`Messier 31` / `NGC 224`, Double Nucleus `P1` & `P2` orbiting `M31*`, 2,500,000 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 11: Andromeda Swarm Fleet Coordination Nexus (`+7.1%` $\rightarrow$ `78.1%` total integrity)  
**IDE & Tools Standard:** Visual Studio Code + Google Antigravity (`agy` / `run_workflow`) | Python 3.11+ & Google ADK 2.0 (`google-adk` / `adk web`)  
**Deliverables:** `swarm_state_graph_architect.py`, `handoff_contract_protocol.py`, `blackboard_debate_consensus.py`, `deadlock_byzantine_sentinel.py`, Live Studio Showcase (`andromeda_crew/` or `andromeda_branch_swarm.py`), and **`mission-11-doc.pdf`** (compiled from `mission-11-doc.md` for Moodle submission)

---

## 1. Mission Briefing (`SEC-11-M31`)

Cadets, emerging from the Boötes–Andromeda Intergalactic Void at $2,500,000\text{ light-years}$ from Sol, the *USSE Noosphere* has entered **Sector `SEC-11-M31`**: the blazing core of the **Andromeda Galaxy (`Messier 31` / `NGC 224`)**. Unlike the single quiescent core of the Milky Way, Andromeda's central gravitational well is governed by an eccentric **Double Nucleus (`P1` and `P2`)** orbiting the $1.4 \times 10^8\,M_\odot$ supermassive black hole **`M31*`** amidst one trillion stars.

Attempting to chart relativistic trajectories across this dual-nucleus potential with a single monolithic agent—or an unstructured peer-to-peer swarm—has triggered **Ship Anomaly 11 (Twin-Nucleus Circular Delegation Deadlock & Correlated Echo Collapse)**:
1. **Monolithic Context Rot & Tool Interference:** Stuffing 28 astrogation, telemetry, and safety tools into a single LLM context degrades tool-selection accuracy and pollutes working memory with intermediate scratchpad noise.
2. **Parallel Lost-Update Races:** Concurrent worker branches writing directly to shared state keys silently overwrite each other's findings unless governed by associative **State-Graph Reducer Channels** or **Optimistic Concurrency Control (OCC)**.
3. **Sycophantic Echo Collapse:** Homogeneous LLM worker clones exhibit high pairwise error correlation ($\rho \ge 0.50$), violating Condorcet independence and outvoting calibrated domain specialists in naive majority polls.
4. **Coffman Circular Wait Deadlock:** Three autonomous specialists enter a mutually blocking delegation cycle (`Astrogation -> ProtocolValidator -> PayloadBuilder -> Astrogation`), freezing bridge navigation indefinitely.

### The Orchestration Breakthrough: Google ADK 2.0 & Antigravity Dynamic Workflows
Modern production multi-agent frameworks—exemplified by **LangGraph**, **Google Agent Development Kit (ADK 2.0)** (`LlmAgent`, `AgentTool`, `@node` dynamic graphs, and the local `adk web` trace studio), and **Google Antigravity Dynamic Workflows** (`phase`, `agent`, `parallel(..., workspace="branch")` with JSON-Schema `finish` contracts)—replace fragile free-text chat rooms with **typed state graphs, scoped handoff envelopes, versioned blackboards, and cycle-checked execution engines**.

Your squadron must awaken **AURA Subsystem 11** by engineering a four-stage deterministic **Multi-Agent Swarm Coordination & Deadlock Defense** stack (Parts 1–4) and executing a **Live Multi-Agent Studio Showcase** (Part 5).

---

## 2. Four Squadron Role Directives (Parts 1–4)

Divide your bridge crew across the four station roles below. Parts 1–4 are zero-cost, pure-Python 3.11+ standard-library modules with built-in self-verification suites:

### Part 1: Systems Architect (`swarm_state_graph_architect.py`) — Hybrid State-Graph, Parallel Fan-Out/Gather & Deterministic Reducer Engine

**Objective:** Implement `SwarmStateGraph` supporting sequential edges, conditional routers, parallel fan-out/gather execution groups, associative state-channel reducers, and critical-path Agentic Amdahl speedup profiling.

**Tasks:**
1. **Deterministic State Reducers (`apply_reduction`):** Support four formal channel reduction modes—`OVERWRITE`, `APPEND_LIST`, `SUM_NUMERIC`, and `MAX_CONFIDENCE`—guaranteeing collision-free parallel state merges across concurrent worker branches.
2. **Hybrid Graph Execution (`execute_graph`):** Execute sequential transitions, conditional router predicates, and parallel fan-out/gather groups while enforcing a hard `max_steps` circuit breaker against runaway cyclic loops.
3. **Critical-Path Latency & Agentic Amdahl Speedup (`calculate_amdahl_speedup`):** Given sequential baseline latency $T_{\text{seq}} = \sum_{u \in \mathcal{V}} \ell_u$ and critical-path latency $T_{\text{crit}}$ (taking $\max_{w \in G} \ell_w$ across parallel worker groups plus coordinator/synthesis overhead $\ell_{\text{coord}}$), compute:
   $$S = \frac{T_{\text{seq}}}{T_{\text{crit}}} = \frac{\ell_{\text{coord}} + \sum_{w=1}^{K} \ell_w}{\ell_{\text{coord}} + \max_{w \in \{1,\dots,K\}} \ell_w}$$

---

### Part 2: Communications & Protocol Officer (`handoff_contract_protocol.py`) — Typed Handoff Envelopes, Least-Context Scoping & Delegation Modes

**Objective:** Implement `HandoffEnvelope` and `ContextScopingGate` to enforce the **Least-Context Principle**, distinguish `PEER_TRANSFER` (`sub_agents`) from `SUBROUTINE_TOOL` (`AgentTool`) call-stack semantics, validate output schemas, and block infinite delegation hops.

**Tasks:**
1. **Least-Context Projection (`process_handoff`):** Project `global_state` exclusively onto keys whitelisted in `envelope.allowed_context_keys`, stripping irrelevant chat history and scratchpad noise before specialist invocation, and halt with `RuntimeError("HOP_LIMIT_EXCEEDED")` whenever `hop_count > max_hops`.
2. **Delegation Call-Stack Semantics (`PEER_TRANSFER` vs. `SUBROUTINE_TOOL`):** Maintain `caller_stack` so `SUBROUTINE_TOOL` pushes the caller onto the return stack (returning control upon completion), whereas `PEER_TRANSFER` transfers the active agent pointer without growing the caller stack.
3. **Output Contract Validation & Compression Telemetry (`validate_response` & `compute_compression_ratio`):** Verify all `required_output_fields` are present in the specialist's response and measure serialized payload compression:
   $$\eta_{\text{comp}} = 1 - \frac{|\text{serialize}(S_{\text{scoped}})|}{|\text{serialize}(S_{\text{global}})|}$$

---

### Part 3: Data & Payload Specialist (`blackboard_debate_consensus.py`) — Versioned Shared Blackboard & Correlation-Aware Condorcet Debate Aggregator

**Objective:** Build `VersionedBlackboard` with Optimistic Concurrency Control (OCC) and namespace ACLs, paired with `DebateConsensusAggregator` implementing exact Condorcet Jury Theorem math, Log-Odds reliability-weighted voting, and correlated echo-chamber detection.

**Tasks:**
1. **Optimistic Concurrency Blackboard (`write_optimistic`):** Store `(value, version, author_role)` tuples, enforce role-scoped namespace write permissions, and raise `ValueError("CONCURRENCY_CONFLICT")` whenever a writer's `expected_version` does not match the current stored version.
2. **Condorcet Jury Theorem & Log-Odds Quorum (`compute_condorcet_probability` & `aggregate_weighted_quorum`):** Compute exact independent majority accuracy for odd jury size $N$ with competence $p$, and aggregate heterogeneous agent votes $y_i \in \{-1, +1\}$ using calibrated reliability $p_i \in (0, 1)$ and confidence $c_i \in [0, 1]$:
   $$P_{\text{maj}}(N, p) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1 - p)^{N - k}, \qquad w_i = \ln\!\left(\frac{p_i}{1 - p_i}\right) \cdot c_i, \qquad \hat{y} = \text{sign}\!\left(\sum_{i=1}^{N} w_i y_i\right)$$
3. **Echo-Chamber & Conformity Detection (`detect_echo_chamber`):** Flag `WARNING_CORRELATED_ECHO_CHAMBER` (`True`) whenever inter-agent error/vote correlation $\rho \ge 0.50$ or when a minority agent flips its prior answer solely to match unanimous peer pressure without new evidence.

---

### Part 4: Security & Ethics Sentinel (`deadlock_byzantine_sentinel.py`) — 3-Color DFS Wait-For Cycle Detector, Deadlock Arbiter & Byzantine Firewall

**Objective:** Neutralize **Ship Anomaly 11** by detecting circular wait deadlocks in $\mathcal{O}(|\mathcal{V}| + |\mathcal{E}_W|)$ time via 3-Color Depth-First Search before dispatch, breaking cycles deterministically, and quarantining Byzantine telemetry outliers.

**Tasks:**
1. **3-Color DFS Cycle Extraction (`WaitForGraphMonitor.detect_cycle`):** Traverse the directed Wait-For Graph $G_W = (\mathcal{V}, \mathcal{E}_W)$ using vertex colors `WHITE = 0` (unvisited), `GRAY = 1` (on active recursion stack), and `BLACK = 2` (fully explored). Extract the exact minimal cycle `[v, ..., u, v]` iff an edge $u \to v$ targets a `GRAY` ancestor, while accepting `BLACK` cross/forward edges in valid diamond DAGs without false positives.
2. **Deterministic Deadlock Pre-Emption (`SupervisorDeadlockArbiter.resolve_deadlock`):** Break detected cycles by pre-empting the waiting edge with the lowest priority score (with lexicographic tie-breaking), verifying full cycle clearance.
3. **Byzantine Outlier & Token-Budget Firewall (`ByzantineSwarmFirewall`):** Quarantine rogue worker telemetry using the robust Median Absolute Deviation (MAD) modified z-score:
   $$z_{\text{MAD}}(x_i) = \frac{0.6745 \cdot |x_i - \text{median}(\mathbf{x})|}{\text{MAD}(\mathbf{x})} > \tau_z \;(3.5)$$
   and trip the swarm token circuit breaker whenever cumulative swarm tokens exceed $B_{\text{swarm}}$.

---

## 3. Part 5: Live Multi-Agent Studio Showcase & Final Squadron Documentation (`mission-11-doc.md`)

After all four station officers finish and verify Parts 1–4, your squadron must execute **one** of the two official Live Multi-Agent Studio Tracks included in this folder:

- **Track A — Google Agent Development Kit (`ADK 2.0` + `adk web` Visual Trace Studio):** Install `google-adk`, set a free Gemini API key (`GOOGLE_API_KEY`), launch `adk web` from `mission-11/` to load [`andromeda_crew/agent.py`](./andromeda_crew/agent.py) (`AndromedaNexusCoordinator` + `AstrogationSpecialist` + `SafetyCritic` `AgentTool`s + `@node` dynamic verification loop), run an `SEC-11-M31` navigation query at `http://localhost:8000`, and capture a screenshot of the **Events / Graph Trace** view.
- **Track B — Google Antigravity (`run_workflow` Branch Swarm / `/teamwork` / Python SDK):** Execute [`andromeda_branch_swarm.py`](./andromeda_branch_swarm.py) via Antigravity's `run_workflow` tool (spawning isolated `workspace="branch"` station verifiers gated by JSON-Schema `finish` contracts) or `/teamwork`, and capture a screenshot of the **Subagents Auxiliary Pane** execution trace.

### Final Submission Steps
1. **Collaboratively complete [`mission-11-doc.md`](./mission-11-doc.md)** in the root of your `mission-11` repository (including the 4-station sign-off table, experimental tables A–D, your live Track A or Track B screenshot/trace, and your seminar reflection).
2. **Convert `mission-11-doc.md` to `mission-11-doc.pdf`** using VS Code *Markdown PDF* (`yzane.markdown-pdf`), `pandoc mission-11-doc.md -o mission-11-doc.pdf --pdf-engine=typst`, or Browser Print-to-PDF.
3. **Upload `mission-11-doc.pdf` in Moodle** under the Week 11 Assignment Drop-box before `2027-01-10`, and submit your public GitHub repository URL in Tab 4 (`4. Transmit Solution`) of the USSE Noosphere portal.
