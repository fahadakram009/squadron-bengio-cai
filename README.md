# Conversational AI (WS 2026/27) — Course Exercises & Missions

Welcome to the official student exercises and starter templates repository for **Conversational AI: History, Applications, and Future** (Winter Semester 2026/27) aboard the starship *USSE Noosphere*.

---

## Course Missions & Problem Sets

This repository contains the starter code, briefing specifications, role deliverable templates, and notebooks for each weekly mission:

| Mission | Sector Waypoint | Focus Topics | Folder |
| :--- | :--- | :--- | :--- |
| **Mission 01** | *Alpha Centauri* | The Architecture of Argument: Socratic Elenchus & Galilean 3-Persona Debate | [`mission-01/`](./mission-01/) |
| **Mission 02** | *Sirius Binary System* | FST Dialogue Systems, Lovelace's Objection & Shannon Noisy Channel | [`mission-02/`](./mission-02/) |
| **Mission 03** | *Pleiades Open Cluster* | Multi-Head Attention, RoPE Rotary Embeddings & Grouped-Query KV-Cache | [`mission-03/`](./mission-03/) |
| **Mission 04** | *Betelgeuse / Orion* | Direct Preference Optimization (DPO), SimPO & Activation Steering | [`mission-04/`](./mission-04/) |
| **Mission 05** | *Crab Pulsar Remnant* | Dual-Path GraphRAG, BM25 + HNSW Hybrid Search & Google NotebookLM | [`mission-05/`](./mission-05/) |
| **Mission 06** | *Eta Carinae Nebula* | Autonomous Warp State Machine, Tool Calling Contracts & Deliberation Loop | [`mission-06/`](./mission-06/) |
| **Mission 07** | *Sagittarius A* (`SEC-07-SGR`)* | Test-Time Compute Scaling, Tree-of-Thoughts, MCTS (PUCT) & Process Reward Models (PRM) | [`mission-07/`](./mission-07/) |
| **Mission 08** | *Cygnus X-1 & Veil Nebula (`SEC-08-CYG`)* | ReWOO DAG Planning, Reflexion Traceback Surgery, MemGPT Virtual Context Paging & MemoryGraft Defense | [`mission-08/`](./mission-08/) |
| **Mission 09** | *Large Magellanic Cloud / Tarantula Nebula (`SEC-09-LMC`)* | Agentic RAG, Corrective RAG (CRAG), Self-RAG Reflection & Hardware-Accelerated Retrieval on Free Google TPUs (`torch_tpu`) | [`mission-09/`](./mission-09/) |
| **Mission 10** | *Boötes Deep Void (`SEC-10-VOID`)* | AI Safety, Semantic Entropy, CaMeL Dual-LLM Injection Defense & OpenTelemetry (`gen_ai.*`) Agent Observability | [`mission-10/`](./mission-10/) |
| **Mission 11** | *Andromeda Galaxy Core (`SEC-11-M31`)* | Multi-Agent Systems (MAS), State-Graph Reducers, Least-Context Handoffs, Condorcet Debate Quorums, 3-Color DFS Deadlock Defense & Google ADK 2.0 / Antigravity Studio | [`mission-11/`](./mission-11/) |

---

## Squad Station Roles

Each squad divides responsibilities across the four operational bridge roles:

1. **Systems Architect (Lead Engineer):** Mathematical formulation, core PyTorch layers, state machines, and algorithmic complexity proofs.
2. **Communications & Protocol Officer:** Protocol schemas, JSON validators, prompt engineering, dialogue reflection, and rank combiners.
3. **Data & Payload Specialist:** Benchmarking, memory profiles, length bias diagnostics, Grounding Labs, and loop sentinel hashers.
4. **Security & Ethics Sentinel:** Hardware hierarchy audits, DoS stress testing, indirect prompt injection defense, and capability retention audits.

---

## Instructions for Cadets

1. **Clone or Fork:** Clone this repository to your local squad development environment:
   ```bash
   git clone https://github.com/Unisvet/couse-c-ai.git
   ```
2. **Virtual Environment & Dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install torch numpy
   ```
3. **IDE Environment:** Work in **Visual Studio Code** using **Google Antigravity (`agy`)** and modern LLM APIs (Gemini, ChatGPT, Claude).
4. **Solve & Verify:** Complete all role deliverables and verify tests using the verification functions embedded in each starter script.
5. **Submit Solution:** Push your squad's solution repository to GitHub, copy your public repository URL, and transmit the link through the course web platform under the corresponding week's problem set tab.
