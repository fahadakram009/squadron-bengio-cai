# Mission 05: Dual-Path GraphRAG, Hybrid Search & The Google NotebookLM Grounding Lab

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 5 — Grounding AI: From Vector Indexing to Google NotebookLM & GraphRAG  
**Waypoint:** Crab Pulsar Remnant / Messier 1 (`SEC-05-M01`, 6,500 ly from Sol)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+ / Python 3.11+ / Google NotebookLM  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 5**.

```
mission-05/
├── README.md                                          <- Mission overview & squad sign-off status
├── ps05_briefing.md                                    <- Full mission specification & problem statement
├── 01_HNSW_Vector_Index_starter.py                    <- Starter script (Systems Architect)
├── 02_BM25_and_RRF_Combiner_starter.py               <- Starter script (Protocol Officer)
├── 03_NotebookLM_Grounding_Lab_template.md            <- Laboratory template (Data Specialist)
└── 04_Prompt_Injection_and_RAG_Audit_starter.py      <- Defense & RAG triad audit (Security Sentinel)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [ ] **Systems Architect (`01_HNSW_Vector_Index_starter.py`):** Pure Python/NumPy HNSW graph indexing, logarithmic skip-list traversal, and benchmark against brute-force flat search.
- [x] **Communications & Protocol Officer (`02_BM25_and_RRF_Combiner.py`)** - Madhu **:** Okapi BM25 lexical scorer and Reciprocal Rank Fusion ($k=60$) combiner merging dense & sparse candidate streams.
- [ ] **Data & Payload Specialist (`03_NotebookLM_Grounding_Lab_template.md`):** Google NotebookLM empirical laboratory: source-grounded citation extraction, negative constraint audit, and Audio Overview evaluation.
- [x] **Security & Ethics Sentinel (`04_Prompt_Injection_and_RAG_Audit_starter.py` / `04_RAG_Provenance_and_Safety.py`)** - Janusz **:** Indirect prompt injection sanitizer and automated RAG Triad audit (Faithfulness, Answer Relevance, Context Recall).

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 5 (Mission 5) -> Problem Set ps05 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
