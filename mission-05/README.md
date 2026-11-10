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

- [x] **Systems Architect (`01_HNSW_Vector_Index_starter.py`)** - Fahad **:** Pure Python/NumPy HNSW graph indexing, logarithmic skip-list traversal, and benchmark against brute-force flat search.
- [x] **Communications & Protocol Officer (`02_BM25_and_RRF_Combiner.py`)** - Madhu **:** Okapi BM25 lexical scorer and Reciprocal Rank Fusion ($k=60$) combiner merging dense & sparse candidate streams.
- [x] **Data & Payload Specialist (`03_NotebookLM_Grounding_Lab_template.md` / `03_NotebookLM_RAG_Audit.py`)** - Osaf **:** Google NotebookLM empirical laboratory (protocol and report generator uploaded 2026-11-08; probe observations pending): source-grounded citation extraction, negative constraint audit, and Audio Overview evaluation.
- [x] **Security & Ethics Sentinel (`04_Prompt_Injection_and_RAG_Audit_starter.py` / `04_RAG_Provenance_and_Safety.py`)** - Janusz **:** Indirect prompt injection sanitizer and automated RAG Triad audit (Faithfulness, Answer Relevance, Context Recall).

---

## Running the HNSW index (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-4; NumPy only, no API key needed)
source .venv/bin/activate && pip install -r mission-05/requirements.txt
cd mission-05
python 01_HNSW_Vector_Index_starter.py                   # demo: 2,000 x 64 build, one query HNSW vs flat, 8,000 x 128 scaling table
python 01_HNSW_Vector_Index_starter.py --selftest        # 35 checks: metric, layer law, graph invariants, exact search, recall, scaling, integration
python 01_HNSW_Vector_Index_starter.py --bench           # N = 25,000 x 768: build, latency, recall, comparisons per query (about 1 min)
python 01_HNSW_Vector_Index_starter.py --bench --data random   # the hard case: isotropic 768-d Gaussians
python 01_HNSW_Vector_Index_starter.py --check-signoffs  # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps05-<mode>.json`.

**What `01_HNSW_Vector_Index_starter.py` implements** (pure Python + NumPy, no faiss / hnswlib):

1. `cosine_similarity(u, v)` and the vectorised `cosine_similarity_matrix(Q, X)`; stored vectors are unit
   length, so every comparison is one dot product and the flat baseline `brute_force_search` is one
   matrix-vector product over all N rows.
2. `_random_level()`: `l = floor(-ln U * m_L)` with `m_L = 1 / ln M`. The self-test draws 40,000 levels and
   checks P(l >= 1) = 1/M and P(l >= 2) = 1/M^2; the 25,000-node index has layers [25000, 1627, 92, 5, 1]
   (measured top-layer fraction 0.0651 against 1/16 = 0.0625).
3. `search_layer(entry_point, query, ef, layer)` (Algorithm 2): heap-based best-first search within one
   layer that keeps the ef best nodes and stops when the nearest unexpanded candidate is worse than the worst
   kept. `query` hops greedily along the long-range edges of the upper layers (`search_layer` with ef = 1)
   and runs the ef sweep only on layer 0. Neighbour sets are chosen with the paper's heuristic (Algorithm 4,
   diverse directions, pruned connections kept) and capped at M per upper layer and 2M on layer 0.
4. Benchmark: one index built to 25,000 x 768 vectors, the same 100 queries run at 1k / 2.5k / 5k / 10k / 25k
   nodes against HNSW and flat search over the same rows.

| N | HNSW median | flat median | speed-up | recall@10 | comparisons per query HNSW | flat |
|---|---|---|---|---|---|---|
| 1,000 | 0.60 ms | 0.04 ms | 0.1x | 1.000 | 592 | 1,000 |
| 5,000 | 0.66 ms | 0.25 ms | 0.4x | 1.000 | 523 | 5,000 |
| 10,000 | 0.60 ms | 0.54 ms | 0.9x | 1.000 | 402 | 10,000 |
| 25,000 | 0.51 ms | 1.61 ms | 3.2x | 1.000 | 304 | 25,000 |

Flat search is linear in N (0.04 ms to 1.61 ms); the HNSW query touches a few hundred vectors at every size,
so its latency stays flat at about 0.5 ms and the crossover against NumPy's BLAS matrix product is near
N = 10,000 on this laptop. The `ef_search` knob trades recall for latency at query time: at N = 25,000,
ef = 16 gives recall 0.944 in 0.32 ms, ef = 64 gives 1.000 in 0.52 ms. Build time is 51 s for 25,000 vectors
(pure Python insert loop); memory is 73 MB of vectors plus 5 MB of graph.

**Why the data generator matters.** The default corpus is a 200-topic Gaussian mixture with intrinsic
dimension 32 projected into 768 dimensions, which is how real text embeddings behave. At N = 10,000 and
ef = 64 recall is 1.000 there, 0.75 for full-dimensional clusters and 0.34 for isotropic 768-d noise
(`--data random`), where all distances concentrate and no graph index navigates well; the self-test also
shows that the heuristic neighbour selection beats plain top-M selection (0.97 vs 0.83 recall) on
full-dimensional clusters.

**Integration of the four stations.** The Protocol Officer's `DenseRetriever` loads `SimpleHNSW` from this
file at run time (`insert(id, vector)`, `query(vector, top_k)`), builds the dense side of the hybrid
pipeline on it and fuses its ranking with BM25 through RRF; the self-test here calls that retriever and
checks recall@10 = 1.00 against brute force, checks `rrf_fuse` on a hand example, and verifies the
Sentinel's `PromptInjectionSanitizer` / `RAGTriadEvaluator` are present for the downstream audit.

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 5 (Mission 5) -> Problem Set ps05 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
