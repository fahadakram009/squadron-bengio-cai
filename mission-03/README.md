# Mission 03: Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Module:** Week 3 — Modern Transformer Architectures and Architectural Divergence  
**Waypoint:** Pleiades Open Cluster (`SEC-03-M45`, 444 ly from Sol)  
**IDE & Tools Standard:** Visual Studio Code + Antigravity (`agy`) | PyTorch 2.4+ / Colab / Modern LLMs  
**Instructor Reference Repository:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

## Mission Folder Structure & Role Deliverables

This repository folder contains the starter assets and templates for **Mission 3**.

```
mission-03/
├── README.md                                          <- Mission overview & squad sign-off status
├── ps03_briefing.md                                    <- Full mission specification & problem statement
├── 01_MultiHead_Attention_Core_starter.py             <- Starter script (Systems Architect)
├── 02_RoPE_Positional_Engine_starter.py               <- Starter script (Protocol Officer)
├── 03_GQA_and_KVCache_Benchmark_starter.py           <- Starter script (Data & Payload Specialist)
└── 04_Memory_Hierarchy_and_Safety_Audit_template.md   <- Deliverable template (Security Sentinel)
```

---

## Station Role Deliverables Checklist

Before transmitting your team's repository link on the course website submission portal, complete all deliverables and record digital sign-offs from all four station officers:

- [x] **Systems Architect (`01_MultiHead_Attention_Core_starter.py`)** - Fahad **:** Pure PyTorch `MultiHeadSelfAttention` with causal masking, head splitting, and empirical variance preservation unit test ($\frac{1}{\sqrt{d_k}}$).
- [x] **Communications & Protocol Officer (`02_RoPE_Positional_Engine_starter.py`)** - Madhu **:** Rotary Positional Embeddings (`apply_rope`), complex 2D planar rotation, and relative position invariance verification.
- [x] **Data & Payload Specialist (`03_GQA_and_KVCache_Benchmark_starter.py`)** - Osaf Ahmad **:** Grouped-Query Attention layer (`GQAAttention`), stateful autoregressive `KVCache`, and latency/memory benchmark table.
- [x] **Security & Ethics Sentinel (`04_Memory_Hierarchy_and_Safety_Audit_template.md`)** - Janusz **:** FlashAttention IO complexity derivation ($O(N)$ SRAM tiling), KV-Cache DoS / OOM analysis, and mitigation evaluation (PagedAttention, MLA).

---

## Running the attention core (Systems Architect)

```bash
# from the repo root (same .venv as Missions 1-2; CPU-only PyTorch is enough, no API key needed)
source .venv/bin/activate && pip install -r mission-03/requirements.txt
cd mission-03
python 01_MultiHead_Attention_Core_starter.py                 # variance-preservation test + demo causal forward pass
python 01_MultiHead_Attention_Core_starter.py --selftest      # 27 checks: masks, head split, oracles, gradients, variance, integration
python 01_MultiHead_Attention_Core_starter.py --bench         # O(N^2) time / score-matrix memory for N = 128 ... 2048
python 01_MultiHead_Attention_Core_starter.py --check-signoffs  # gate: all four boxes above must be ticked
```

Each run writes `logs/<timestamp>-ps03-<mode>.json` (check list, variance table, benchmark rows).

**What `MultiHeadSelfAttention` implements** (pure tensor algebra, no `nn.MultiheadAttention` /
`F.scaled_dot_product_attention` in the forward path): bias-free `W_Q, W_K, W_V, W_O` in R^{d_model x d_model};
head split `(b, n, d_model) -> (b, h, n, d_head)` and its inverse; a cached strictly-lower-triangular causal mask
(`-inf` above the diagonal before softmax) that combines with a `(batch, seq_len)` key-padding mask without producing
NaN rows; `softmax(QK^T / sqrt(d_k)) V` written out in `scaled_dot_product_attention`. The self-test proves the
vectorised module equals an explicit per-head loop, equals PyTorch's two reference implementations with copied
weights (max difference 0 and 1.5e-7), blocks all information flow from future positions, and back-propagates
finite gradients to all four projections.

**Variance preservation** (`verify_variance_preservation`, d_k in {16, 64, 256, 1024}, unit-variance inputs):
`Var(q.k)` measures 16.3 / 64.0 / 255.4 / 1024.0 and `Var(q.k / sqrt(d_k))` 1.019 / 1.000 / 0.998 / 1.000.
The consequence is shown alongside: at d_k = 1024 the unscaled softmax has 0.21 bits of entropy (one-hot
attention, gradient p(1-p) ~ 0) while the scaled softmax keeps 6.3 of the possible 7 bits.

**Integration of the four stations.** The module exposes a `rope=` hook on `forward` (applied to Q and K after the
head split) and the self-test imports the other scripts at run time:

| Station file | What the attention core does with it | If absent |
|---|---|---|
| `02_RoPE_Positional_Engine_starter.py` | imports `precompute_freqs_cis` / `apply_rotary_emb`, runs attention with the hook and checks the attention map is invariant to a global shift of 37 positions (relative-position property at the attention level) | check skipped |
| `03_GQA_and_KVCache_Benchmark_starter.py` | instantiates `GQAAttention` with H_KV = H_Q, copies the four weight matrices and checks it equals `MultiHeadSelfAttention` (GQA must reduce to MHA) | check skipped |
| `04_Memory_Hierarchy_and_Safety_Audit_template.md` | `--bench` supplies the measured O(N^2) score-matrix growth the IO-complexity derivation starts from | - |

---

## How to Submit
1. Push your squad's completed solution repository to GitHub.
2. Go to the course website portal under **Week 3 (Mission 3) -> Problem Set ps03 & Submission**.
3. Paste your public squad GitHub repository URL and click **Transmit Solution Link to Database**.
