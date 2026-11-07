"""
Mission 05 - Part 2: Communications & Protocol Officer (Madhu)
Okapi BM25 Lexical Retrieval & Reciprocal Rank Fusion (RRF)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core

BM25(D, Q) = sum_i IDF(q_i) * f(q_i, D) * (k1 + 1) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl)),  k1 = 1.5, b = 0.75
RRF(d)     = sum over {dense, sparse} of 1 / (k + rank(d)),  k = 60

Changes compared to the starter:
- tokenizer keeps codes like CV-4471-K or ALPHA-9 as one token (the starter used .split(),
  so "ALPHA-9." at the end of a sentence did not match "ALPHA-9")
- BM25 uses an inverted index, rrf_fuse() works for any number of rankings
- benchmark on a small ship archive: conceptual queries (dense should win) and exact part codes (BM25 should win)

Dense side: the HNSW index from Part 1 (01_HNSW_Vector_Index.py or its starter). If it can't be loaded,
brute-force cosine search is used. The embeddings come from a small offline encoder (ToyConceptEncoder):
synonyms get similar vectors, unknown strings like part codes are built from character trigrams,
so near-identical codes end up with almost the same vector (like with a real embedding model).

Run:
    python 02_BM25_and_RRF_Combiner.py              # tests + benchmark + figure
    python 02_BM25_and_RRF_Combiner.py --selftest   # tests only
    python 02_BM25_and_RRF_Combiner.py --demo       # the starter's example
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FIGURE = HERE / "figures" / "hybrid_retrieval_benchmark.png"
HNSW_FILES = [HERE / "01_HNSW_Vector_Index.py", HERE / "01_HNSW_Vector_Index_starter.py"]

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its of on or our
the this to was we what when where which who why with you your me my please there any after before""".split())

# letters/digits, optionally joined by - _ . /  (CV-4471-K, ALPHA-9, v2.1 stay one token)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Okapi:
    """Okapi BM25. score() returns every document as (idx, score), best first, like the starter."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, tokenizer=tokenize):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [tokenizer(doc) for doc in corpus]
        self.doc_lens = [len(d) for d in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0
        self.tf = [Counter(d) for d in self.tokenized_corpus]
        self.df = Counter()
        self.postings = defaultdict(list)          # term -> docs containing it
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.df[term] += 1
                self.postings[term].append(i)
        n = self.corpus_size
        # same IDF as the starter, stays positive for common terms
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in self.df.items()}

    def term_score(self, term: str, doc_idx: int) -> float:
        f = self.tf[doc_idx].get(term, 0)
        if f == 0:
            return 0.0
        norm = 1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl) if self.avgdl > 0 else 1.0
        return self.idf[term] * f * (self.k1 + 1.0) / (f + self.k1 * norm)

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        for term in self.tokenizer(query):
            for doc_idx in self.postings.get(term, []):
                scores[doc_idx] += self.term_score(term, doc_idx)
        return scores

    def score(self, query: str) -> List[Tuple[int, float]]:
        s = self.get_scores(query)
        order = sorted(range(self.corpus_size), key=lambda i: (-s[i], i))
        return [(i, float(s[i])) for i in order]

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Only documents with score > 0, best first."""
        s = self.get_scores(query)
        hits = sorted(np.nonzero(s > 0)[0].tolist(), key=lambda i: (-s[i], i))
        return [(i, float(s[i])) for i in hits[:top_k]]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(rankings: Dict[str, List[int]], k: int = 60):
    """rankings = {method: [doc ids, best first]}. Ranks start at 1, only ranks are used (no scores).
    Returns [(doc, rrf score, {method: rank})], best first; ties -> best single rank, then doc id."""
    fused = defaultdict(float)
    ranks = defaultdict(dict)
    for method, ranking in rankings.items():
        seen = set()
        r = 0
        for doc in ranking:
            if doc in seen:
                continue
            seen.add(doc)
            r += 1
            fused[doc] += 1.0 / (k + r)
            ranks[doc][method] = r
    out = [(d, s, ranks[d]) for d, s in fused.items()]
    out.sort(key=lambda x: (-x[1], min(x[2].values()), x[0]))
    return out


def reciprocal_rank_fusion(dense_rankings: List[int], sparse_rankings: List[int], k: int = 60) -> List[Tuple[int, float]]:
    """Starter interface."""
    return [(d, s) for d, s, _ in rrf_fuse({"dense": dense_rankings, "sparse": sparse_rankings}, k)]


def score_fusion(dense: Dict[int, float], sparse: Dict[int, float], mode: str = "raw", alpha: float = 0.5) -> List[int]:
    """Baseline for comparison: add the scores directly (raw) or after min-max scaling."""
    docs = set(dense) | set(sparse)

    def scale(d):
        if mode != "minmax" or not d:
            return d
        lo, hi = min(d.values()), max(d.values())
        return {x: (v - lo) / (hi - lo) if hi > lo else 1.0 for x, v in d.items()}

    ds, ss = scale(dense), scale(sparse)
    total = {x: alpha * ds.get(x, 0.0) + (1 - alpha) * ss.get(x, 0.0) for x in docs}
    return sorted(docs, key=lambda x: (-total[x], x))


# ---------------------------------------------------------------------------
# dense side: toy encoder + HNSW from Part 1
# ---------------------------------------------------------------------------

# synonyms: documents use the first two words, queries mostly the last two
CONCEPTS = {
    "shield": ["shield", "deflector", "barrier", "screen"],
    "radiation": ["radiation", "synchrotron", "particle", "flux"],
    "coolant": ["coolant", "cryogenic", "chiller", "refrigerant"],
    "valve": ["valve", "regulator", "stopcock", "tap"],
    "reactor": ["reactor", "fusion", "core", "plasma"],
    "leak": ["leak", "rupture", "seepage", "breach"],
    "navigation": ["navigation", "astrogation", "heading", "course"],
    "sensor": ["sensor", "scanner", "detector", "probe"],
    "power": ["power", "capacitor", "energy", "grid"],
    "failure": ["failure", "fault", "malfunction", "outage"],
    "repair": ["repair", "recalibration", "fix", "restore"],
    "hull": ["hull", "plating", "bulkhead", "armour"],
    "oxygen": ["oxygen", "scrubber", "air", "atmosphere"],
    "engine": ["engine", "thruster", "propulsion", "drive"],
    "warp": ["warp", "hyperspace", "jump", "ftl"],
    "comms": ["transmitter", "antenna", "comms", "radio"],
    "archive": ["archive", "database", "memory", "records"],
    "heat": ["overheating", "thermal", "heat", "temperature"],
    "gravity": ["gravity", "inertial", "dampener", "stabiliser"],
    "docking": ["docking", "airlock", "hangar", "shuttle"],
    "medical": ["medical", "sickbay", "injury", "triage"],
    "crew": ["crew", "personnel", "staff", "officers"],
}
WORD2CONCEPT = {w: c for c, ws in CONCEPTS.items() for w in ws}


def _seed_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


class ToyConceptEncoder:
    """word vector = concept vector + some word noise; unknown word = mean of its char-trigram vectors.
    sentence vector = normalized sum of word vectors."""

    def __init__(self, dim: int = 256, synonym_noise: float = 0.7, unknown_weight: float = 0.6, seed: int = 0):
        self.dim, self.noise, self.unknown_weight, self.seed = dim, synonym_noise, unknown_weight, seed
        self._cache = {}

    def _rand(self, key: str) -> np.ndarray:
        v = np.random.default_rng(_seed_of(f"{self.seed}:{key}")).standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def word_vector(self, w: str) -> np.ndarray:
        if w not in self._cache:
            if w in WORD2CONCEPT:
                v = self._rand("concept:" + WORD2CONCEPT[w]) + self.noise * self._rand("word:" + w)
            else:
                padded = f"#{w}#"
                grams = [padded[i:i + 3] for i in range(len(padded) - 2)] or [padded]
                v = self.unknown_weight * np.mean([self._rand("tri:" + g) for g in grams], axis=0)
            self._cache[w] = v
        return self._cache[w]

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if toks:
                out[i] = np.sum([self.word_vector(w) for w in toks], axis=0)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def load_sibling(path: Path, name: str):
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"  (could not import {path.name}: {type(exc).__name__}: {exc})")
        return None


def find_hnsw_class():
    """Part 1's HNSW class (needs insert() and query())."""
    for path in HNSW_FILES:
        mod = load_sibling(path, "hnsw_index")
        if mod is None:
            continue
        for name in ("SimpleHNSW", "HNSWIndex", "HNSW"):
            cls = getattr(mod, name, None)
            if cls is not None and hasattr(cls, "insert") and hasattr(cls, "query"):
                return cls, f"{path.name}:{name}"
    return None, "brute-force cosine (Part 1 not found)"


class DenseRetriever:
    def __init__(self, doc_vectors: np.ndarray, use_hnsw: bool = True, M: int = 16, ef_construction: int = 100,
                 seed: int = 0):
        self.vectors = doc_vectors
        self.index, self.source = None, "brute-force cosine"
        if use_hnsw:
            cls, src = find_hnsw_class()
            if cls is not None:
                random.seed(seed)      # Part 1 uses the random module for the layers
                np.random.seed(seed)
                try:
                    idx = cls(dim=doc_vectors.shape[1], M=M, ef_construction=ef_construction)
                    for i, v in enumerate(doc_vectors):
                        idx.insert(i, v)
                    self.index, self.source = idx, src
                except Exception as exc:
                    print(f"  (HNSW from Part 1 failed: {type(exc).__name__}: {exc}; using brute force)")
            else:
                self.source = src

    def brute_force(self, q: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        sims = self.vectors @ (q / (np.linalg.norm(q) + 1e-12))
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(int(i), float(sims[i])) for i in order]

    def search(self, q: np.ndarray, top_k: int = 100) -> List[Tuple[int, float]]:
        if self.index is None:
            return self.brute_force(q, top_k)
        return [(int(i), float(s)) for i, s in self.index.query(q, top_k=top_k)]


# ---------------------------------------------------------------------------
# benchmark data
# ---------------------------------------------------------------------------

def make_code(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}"


def near_codes(rng: random.Random, prefix: str, n: int) -> List[str]:
    """n codes with the same digits in a different order (hard for dense, easy for BM25)."""
    base = list(str(rng.randint(1000, 9999)))
    codes = set()
    while len(codes) < n:
        digits = base[:]
        rng.shuffle(digits)
        codes.add(f"{prefix}-{''.join(digits)}-{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}")
    return sorted(codes)


def build_archive(n_families: int = 40, docs_per_family: int = 4, n_filler: int = 40, seed: int = 0,
                  query_overlap: float = 0.4):
    """Each family = same problem (3 concepts), different part codes.
    conceptual query -> relevant = whole family; code query -> relevant = that one document.
    query_overlap = chance a query uses the document's own word instead of a synonym."""
    rng = random.Random(seed)
    names = list(CONCEPTS)
    doc_side = {c: ws[:2] for c, ws in CONCEPTS.items()}
    query_side = {c: ws[2:] for c, ws in CONCEPTS.items()}
    doc_templates = [
        "Procedure {code}: {a} {b} {c} handling. Isolate the unit, log the {a} readings and follow checklist step 4.",
        "Maintenance bulletin {code} covers {b} {a} with {c}. Engineers must sign off the {b} log before restart.",
        "Technical note {code}. Observed {c} in the {a} {b} assembly; replacement kit stored in bay 7.",
        "Field manual entry {code}: {a} {c} affecting the {b}. Run self check, then report to engineering.",
    ]
    query_templates = [
        "how do we deal with {c} of the {a} {b}",
        "what to do about {a} {b} {c}",
        "instructions for {b} {c} in the {a}",
        "{a} {b} has a {c}, which procedure applies",
    ]
    prefixes = ["CV", "XR", "PL", "NV", "SH", "TQ", "MK", "DR", "AX", "QF"]
    docs, doc_family, doc_codes, families, used, queries = [], [], [], [], set(), []
    while len(families) < n_families:
        trio = tuple(sorted(rng.sample(names, 3)))
        if trio not in used:
            used.add(trio)
            families.append(trio)
    for fid, trio in enumerate(families):
        codes = near_codes(rng, rng.choice(prefixes), docs_per_family)
        members = []
        for code in codes:
            a, b, c = [rng.choice(doc_side[x]) for x in rng.sample(list(trio), 3)]
            docs.append(rng.choice(doc_templates).format(code=code, a=a, b=b, c=c))
            doc_family.append(fid)
            doc_codes.append(code)
            members.append(len(docs) - 1)
        words = [rng.choice(doc_side[x]) if rng.random() < query_overlap else rng.choice(query_side[x]) for x in trio]
        rng.shuffle(words)
        queries.append({"type": "conceptual", "text": rng.choice(query_templates).format(a=words[0], b=words[1], c=words[2]),
                        "relevant": members, "family": fid})
        target = rng.choice(members)
        queries.append({"type": "exact code", "text": rng.choice(["{code}", "part {code}", "lookup {code}",
                                                                    "serial {code}?"]).format(code=doc_codes[target]),
                        "relevant": [target], "family": fid})
    filler_topics = ["Crew rota for the next jump cycle", "Galley inventory and ration counts",
                     "Recreation deck schedule", "Botany lab watering plan", "Library reading list for cadets"]
    for i in range(n_filler):
        code = make_code(rng, "LOG")
        docs.append(f"Log {code}: {rng.choice(filler_topics)}, entry {i + 1}. Nothing technical to report.")
        doc_family.append(-1)
        doc_codes.append(code)
    return {"docs": docs, "family": doc_family, "codes": doc_codes, "queries": queries}


# ---------------------------------------------------------------------------
# metrics + benchmark
# ---------------------------------------------------------------------------

def mrr_at(ranked, relevant, k: int = 10) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def recall_at(ranked, relevant, k: int = 5) -> float:
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def ndcg_at(ranked, relevant, k: int = 10) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(rankings, queries):
    """rankings[method][query i] = ranked doc ids -> metrics[method][query type or 'all']"""
    out = {}
    types = sorted({q["type"] for q in queries}) + ["all"]
    for method, per_query in rankings.items():
        out[method] = {}
        for t in types:
            idx = [i for i, q in enumerate(queries) if t == "all" or q["type"] == t]
            out[method][t] = {
                "MRR@10": float(np.mean([mrr_at(per_query[i], queries[i]["relevant"], 10) for i in idx])),
                "Recall@5": float(np.mean([recall_at(per_query[i], queries[i]["relevant"], 5) for i in idx])),
                "NDCG@10": float(np.mean([ndcg_at(per_query[i], queries[i]["relevant"], 10) for i in idx])),
                "n": len(idx),
            }
    return out


def run_benchmark(seed: int = 0, k: int = 60, depth: int = 100, use_hnsw: bool = True, quiet: bool = False):
    say = (lambda *a: None) if quiet else print
    data = build_archive(seed=seed)
    docs, queries = data["docs"], data["queries"]
    enc = ToyConceptEncoder(seed=seed)
    doc_vecs = enc.encode(docs)
    q_vecs = enc.encode([q["text"] for q in queries])
    dense = DenseRetriever(doc_vecs, use_hnsw=use_hnsw, seed=seed)
    bm25 = BM25Okapi(docs)

    say(f"=== Ship archive benchmark: {len(docs)} documents, {len(queries)} queries "
        f"({sum(q['type'] == 'conceptual' for q in queries)} conceptual, "
        f"{sum(q['type'] == 'exact code' for q in queries)} exact code) ===")
    say(f"  dense search: {dense.source} | BM25 k1={bm25.k1}, b={bm25.b} | RRF k={k}")

    ranks = {"Dense (HNSW)": [], "BM25": [], "Hybrid RRF": [], "Raw score sum": [], "Min-max score sum": []}
    hnsw_recall, traces, max_bm25 = [], [], 0.0
    for qi, q in enumerate(queries):
        d_hits = dense.search(q_vecs[qi], top_k=depth)
        s_hits = bm25.search(q["text"], top_k=depth)
        max_bm25 = max([max_bm25] + [sc for _, sc in s_hits[:1]])
        fused = rrf_fuse({"dense": [d for d, _ in d_hits], "sparse": [d for d, _ in s_hits]}, k=k)
        ranks["Dense (HNSW)"].append([d for d, _ in d_hits])
        ranks["BM25"].append([d for d, _ in s_hits])
        ranks["Hybrid RRF"].append([d for d, _, _ in fused])
        ranks["Raw score sum"].append(score_fusion(dict(d_hits), dict(s_hits), "raw"))
        ranks["Min-max score sum"].append(score_fusion(dict(d_hits), dict(s_hits), "minmax"))
        if dense.index is not None:
            exact = {d for d, _ in dense.brute_force(q_vecs[qi], 10)}
            hnsw_recall.append(len(exact & {d for d, _ in d_hits[:10]}) / 10)
        traces.append({"dense": d_hits[:5], "sparse": s_hits[:5], "fused": fused[:5]})

    metrics = evaluate(ranks, queries)
    say(f"\n  {'method':<20}{'query type':<14}{'MRR@10':>8}{'Recall@5':>10}{'NDCG@10':>9}")
    for m in ("Dense (HNSW)", "BM25", "Hybrid RRF"):
        for t in ("conceptual", "exact code", "all"):
            r = metrics[m][t]
            say(f"  {m:<20}{t:<14}{r['MRR@10']:>8.3f}{r['Recall@5']:>10.3f}{r['NDCG@10']:>9.3f}")
        say("")
    if hnsw_recall:
        say(f"  HNSW recall@10 vs brute-force search: {np.mean(hnsw_recall):.3f}")

    # effect of the RRF constant k
    sweep = {}
    for kk in (1, 5, 20, 60, 100, 250, 1000):
        fused_all = [[d for d, _, _ in rrf_fuse({"dense": ranks["Dense (HNSW)"][qi], "sparse": ranks["BM25"][qi]}, k=kk)]
                     for qi in range(len(queries))]
        m = evaluate({"rrf": fused_all}, queries)["rrf"]
        sweep[kk] = {t: m[t]["MRR@10"] for t in ("conceptual", "exact code", "all")}
    say("\n=== RRF constant k (overall MRR@10) ===")
    say("  " + "  ".join(f"k={kk}: {v['all']:.3f}" for kk, v in sweep.items()))
    big = [v["all"] for kk, v in sweep.items() if kk >= 20]
    say(f"  for k >= 20 the result changes by only {max(big) - min(big):.3f}; k = 60 as in the brief")

    say("\n=== Rank fusion vs score fusion (MRR@10) ===")
    for m in ("Raw score sum", "Min-max score sum", "Hybrid RRF"):
        r = metrics[m]
        say(f"  {m:<20} conceptual {r['conceptual']['MRR@10']:.3f} | exact code {r['exact code']['MRR@10']:.3f} | "
            f"all {r['all']['MRR@10']:.3f}")
    best_all = max(("Raw score sum", "Min-max score sum", "Hybrid RRF"), key=lambda x: metrics[x]["all"]["MRR@10"])
    say(f"  best overall: {best_all}. BM25 scores go up to ~{max_bm25:.0f}, cosine only up to 1, so the raw sum")
    say("  is decided by BM25 whenever it matches anything. RRF only uses ranks.\n")

    # one example of each query type
    examples = []
    for t in ("exact code", "conceptual"):
        qi = next(i for i, q in enumerate(queries) if q["type"] == t)
        q, tr = queries[qi], traces[qi]
        rel = set(q["relevant"])
        say(f"=== Example ({t}): \"{q['text']}\"   relevant docs: {q['relevant']} ===")
        say(f"  {'rank':<6}{'dense (cosine)':<26}{'BM25 (score)':<26}{'RRF (dense rank / BM25 rank)':<32}")
        for r in range(5):
            cells = []
            for col, fmt in (("dense", "{:.3f}"), ("sparse", "{:.2f}")):
                if r < len(tr[col]):
                    d, sc = tr[col][r]
                    cells.append(f"doc {d:>3} ({fmt.format(sc)})" + (" *" if d in rel else ""))
                else:
                    cells.append("-")
            if r < len(tr["fused"]):
                d, sc, rr = tr["fused"][r]
                cells.append(f"doc {d:>3} {sc:.4f} ({rr.get('dense', '-')}/{rr.get('sparse', '-')})" + (" *" if d in rel else ""))
            else:
                cells.append("-")
            say(f"  {r + 1:<6}{cells[0]:<26}{cells[1]:<26}{cells[2]:<32}")
        say(f"  (* = relevant)  doc {q['relevant'][0]}: \"{docs[q['relevant'][0]][:85]}\"\n")
        examples.append({"type": t, "query": q["text"], "relevant": q["relevant"], "dense_top5": tr["dense"],
                         "bm25_top5": tr["sparse"], "rrf_top5": tr["fused"]})

    mc = lambda x: metrics[x]["conceptual"]["MRR@10"]
    me = lambda x: metrics[x]["exact code"]["MRR@10"]
    ma = lambda x: metrics[x]["all"]["MRR@10"]
    say("=== Summary ===")
    say(f"  - conceptual queries: dense {mc('Dense (HNSW)'):.2f} vs BM25 {mc('BM25'):.2f} MRR (paraphrases mostly use other words)")
    say(f"  - exact codes: BM25 {me('BM25'):.2f} vs dense {me('Dense (HNSW)'):.2f} (similar codes get similar vectors)")
    say(f"  - hybrid RRF: {mc('Hybrid RRF'):.2f} on conceptual, {me('Hybrid RRF'):.2f} on codes, {ma('Hybrid RRF'):.2f} overall")
    say(f"    (dense alone {ma('Dense (HNSW)'):.2f}, BM25 alone {ma('BM25'):.2f})")
    say(f"  - on paraphrases RRF is below dense alone ({mc('Hybrid RRF'):.2f} vs {mc('Dense (HNSW)'):.2f}): BM25's word matches")
    say("    on the wrong documents get an equal vote\n")

    return {"n_docs": len(docs), "n_queries": len(queries), "dense_source": dense.source, "k": k, "depth": depth,
            "metrics": metrics, "k_sweep": sweep,
            "hnsw_recall_at_10": float(np.mean(hnsw_recall)) if hnsw_recall else None, "examples": examples}


def plot_benchmark(res, path: Path = DEFAULT_FIGURE):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed - pip install matplotlib")
        return None
    m = res["metrics"]
    types = ["conceptual", "exact code", "all"]

    def bars(a, methods, colors, metric, title, labels):
        w = 0.26
        for j, meth in enumerate(methods):
            vals = [m[meth][t][metric] for t in types]
            bs = a.bar([i + (j - 1) * w for i in range(len(types))], vals, w, label=meth, color=colors[j])
            for bar, v in zip(bs, vals):
                a.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
        a.set_xticks(range(len(types)))
        a.set_xticklabels(labels)
        a.set_ylim(0, 1.25)
        a.set_ylabel(metric)
        a.set_title(title)
        a.grid(axis="y", alpha=0.3)
        a.legend(fontsize=8, loc="upper center", ncol=3)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    main = ["Dense (HNSW)", "BM25", "Hybrid RRF"]
    main_c = ["#0969da", "#bf8700", "#1a7f37"]
    long_labels = ["conceptual\n(paraphrased)", "exact code\n(e.g. CV-4471-K)", "all queries"]
    bars(ax[0, 0], main, main_c, "MRR@10", "(a) MRR@10 by query type", long_labels)
    bars(ax[0, 1], main, main_c, "Recall@5", "(b) Recall@5 by query type", long_labels)

    c = ax[1, 0]
    ks = list(res["k_sweep"])
    for t, col, mk in (("conceptual", "#0969da", "o"), ("exact code", "#bf8700", "s"), ("all", "#1a7f37", "^")):
        c.plot(ks, [res["k_sweep"][kk][t] for kk in ks], marker=mk, color=col, label=t)
    c.axvline(60, color="black", lw=0.8, ls="--")
    c.text(62, 0.05, "k = 60", fontsize=9)
    c.set_xscale("log")
    c.set_ylim(0, 1.05)
    c.set_xlabel("RRF constant k (log scale)")
    c.set_ylabel("MRR@10")
    c.set_title("(c) Effect of the RRF constant k")
    c.grid(alpha=0.3)
    c.legend(fontsize=8, loc="lower right")

    bars(ax[1, 1], ["Raw score sum", "Min-max score sum", "Hybrid RRF"], ["#cf222e", "#8250df", "#1a7f37"],
         "MRR@10", "(d) Rank fusion vs. adding raw / min-max scores", ["conceptual", "exact code", "all queries"])

    fig.suptitle(f"Mission 05 / Part 2 - BM25 + dense retrieval fused with RRF (k={res['k']}), "
                 f"{res['n_docs']} archive documents, {res['n_queries']} queries; dense = {res['dense_source']}",
                 fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self, quiet: bool = False):
        self.items = []
        self.quiet = quiet

    def add(self, name: str, ok: bool, detail: str = ""):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        if not self.quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self):
        return sum(not c["ok"] for c in self.items)

    @property
    def passed(self):
        return sum(c["ok"] for c in self.items)


def starter_bm25_scores(corpus, query, k1=1.5, b=0.75, tok=tokenize):
    """The starter's loop over every document, to check the inverted index against."""
    tc = [tok(d) for d in corpus]
    n = len(tc)
    avgdl = sum(map(len, tc)) / n
    df = Counter(t for d in tc for t in set(d))
    out = []
    for d in tc:
        tf = Counter(d)
        s = 0.0
        for t in tok(query):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def self_test(seed: int = 0, quiet: bool = False) -> Checks:
    c = Checks(quiet)
    say = (lambda *a: None) if quiet else print
    say("=== Part 2 self-test: BM25 + Reciprocal Rank Fusion ===")
    split = lambda s: s.split()

    # tokenizer
    toks = tokenize("Run procedure CV-4471-K, then ALPHA-9. Check v2.1!")
    c.add("tokenizer keeps serial codes whole and strips punctuation",
          toks == ["run", "procedure", "cv-4471-k", "then", "alpha-9", "check", "v2.1"], str(toks))
    starter = [d.lower().split() for d in ["plasma code ALPHA-9.", "code ALPHA-9"]]
    c.add("starter .split() would miss 'ALPHA-9.' at the end of a sentence",
          "alpha-9" not in starter[0] and "alpha-9" in starter[1])
    bm = BM25Okapi(["plasma containment code ALPHA-9.", "plasma leak in the reactor", "galley schedule"])
    c.add("query 'alpha-9' finds the document even with the trailing full stop",
          bm.search("ALPHA-9")[0][0] == 0 and len(bm.search("ALPHA-9")) == 1)

    # formula by hand
    bm = BM25Okapi(["red red blue", "blue green", "green green green yellow", "red"], tokenizer=split)
    n, avgdl = 4, (3 + 2 + 4 + 1) / 4
    idf_red = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    hand = idf_red * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl))
    c.add("IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))", abs(bm.idf["red"] - idf_red) < 1e-12)
    c.add("BM25 matches the formula on a hand-computed example", abs(bm.get_scores("red")[0] - hand) < 1e-12,
          f"{hand:.6f}")
    c.add("k1 = 1.5 and b = 0.75 by default", bm.k1 == 1.5 and bm.b == 0.75)

    # saturation, length normalization, k1 = 0
    one = BM25Okapi(["x " * f + "pad " * (10 - f) for f in range(1, 10)] + ["other doc"], tokenizer=split)
    sc = [one.get_scores("x")[i] for i in range(9)]
    c.add("term frequency saturates: score rises with f but stays below IDF * (k1 + 1)",
          all(a < b for a, b in zip(sc, sc[1:])) and max(sc) < one.idf["x"] * (one.k1 + 1)
          and (sc[-1] - sc[-2]) < (sc[1] - sc[0]), f"gain f1->f2 {sc[1] - sc[0]:.3f}, f8->f9 {sc[-1] - sc[-2]:.3f}")
    ln = BM25Okapi(["x a", "x a b c d e f g", "z"], tokenizer=split)
    ln0 = BM25Okapi(["x a", "x a b c d e f g", "z"], b=0.0, tokenizer=split)
    c.add("length normalization: same f, longer document scores lower; b = 0 turns it off",
          ln.get_scores("x")[0] > ln.get_scores("x")[1] and abs(ln0.get_scores("x")[0] - ln0.get_scores("x")[1]) < 1e-12)
    k0 = BM25Okapi(["x x x x", "x", "z"], k1=0.0, tokenizer=split)
    c.add("k1 = 0 makes BM25 binary (only presence counts)", abs(k0.get_scores("x")[0] - k0.get_scores("x")[1]) < 1e-12)
    c.add("rare terms weigh more than common ones (IDF)", bm.idf["yellow"] > bm.idf["red"] > 0)

    # inverted index == starter loop
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(30)]
    corp = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 15))) for _ in range(60)]
    big = BM25Okapi(corp)
    worst = 0.0
    for _ in range(20):
        q = " ".join(rng.choice(words) for _ in range(3))
        worst = max(worst, float(np.max(np.abs(big.get_scores(q) - np.array(starter_bm25_scores(corp, q))))))
    c.add("inverted-index scoring == the starter's loop over every document", worst < 1e-10, f"max diff {worst:.1e}")
    out = big.score("w1 w2")
    c.add("score() keeps the starter's output: every doc as (idx, score), best first",
          len(out) == 60 and all(out[i][1] >= out[i + 1][1] for i in range(59)))

    # RRF
    fused = rrf_fuse({"dense": [7, 3, 9], "sparse": [7, 5]}, k=60)
    got = {d: s for d, s, _ in fused}
    c.add("RRF: rank 1 in both lists = 1/61 + 1/61", abs(got[7] - 2 / 61) < 1e-15)
    c.add("RRF: a document in only one list gets only that term (doc 9: 1/63)", abs(got[9] - 1 / 63) < 1e-15)
    c.add("RRF ranks are 1-based and recorded per method", fused[0][2] == {"dense": 1, "sparse": 1})
    a = rrf_fuse({"dense": [1, 2, 3], "sparse": [3, 1, 4]})
    b = rrf_fuse({"sparse": [3, 1, 4], "dense": [1, 2, 3]})
    c.add("RRF does not depend on the order of the input lists", [x[:2] for x in a] == [x[:2] for x in b])
    tie = rrf_fuse({"dense": [1, 2], "sparse": [2, 1]})
    c.add("ties are broken the same way every time (best single rank, then doc id)", [d for d, _, _ in tie] == [1, 2])
    c.add("reciprocal_rank_fusion() (starter interface) gives the same result",
          [d for d, _ in reciprocal_rank_fusion([7, 3, 9], [7, 5])] == [d for d, _, _ in fused])
    dense_s, sparse_s = {1: 0.95, 2: 0.30}, {2: 0.5, 1: 0.4}
    sparse_big = {x: v * 1000 for x, v in sparse_s.items()}
    order = lambda d: sorted(d, key=lambda x: -d[x])
    rrf1 = [d for d, _, _ in rrf_fuse({"d": order(dense_s), "s": order(sparse_s)})]
    rrf2 = [d for d, _, _ in rrf_fuse({"d": order(dense_s), "s": order(sparse_big)})]
    c.add("RRF ignores score scale (BM25 x1000 -> same ranking); raw score sum flips",
          rrf1 == rrf2 and score_fusion(dense_s, sparse_s) == [1, 2] and score_fusion(dense_s, sparse_big) == [2, 1])

    c.add("MRR / Recall@k / NDCG@k on a hand example",
          abs(mrr_at([4, 2, 9], [2]) - 0.5) < 1e-12 and abs(recall_at([4, 2, 9], [2, 9, 5], 2) - 1 / 3) < 1e-12
          and abs(ndcg_at([2, 4], [2]) - 1.0) < 1e-12 and abs(ndcg_at([4, 2], [2]) - 1 / math.log2(3)) < 1e-12)

    # encoder
    enc = ToyConceptEncoder(seed=seed)
    v = enc.encode(["deflector recalibration", "screen fix", "galley schedule", "CV-4471-K", "CV-4417-K", "XR-2290-B"])
    cs = v @ v.T
    c.add("dense encoder: paraphrases are close, unrelated text is not",
          cs[0, 1] > 0.6 and cs[0, 2] < 0.3, f"paraphrase {cs[0, 1]:.2f}, unrelated {cs[0, 2]:.2f}")
    c.add("dense encoder: near-identical part codes get similar vectors (why dense search misses them)",
          cs[3, 4] > 0.4 and cs[3, 4] > cs[3, 5] + 0.3, f"CV-4471-K vs CV-4417-K {cs[3, 4]:.2f}, vs XR-2290-B {cs[3, 5]:.2f}")

    # HNSW from Part 1
    rngv = np.random.default_rng(seed)
    vecs = rngv.standard_normal((400, 32))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dr = DenseRetriever(vecs, seed=seed)
    if dr.index is not None:
        rec = np.mean([len({d for d, _ in dr.search(q, 10)} & {d for d, _ in dr.brute_force(q, 10)}) / 10
                       for q in rngv.standard_normal((30, 32))])
        c.add(f"dense search through Part 1's HNSW ({dr.source}): recall@10 vs brute force", rec >= 0.85, f"{rec:.2f}")
    else:
        c.add("Part 1 HNSW not found, brute-force cosine search used instead", True, dr.source)

    # benchmark claims from the brief
    m = run_benchmark(seed=seed, quiet=True)["metrics"]
    dense_c, bm25_c, rrf_c = (m[x]["conceptual"]["MRR@10"] for x in ("Dense (HNSW)", "BM25", "Hybrid RRF"))
    dense_e, bm25_e, rrf_e = (m[x]["exact code"]["MRR@10"] for x in ("Dense (HNSW)", "BM25", "Hybrid RRF"))
    c.add("conceptual queries: dense search beats BM25", dense_c > bm25_c + 0.2, f"{dense_c:.2f} vs {bm25_c:.2f}")
    c.add("exact-code queries: BM25 beats dense search", bm25_e > dense_e + 0.2, f"{bm25_e:.2f} vs {dense_e:.2f}")
    c.add("hybrid RRF retrieves both kinds (>= 0.7 MRR on each, far above the weaker method)",
          min(rrf_c, rrf_e) >= 0.7 and rrf_c > bm25_c + 0.2 and rrf_e > dense_e + 0.5,
          f"conceptual {rrf_c:.2f}, exact code {rrf_e:.2f}")
    c.add("hybrid RRF is the best method over all queries",
          m["Hybrid RRF"]["all"]["MRR@10"] > max(m["Dense (HNSW)"]["all"]["MRR@10"], m["BM25"]["all"]["MRR@10"]))

    say(f"--- {c.passed} passed, {c.failed} failed ---\n")
    return c


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def starter_demo():
    """The starter's example, with the new tokenizer."""
    docs = [
        "AURA Subsystem 05 contains the dual-path associative memory core",
        "Pulsar radiation in Crab nebula induces high frequency electromagnetic interference",
        "Warp drive emergency plasma containment procedure code ALPHA-9",
        "Telemetry logs from Sirius waypoint recorded zero anomalies",
        "Relativistic particles from neutron star require magnetic shield recalibration",
    ]
    bm25 = BM25Okapi(docs)
    results = bm25.score("plasma containment emergency ALPHA-9")
    print("=== Subsystem 05: BM25 Lexical Retrieval Scores ===")
    for doc_id, score in results[:3]:
        print(f"Doc {doc_id} (Score: {score:.4f}): '{docs[doc_id]}'")
    print("\n=== Reciprocal Rank Fusion Combined Rankings ===")
    for doc_id, rrf_val in reciprocal_rank_fusion([2, 0, 1, 4, 3], [d for d, _ in results], k=60)[:3]:
        print(f"Doc {doc_id} -> RRF Score: {rrf_val:.6f}")
    print()


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_log(mode: str, payload) -> Path:
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps05-bm25-rrf-{mode}.json"
    path.write_text(json.dumps(to_json(payload), indent=2))
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Mission 05 Part 2 - BM25 + Reciprocal Rank Fusion")
    p.add_argument("--selftest", action="store_true", help="run the tests only")
    p.add_argument("--demo", action="store_true", help="run the starter's example only")
    p.add_argument("--k", type=int, default=60, help="RRF constant")
    p.add_argument("--no-hnsw", action="store_true", help="brute-force cosine search instead of Part 1's HNSW")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_FIGURE, help="figure path")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-log", action="store_true", help="don't write logs/*.json")
    args = p.parse_args(argv)

    if args.demo:
        starter_demo()
        return 0
    print("=" * 78)
    print("USSE NOOSPHERE - DUAL-PATH RETRIEVAL: BM25 + RECIPROCAL RANK FUSION (Part 2)")
    print("=" * 78)
    checks = self_test(args.seed, quiet=args.quiet)
    payload = {"checks": checks.items, "passed": checks.passed, "failed": checks.failed}
    if not args.selftest:
        res = run_benchmark(seed=args.seed, k=args.k, use_hnsw=not args.no_hnsw, quiet=args.quiet)
        payload["benchmark"] = res
        fig = plot_benchmark(res, args.out)
        if fig is not None:
            payload["figure"] = str(fig)
            if not args.quiet:
                print(f"Benchmark figure written to {fig}")
    if not args.no_log:
        log = write_log("selftest" if args.selftest else "all", payload)
        if not args.quiet:
            print(f"Log written to {log.relative_to(HERE)}")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())