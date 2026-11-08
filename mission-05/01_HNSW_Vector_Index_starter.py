"""
Mission 05 - Part 1: Systems Architect (Lead Engineer)
Hierarchical Navigable Small World (HNSW) Vector Proximity Index

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core
Squadron Bengio - Systems Architect: Fahad Akram

Pure Python + NumPy implementation of Malkov & Yashunin (2018), no faiss / hnswlib / sklearn.

  cosine_similarity(u, v)            scalar cosine of two vectors (starter name kept)
  cosine_similarity_matrix(Q, X)     vectorised: every row of Q against every row of X
  SimpleHNSW(dim, M, ef_construction, ...)
      _random_level()                l = floor(-ln(uniform(0, 1)) * m_L),  m_L = 1 / ln(M)
      search_layer(entry_point, query, ef, layer)
                                     ef-bounded best-first search inside one layer (Algorithm 2)
      insert(node_id, vector)        greedy descent through the upper layers, then connect at every
                                     layer <= l with bidirectional edges and degree pruning (Algorithm 1)
      query(vector, top_k, ef)       greedy hops along the long-range edges of the top layers, then a
                                     search_layer(ef) sweep on layer 0 (Algorithm 5)
      n_dist_calcs                   counter of vector comparisons, the quantity that is O(log N)
  brute_force_search(X, q, top_k)    the flat baseline: one matrix-vector product over all N rows

Modes
  python 01_HNSW_Vector_Index_starter.py                 demo on 2,000 x 64 vectors + small scaling table
  python 01_HNSW_Vector_Index_starter.py --selftest      35 checks (metric, layers, graph invariants, exact
                                                         search at ef = N, recall, scaling, integration)
  python 01_HNSW_Vector_Index_starter.py --bench         N = 25,000 x 768: build, latency, recall, distance
                                                         computations, HNSW vs flat at 1k ... 25k
  python 01_HNSW_Vector_Index_starter.py --check-signoffs
Every run writes logs/<timestamp>-ps05-<mode>.json.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
PART2 = HERE / "02_BM25_and_RRF_Combiner.py"
PART4 = HERE / "04_Prompt_Injection_and_RAG_Audit_starter.py"


# ---------------------------------------------------------------------------
# 1. Cosine similarity core
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray) -> np.ndarray:
    """Unit-length rows (a zero vector stays zero instead of producing NaN)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Sim(u, v) = u.v / (|u| |v|) for two 1-D vectors; 0 if either is the zero vector."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def cosine_similarity_matrix(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(q, d) x (n, d) -> (q, n) cosine similarities in one matrix product."""
    return normalize(np.atleast_2d(Q)) @ normalize(np.atleast_2d(X)).T


def brute_force_search(X_unit: np.ndarray, q: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    """Flat search: one matrix-vector product over all N unit rows, then a partial sort. O(N d)."""
    sims = X_unit @ normalize(q)
    k = min(top_k, sims.shape[0])
    if k == 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return [(int(i), float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# 2. HNSW
# ---------------------------------------------------------------------------

class SimpleHNSW:
    """
    Multi-layer navigable small-world graph over unit vectors with cosine similarity.

    Layer l holds every node whose assigned level is >= l; layer 0 holds all of them. Upper layers are
    sparse "express lanes": a query hops greedily to the nearest node on each layer, descends, and only
    on layer 0 does a bounded best-first search (ef candidates) collect the final neighbours. Nodes keep
    at most M neighbours per upper layer and M_max0 = 2M on layer 0 (the paper's defaults).

    Interface used by the other stations: SimpleHNSW(dim=, M=, ef_construction=), insert(id, vector),
    query(vector, top_k=) -> [(id, cosine), ...]. The `random` module draws the layer levels, so
    random.seed(...) makes a build reproducible (a private Random is used when seed= is given).
    """

    def __init__(self, dim: int = 768, M: int = 16, ef_construction: int = 64, M_max0: Optional[int] = None,
                 ef_search: Optional[int] = None, seed: Optional[int] = None, heuristic: bool = True):
        if M < 2:
            raise ValueError("M must be at least 2")
        self.heuristic = heuristic      # Algorithm 4 (diverse neighbours) vs Algorithm 3 (plain top-M)
        self.dim = int(dim)
        self.M = int(M)
        self.M_max0 = int(M_max0) if M_max0 else 2 * self.M
        self.ef_construction = max(int(ef_construction), self.M)
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(self.M)
        self._rng = random.Random(seed) if seed is not None else None
        self.entry_point: int = -1          # internal row index of the entry node
        self.max_level: int = -1
        self._data = np.zeros((1024, self.dim), dtype=np.float32)   # unit vectors, grows by doubling
        self._n = 0
        self._ids: List[int] = []           # row -> external id
        self._rows: Dict[int, int] = {}     # external id -> row
        self._levels: List[int] = []        # row -> assigned level
        self.layers: List[Dict[int, List[int]]] = []   # layer -> {row: [neighbour rows]}
        self.n_dist_calcs = 0

    # -- helpers -----------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    @property
    def vectors(self) -> np.ndarray:
        """The stored unit vectors, row i belongs to id self.ids[i]."""
        return self._data[: self._n]

    @property
    def ids(self) -> List[int]:
        return self._ids

    def _random_level(self) -> int:
        """l = floor(-ln(U) * m_L), U ~ uniform(0, 1): P(level >= k) = M^-k."""
        rnd = self._rng.random if self._rng is not None else random.random
        u = rnd()
        while u == 0.0:
            u = rnd()
        return int(math.floor(-math.log(u) * self.m_L))

    def _sims(self, rows: Sequence[int], q: np.ndarray) -> np.ndarray:
        """Cosine of the unit query against several stored unit rows in one product; counts them."""
        self.n_dist_calcs += len(rows)
        return self._data[rows] @ q

    def _m_max(self, layer: int) -> int:
        return self.M_max0 if layer == 0 else self.M

    # -- Algorithm 2: SEARCH-LAYER -------------------------------------------------
    def search_layer(self, entry_point, query: np.ndarray, ef: int, layer: int) -> List[Tuple[float, int]]:
        """
        Best-first search restricted to one layer. Starts from one or several entry rows, keeps the ef
        best nodes seen so far, and stops when the closest unexpanded candidate is worse than the worst
        of those ef. Returns [(similarity, row)] sorted best first (at most ef entries).
        """
        eps = [entry_point] if isinstance(entry_point, (int, np.integer)) else list(entry_point)
        q = np.asarray(query, dtype=np.float32)
        adj = self.layers[layer]
        visited = set(eps)
        ep_sims = self._sims(eps, q)
        candidates = [(-float(s), r) for s, r in zip(ep_sims, eps)]   # max-heap on similarity
        heapq.heapify(candidates)
        results = [(float(s), r) for s, r in zip(ep_sims, eps)]       # min-heap: results[0] is the worst
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break                                                  # nothing closer can be reached
            fresh = [r for r in adj.get(node, ()) if r not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for s, r in zip(self._sims(fresh, q), fresh):
                s = float(s)
                if len(results) < ef:
                    heapq.heappush(results, (s, r))
                    heapq.heappush(candidates, (-s, r))
                elif s > results[0][0]:
                    heapq.heapreplace(results, (s, r))
                    heapq.heappush(candidates, (-s, r))
        return sorted(results, reverse=True)

    def _greedy_closest(self, entry_point: int, query: np.ndarray, layer: int) -> int:
        """Hop to the best neighbour until no neighbour improves (search_layer with ef = 1, cheaper)."""
        q = np.asarray(query, dtype=np.float32)
        best = entry_point
        best_sim = float(self._sims([best], q)[0])
        while True:
            nbrs = self.layers[layer].get(best, [])
            if not nbrs:
                return best
            sims = self._sims(nbrs, q)
            j = int(np.argmax(sims))
            if sims[j] <= best_sim:
                return best
            best, best_sim = nbrs[j], float(sims[j])

    # -- Algorithm 1: INSERT ---------------------------------------------------------
    def _select_neighbors(self, qvec: np.ndarray, cands: List[Tuple[float, int]], m: int) -> List[int]:
        """
        Pick at most m neighbours for a node with vector `qvec` out of candidates [(sim to qvec, row)].

        Algorithm 3 (heuristic=False): simply the m most similar.
        Algorithm 4 (heuristic=True): walk the candidates best first and keep one only if it is closer to
        the node than to every neighbour already kept; discarded candidates fill the remaining slots
        (keepPrunedConnections). The kept set is spread in direction, so dense clusters do not swallow
        all M edges and the graph stays connected across clusters - the navigability HNSW relies on.
        """
        ordered = sorted(cands, reverse=True)
        if not self.heuristic:
            return [r for _, r in ordered[:m]]
        selected: List[int] = []
        discarded: List[int] = []
        for sim, r in ordered:
            if len(selected) >= m:
                break
            if not selected:
                selected.append(r)
                continue
            self.n_dist_calcs += len(selected)
            closest_selected = float((self._data[selected] @ self._data[r]).max())
            if sim > closest_selected:
                selected.append(r)
            else:
                discarded.append(r)
        for r in discarded:
            if len(selected) >= m:
                break
            selected.append(r)
        return selected

    def _prune(self, row: int, layer: int) -> None:
        """Shrink the neighbour list of `row` on `layer` to M_max with the same selection rule."""
        nbrs = self.layers[layer][row]
        m_max = self._m_max(layer)
        if len(nbrs) <= m_max:
            return
        sims = self._sims(nbrs, self._data[row])
        self.layers[layer][row] = self._select_neighbors(self._data[row], list(zip(sims.tolist(), nbrs)), m_max)

    def insert(self, node_id: int, vector: np.ndarray) -> int:
        """Insert one vector under an external id. Returns the level the node was assigned."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"vector has dimension {vec.shape[0]}, index expects {self.dim}")
        node_id = int(node_id)
        if node_id in self._rows:
            raise ValueError(f"id {node_id} is already in the index")
        if self._n == self._data.shape[0]:
            self._data = np.vstack([self._data, np.zeros_like(self._data)])
        row = self._n
        self._data[row] = normalize(vec)
        self._n += 1
        self._ids.append(node_id)
        self._rows[node_id] = row
        level = self._random_level()
        self._levels.append(level)
        q = self._data[row]

        while len(self.layers) <= level:
            self.layers.append({})
        for l in range(level + 1):
            self.layers[l][row] = []

        if self.entry_point == -1:
            self.entry_point, self.max_level = row, level
            return level

        # 1. greedy descent through the layers above the new node's level
        ep = self.entry_point
        for l in range(self.max_level, level, -1):
            ep = self._greedy_closest(ep, q, l)

        # 2. connect on every layer from min(level, max_level) down to 0
        eps: List[int] = [ep]
        for l in range(min(level, self.max_level), -1, -1):
            cands = self.search_layer(eps, q, self.ef_construction, l)
            neighbors = self._select_neighbors(q, cands, self.M)
            self.layers[l][row] = list(neighbors)
            for nb in neighbors:
                self.layers[l][nb].append(row)
                self._prune(nb, l)                # shrink connections that exceed M_max
            eps = [r for _, r in cands]           # the ef best become the entry points of the next layer

        if level > self.max_level:
            self.entry_point, self.max_level = row, level
        return level

    def add_items(self, vectors: np.ndarray, ids: Optional[Iterable[int]] = None) -> None:
        vectors = np.atleast_2d(vectors)
        ids = range(self._n, self._n + len(vectors)) if ids is None else list(ids)
        for i, v in zip(ids, vectors):
            self.insert(i, v)

    # -- Algorithm 5: K-NN-SEARCH ----------------------------------------------------
    def query(self, query_vec: np.ndarray, top_k: int = 5, ef: Optional[int] = None) -> List[Tuple[int, float]]:
        """Top-k most similar stored vectors as [(external id, cosine)], best first."""
        if self._n == 0:
            return []
        q = normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dimension {q.shape[0]}, index expects {self.dim}")
        ef = max(ef or self.ef_search or self.ef_construction, top_k)
        ep = self.entry_point
        for l in range(self.max_level, 0, -1):            # long-range hops on the sparse layers
            ep = self._greedy_closest(ep, q, l)
        cands = self.search_layer(ep, q, ef, 0)           # dense sweep on layer 0
        return [(self._ids[r], s) for s, r in cands[:top_k]]

    search = query

    # -- introspection -----------------------------------------------------------------
    def layer_sizes(self) -> List[int]:
        return [len(layer) for layer in self.layers]

    def stats(self) -> Dict[str, Any]:
        degs0 = [len(v) for v in self.layers[0].values()] if self.layers else []
        edges = sum(len(v) for layer in self.layers for v in layer.values())
        return {
            "n": self._n, "dim": self.dim, "M": self.M, "M_max0": self.M_max0, "ef_construction": self.ef_construction,
            "m_L": self.m_L, "max_level": self.max_level, "layer_sizes": self.layer_sizes(),
            "entry_point_id": self._ids[self.entry_point] if self.entry_point >= 0 else None,
            "avg_degree_layer0": float(np.mean(degs0)) if degs0 else 0.0,
            "max_degree_layer0": max(degs0) if degs0 else 0,
            "graph_memory_mb": edges * 8 / 2 ** 20, "vector_memory_mb": self._n * self.dim * 4 / 2 ** 20,
        }

    def connected_component_size(self, layer: int = 0) -> int:
        """BFS from the entry point restricted to one layer (used by the self-test)."""
        if self._n == 0:
            return 0
        adj = self.layers[layer]
        start = next(iter(adj)) if self.entry_point not in adj else self.entry_point
        seen, stack = {start}, [start]
        while stack:
            r = stack.pop()
            for nb in adj[r]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen)


# ---------------------------------------------------------------------------
# 3. Data, recall, benchmark
# ---------------------------------------------------------------------------

def make_embeddings(n: int, dim: int, seed: int = 0, kind: str = "clustered", n_clusters: int = 200,
                    spread: float = 0.35, intrinsic_dim: Optional[int] = 32) -> np.ndarray:
    """
    Synthetic 'embeddings' as unit rows.
      clustered (default): a Gaussian mixture (topics) generated in a latent space of `intrinsic_dim`
          dimensions and projected into `dim` dimensions with a fixed random map, plus a little isotropic
          noise. Real text embeddings have a low intrinsic dimension (tens), which is what makes
          graph search navigable; this reproduces that.
      random: isotropic Gaussian in all `dim` dimensions - the hardest case for every ANN index, because
          in 768-d all distances concentrate and "nearest" barely means anything.
    """
    rng = np.random.default_rng(seed)
    if kind == "random":
        return normalize(rng.standard_normal((n, dim), dtype=np.float32))
    latent = min(intrinsic_dim or dim, dim)
    centers = normalize(rng.standard_normal((n_clusters, latent), dtype=np.float32))
    labels = rng.integers(0, n_clusters, n)
    noise = rng.standard_normal((n, latent), dtype=np.float32) / math.sqrt(latent)     # norm ~ 1
    Z = centers[labels] + spread * noise                                               # noise norm ~ spread
    if latent == dim:
        return normalize(Z)
    proj = rng.standard_normal((latent, dim), dtype=np.float32) / math.sqrt(dim)      # fixed random map
    X = Z @ proj + 0.02 * rng.standard_normal((n, dim), dtype=np.float32) / math.sqrt(dim)
    return normalize(X)


def recall_at_k(index: SimpleHNSW, X_unit: np.ndarray, queries: np.ndarray, k: int = 10,
                ef: Optional[int] = None) -> float:
    hits = 0
    for q in queries:
        truth = {i for i, _ in brute_force_search(X_unit, q, k)}
        got = {i for i, _ in index.query(q, top_k=k, ef=ef)}
        hits += len(truth & got)
    return hits / (k * len(queries))


def timed(fn, *args, **kw) -> Tuple[Any, float]:
    t = time.perf_counter()
    out = fn(*args, **kw)
    return out, (time.perf_counter() - t) * 1000.0


def latency_stats(samples_ms: List[float]) -> Dict[str, float]:
    s = sorted(samples_ms)
    return {"mean_ms": statistics.fmean(s), "median_ms": statistics.median(s),
            "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "min_ms": s[0]}


def benchmark(n: int = 25_000, dim: int = 768, n_queries: int = 100, M: int = 16, ef_construction: int = 64,
              ef_search: int = 64, k: int = 10, checkpoints: Sequence[int] = (1_000, 2_500, 5_000, 10_000, 25_000),
              seed: int = 0, kind: str = "clustered", quiet: bool = False) -> Dict[str, Any]:
    """
    One index is built up to n vectors; at each checkpoint (index size) the same queries are run against
    HNSW and against flat search over the same first-n rows, so the scaling of latency, recall and the
    number of vector comparisons is read off a single build.
    """
    say = (lambda *a, **kw: None) if quiet else print
    X = make_embeddings(n, dim, seed=seed, kind=kind)
    rng = np.random.default_rng(seed + 1)
    # queries: perturbed database vectors (realistic) plus a few fresh ones
    base = X[rng.integers(0, n, n_queries)]
    Q = normalize(base + 0.15 * rng.standard_normal(base.shape, dtype=np.float32))
    random.seed(seed)
    index = SimpleHNSW(dim=dim, M=M, ef_construction=ef_construction, ef_search=ef_search)
    checkpoints = sorted({c for c in checkpoints if c <= n} | {n})
    rows: List[Dict[str, Any]] = []
    say(f"Building HNSW over N = {n:,} x {dim} ({kind} unit vectors), M = {M}, ef_construction = {ef_construction}")
    say(f"  {'N':>7} | {'build s':>8} | {'HNSW ms':>8} | {'flat ms':>8} | {'speed-up':>8} | {'recall@%d' % k:>9} |"
        f" {'dist/query':>10} | {'flat dist':>9} | layers")
    t_build = 0.0
    done = 0
    for cp in checkpoints:
        t = time.perf_counter()
        for i in range(done, cp):
            index.insert(i, X[i])
        t_build += time.perf_counter() - t
        done = cp
        Xn = X[:cp]
        index.n_dist_calcs = 0
        h_ms, f_ms, hits = [], [], 0
        for q in Q:
            got, ms = timed(index.query, q, k)
            h_ms.append(ms)
            truth, ms2 = timed(brute_force_search, Xn, q, k)
            f_ms.append(ms2)
            hits += len({i for i, _ in got} & {i for i, _ in truth})
        dist_per_query = index.n_dist_calcs / len(Q)
        h, f = latency_stats(h_ms), latency_stats(f_ms)
        row = {"N": cp, "build_s_cumulative": t_build, "hnsw": h, "flat": f, "speedup_median": f["median_ms"] / h["median_ms"],
               "recall": hits / (k * len(Q)), "dist_calcs_per_query": dist_per_query, "flat_dist_calcs": cp,
               "layer_sizes": index.layer_sizes()}
        rows.append(row)
        say(f"  {cp:>7,} | {t_build:>8.1f} | {h['median_ms']:>8.3f} | {f['median_ms']:>8.3f} | {row['speedup_median']:>7.1f}x |"
            f" {row['recall']:>9.3f} | {dist_per_query:>10.0f} | {cp:>9,} | {index.layer_sizes()}")

    # ef sweep at full size: the recall / latency knob
    sweep = []
    for ef in (16, 32, 64, 128, 256):
        index.n_dist_calcs = 0
        ms, hits = [], 0
        for q in Q:
            got, t_ms = timed(index.query, q, k, ef)
            ms.append(t_ms)
            hits += len({i for i, _ in got} & {i for i, _ in brute_force_search(X, q, k)})
        sweep.append({"ef": ef, "recall": hits / (k * len(Q)), "median_ms": statistics.median(ms),
                      "dist_calcs_per_query": index.n_dist_calcs / len(Q)})
    say(f"\nef_search sweep at N = {n:,} (recall vs latency knob at query time)")
    say(f"  {'ef':>4} | {'recall@%d' % k:>9} | {'median ms':>9} | {'dist/query':>10}")
    for s in sweep:
        say(f"  {s['ef']:>4} | {s['recall']:>9.3f} | {s['median_ms']:>9.3f} | {s['dist_calcs_per_query']:>10.0f}")

    st = index.stats()
    say(f"\nIndex: {st['n']:,} nodes, layers {st['layer_sizes']}, expected top-layer fraction 1/M = {1 / M:.4f}"
        f" (measured {st['layer_sizes'][1] / st['n']:.4f}), mean degree on layer 0 {st['avg_degree_layer0']:.1f}"
        f" (max {st['max_degree_layer0']}, cap {st['M_max0']})")
    say(f"  memory: vectors {st['vector_memory_mb']:.1f} MB, graph {st['graph_memory_mb']:.1f} MB; build {t_build:.1f} s total")
    last = rows[-1]
    say(f"  at N = {n:,}: HNSW median {last['hnsw']['median_ms']:.3f} ms vs flat {last['flat']['median_ms']:.3f} ms,"
        f" {last['dist_calcs_per_query']:.0f} vs {n:,} vector comparisons per query, recall@{k} {last['recall']:.3f}")
    return {"n": n, "dim": dim, "kind": kind, "M": M, "ef_construction": ef_construction, "ef_search": ef_search,
            "k": k, "n_queries": n_queries, "rows": rows, "ef_sweep": sweep, "stats": st}


# ---------------------------------------------------------------------------
# 4. Self-test
# ---------------------------------------------------------------------------

def load_sibling(path: Path, name: str):
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(name, None)
        print(f"  (could not import {path.name}: {type(exc).__name__}: {exc})")
        return None


class Checks:
    def __init__(self, quiet: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.quiet = quiet

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})
        if not self.quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if not r["ok"])

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r["ok"])


def self_test(seed: int = 0, quiet: bool = False) -> Checks:
    c = Checks(quiet)
    say = (lambda *a, **k: None) if quiet else print
    rng = np.random.default_rng(seed)

    say("\n[1] cosine similarity")
    u = rng.standard_normal(16)
    c.add("sim(u, u) = 1, sim(u, -u) = -1", abs(cosine_similarity(u, u) - 1) < 1e-9 and abs(cosine_similarity(u, -u) + 1) < 1e-9)
    e1, e2 = np.eye(16)[0], np.eye(16)[1]
    c.add("orthogonal vectors give 0, zero vector gives 0 (no NaN)", cosine_similarity(e1, e2) == 0.0 and cosine_similarity(u, np.zeros(16)) == 0.0)
    c.add("scale invariance: sim(3u, 0.1v) = sim(u, v)", abs(cosine_similarity(3 * u, 0.1 * e1) - cosine_similarity(u, e1)) < 1e-9)
    Qm, Xm = rng.standard_normal((3, 16)), rng.standard_normal((5, 16))
    S = cosine_similarity_matrix(Qm, Xm)
    loop = np.array([[cosine_similarity(a, b) for b in Xm] for a in Qm])
    c.add("vectorised matrix == scalar function on every pair", np.abs(S - loop).max() < 1e-5)
    Xu = normalize(Xm)
    bf = brute_force_search(Xu, Qm[0], 3)
    c.add("brute force == argsort of the full similarity row", [i for i, _ in bf] == list(np.argsort(-loop[0])[:3]))

    say("\n[2] layer assignment")
    random.seed(seed)
    idx = SimpleHNSW(dim=8, M=16)
    levels = [idx._random_level() for _ in range(40000)]
    frac1 = sum(l >= 1 for l in levels) / len(levels)
    frac2 = sum(l >= 2 for l in levels) / len(levels)
    c.add("m_L = 1 / ln M", abs(idx.m_L - 1 / math.log(16)) < 1e-12)
    c.add("P(level >= 1) = 1/M and P(level >= 2) = 1/M^2 (40,000 draws)",
          abs(frac1 - 1 / 16) < 0.006 and abs(frac2 - 1 / 256) < 0.002, f"{frac1:.4f} vs {1 / 16:.4f}, {frac2:.4f} vs {1 / 256:.4f}")
    c.add("levels are non-negative integers with mean 1/(M-1)",
          min(levels) == 0 and abs(statistics.fmean(levels) - 1 / 15) < 0.01, f"mean {statistics.fmean(levels):.4f}")
    a = SimpleHNSW(dim=4, M=8, seed=3)
    b = SimpleHNSW(dim=4, M=8, seed=3)
    c.add("seed= gives a reproducible level sequence", [a._random_level() for _ in range(50)] == [b._random_level() for _ in range(50)])

    say("\n[3] graph invariants on a built index")
    n, d = 2000, 64
    X = make_embeddings(n, d, seed=seed)
    random.seed(seed)
    idx = SimpleHNSW(dim=d, M=8, ef_construction=48)
    for i in range(n):
        idx.insert(i, X[i])
    sizes = idx.layer_sizes()
    c.add("layer 0 holds every node, sizes shrink geometrically upward", sizes[0] == n and all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1)), str(sizes))
    c.add("a node on layer l is present on every layer below it",
          all(all(r in idx.layers[l] for l in range(lvl + 1)) for r, lvl in enumerate(idx._levels)))
    c.add("entry point lives on the top layer", idx.entry_point in idx.layers[idx.max_level] and idx.max_level == len(sizes) - 1)
    degree_ok = all(len(nbrs) <= idx._m_max(l) for l in range(len(sizes)) for nbrs in idx.layers[l].values())
    c.add("no node exceeds M_max (M upper layers, 2M on layer 0) after pruning", degree_ok)
    sym = sum(r in idx.layers[0][nb] for r, nbrs in idx.layers[0].items() for nb in nbrs) / max(1, sum(len(v) for v in idx.layers[0].values()))
    c.add("most layer-0 edges are bidirectional (pruning may drop one direction)", sym > 0.7, f"{sym:.2f}")
    no_self = all(r not in nbrs for l in range(len(sizes)) for r, nbrs in idx.layers[l].items())
    c.add("no self-loops, no duplicate neighbours", no_self and all(len(set(nbrs)) == len(nbrs) for l in range(len(sizes)) for nbrs in idx.layers[l].values()))
    comp = idx.connected_component_size(0)
    c.add("layer 0 is one connected component (every vector reachable from the entry point)", comp == n, f"{comp}/{n}")
    c.add("stored vectors are unit length, ids map back to rows", np.allclose(np.linalg.norm(idx.vectors, axis=1), 1, atol=1e-5) and idx.ids[idx._rows[1234]] == 1234)

    say("\n[4] search correctness")
    Qs = normalize(X[rng.integers(0, n, 40)] + 0.2 * rng.standard_normal((40, d), dtype=np.float32))
    exact_ok = True
    for q in Qs[:10]:
        full = idx.search_layer(idx.entry_point, q, ef=n, layer=0)
        got = [idx.ids[r] for _, r in full[:10]]
        exact_ok &= got == [i for i, _ in brute_force_search(X, q, 10)]
    c.add("search_layer with ef = N on layer 0 returns the exact top-10 (graph is navigable)", exact_ok)
    g = idx._greedy_closest(idx.entry_point, Qs[0], 0)
    s1 = idx.search_layer(idx.entry_point, Qs[0], ef=1, layer=0)
    c.add("greedy descent == search_layer(ef = 1) ends at a local optimum with similarity >= start",
          float(idx._sims([g], Qs[0])[0]) >= float(idx._sims([idx.entry_point], Qs[0])[0]) and len(s1) == 1)
    r10 = recall_at_k(idx, X, Qs, 10, ef=64)
    c.add("recall@10 >= 0.9 at ef = 64 (2,000 x 64 clustered)", r10 >= 0.9, f"{r10:.3f}")
    r_lo, r_hi = recall_at_k(idx, X, Qs, 10, ef=10), recall_at_k(idx, X, Qs, 10, ef=400)
    c.add("recall grows with ef (ef = 10 < ef = 400) and reaches ~1.0", r_lo <= r_hi and r_hi >= 0.98, f"{r_lo:.3f} -> {r_hi:.3f}")
    self_hits = sum(idx.query(X[i], top_k=1)[0][0] == i for i in rng.integers(0, n, 50))
    c.add("querying a stored vector returns itself at rank 1 (>= 48/50)", self_hits >= 48, f"{self_hits}/50")
    top = idx.query(Qs[1], top_k=5)
    c.add("results are sorted best first with cosine in [-1, 1]", all(top[i][1] >= top[i + 1][1] for i in range(4)) and all(-1 <= s <= 1 for _, s in top))
    c.add("query is scale invariant (q and 10q give the same ids)", [i for i, _ in idx.query(Qs[2] * 10, 5)] == [i for i, _ in idx.query(Qs[2], 5)])
    c.add("empty index returns []", SimpleHNSW(dim=4, M=4).query(np.ones(4)) == [])
    small = SimpleHNSW(dim=d, M=8, ef_construction=16, seed=0)
    small.add_items(X[:7])
    c.add("add_items works and top_k > N returns all N results", len(small.query(Qs[0], top_k=20)) == 7)
    try:
        idx.insert(5, X[5]); c.add("duplicate id is rejected", False)
    except ValueError:
        c.add("duplicate id is rejected", True)
    try:
        idx.insert(99999, np.ones(d + 1)); c.add("dimension mismatch is rejected", False)
    except ValueError:
        c.add("dimension mismatch is rejected", True)
    sparse_ids = SimpleHNSW(dim=d, M=8, seed=0)
    sparse_ids.add_items(X[:5], ids=[100, 200, 300, 400, 500])
    c.add("ids need not be contiguous (external ids are returned)", sparse_ids.query(X[2], 1)[0][0] == 300)

    say("\n[5] complexity: vector comparisons per query")
    counts = {}
    for size in (500, 2000, 8000):
        random.seed(seed)
        h = SimpleHNSW(dim=32, M=8, ef_construction=32, ef_search=32)
        Xs = make_embeddings(size, 32, seed=seed + 7)
        for i in range(size):
            h.insert(i, Xs[i])
        h.n_dist_calcs = 0
        for q in Qs[:20]:
            h.query(q[:32], 10)
        counts[size] = h.n_dist_calcs / 20
    c.add("HNSW compares far fewer vectors than flat search (N = 8,000: < N / 10)", counts[8000] < 800, f"{counts[8000]:.0f} of 8,000")
    growth = counts[8000] / counts[500]
    c.add("comparisons grow sub-linearly: x16 data -> less than x3 comparisons", growth < 3.0,
          f"{counts[500]:.0f} -> {counts[2000]:.0f} -> {counts[8000]:.0f}")

    Xc = make_embeddings(4000, 64, seed=seed + 3, intrinsic_dim=None)   # full-dimensional clusters: the hard case
    Qc = normalize(Xc[rng.integers(0, 4000, 30)] + 0.15 * rng.standard_normal((30, 64), dtype=np.float32))
    rec = {}
    for heur in (False, True):
        random.seed(seed)
        hh = SimpleHNSW(dim=64, M=16, ef_construction=64, ef_search=64, heuristic=heur)
        hh.add_items(Xc)
        rec[heur] = recall_at_k(hh, Xc, Qc, 10)
    c.add("Algorithm 4 (heuristic neighbours) beats plain top-M on clustered data (4,000 x 64, 200 clusters)",
          rec[True] > rec[False] + 0.1 and rec[True] >= 0.9, f"heuristic {rec[True]:.3f} vs simple {rec[False]:.3f}")

    say("\n[6] integration with the other stations")
    p2 = load_sibling(PART2, "part2_bm25_rrf")
    if p2 is not None and hasattr(p2, "DenseRetriever"):
        vecs = make_embeddings(400, 32, seed=seed)
        dr = p2.DenseRetriever(vecs, seed=seed)
        if dr.index is not None:
            rec = np.mean([len({dd for dd, _ in dr.search(q, 10)} & {dd for dd, _ in dr.brute_force(q, 10)}) / 10
                           for q in rng.standard_normal((30, 32))])
            c.add(f"Part 2 DenseRetriever builds this index ({dr.source}) and gets recall@10 >= 0.9", rec >= 0.9, f"{rec:.2f}")
        else:
            c.add("Part 2 DenseRetriever did not pick up this file", False, dr.source)
        if hasattr(p2, "rrf_fuse"):
            fused = p2.rrf_fuse({"dense": [1, 2, 3], "sparse": [3, 4, 5]}, k=60)
            top_id = fused[0][0]
            c.add("Part 2 rrf_fuse: a document ranked by both streams wins (k = 60)", top_id == 3)
    else:
        c.add("Part 2 not importable - integration check skipped", True)
    p4 = load_sibling(PART4, "part4_rag_audit")
    if p4 is not None:
        names = [n_ for n_ in dir(p4) if re.search(r"inject|sanit|triad|faithful", n_, re.I)]
        c.add("Part 4 exposes injection / RAG-triad components", len(names) > 0, ", ".join(names[:4]))
    else:
        c.add("Part 4 not importable - presence check skipped", True)

    say(f"\n--- {c.passed} passed, {c.failed} failed ---")
    return c


# ---------------------------------------------------------------------------
# 5. Demo, gate, logging, CLI
# ---------------------------------------------------------------------------

def demo(seed: int = 0) -> Dict[str, Any]:
    print("=" * 78)
    print("USSE NOOSPHERE - SUBSYSTEM 05: HNSW VECTOR PROXIMITY INDEX (Systems Architect)")
    print("=" * 78)
    n, d = 2000, 64
    X = make_embeddings(n, d, seed=seed)
    random.seed(seed)
    idx = SimpleHNSW(dim=d, M=16, ef_construction=64)
    _, build_ms = timed(idx.add_items, X)
    st = idx.stats()
    print(f"1. Built {n:,} x {d} index in {build_ms / 1000:.2f} s: layers {st['layer_sizes']} "
          f"(1/M = {1 / 16:.4f} of nodes expected on layer 1, measured {st['layer_sizes'][1] / n:.4f}),"
          f" mean degree on layer 0 {st['avg_degree_layer0']:.1f}")
    q = normalize(X[17] + 0.2 * np.random.default_rng(seed).standard_normal(d, dtype=np.float32))
    idx.n_dist_calcs = 0
    hits, h_ms = timed(idx.query, q, 5)
    nd = idx.n_dist_calcs
    flat, f_ms = timed(brute_force_search, X, q, 5)
    print(f"2. Query (perturbed copy of node 17): HNSW {h_ms:.3f} ms with {nd} vector comparisons, flat {f_ms:.3f} ms with {n:,}")
    print("   HNSW top 5          | flat top 5")
    for (a, sa), (b, sb) in zip(hits, flat):
        print(f"   node {a:04d}  {sa:.4f}   | node {b:04d}  {sb:.4f}")
    print()
    bench = benchmark(n=8000, dim=128, n_queries=50, checkpoints=(1000, 2000, 4000, 8000), seed=seed)
    print("\nNext: --selftest (35 checks), --bench (N = 25,000 x 768), --check-signoffs (gate).")
    print("=" * 78)
    return {"demo_stats": st, "query_ms": h_ms, "flat_ms": f_ms, "dist_calcs": nd, "small_bench": bench}


def check_signoffs(readme: Path = HERE / "README.md") -> Tuple[bool, List[Tuple[str, bool]]]:
    rows = []
    pattern = re.compile(r"^- \[( |x|X)\] \*\*([^*(`]+)")
    for line in readme.read_text(encoding="utf-8").splitlines():
        mm = pattern.match(line.strip())
        if mm:
            rows.append((mm.group(2).strip(), mm.group(1).lower() == "x"))
    return bool(rows) and all(ok for _, ok in rows), rows


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps05-{mode}.json"
    payload = {"mission": "ps05", "station": "01 Systems Architect", "mode": mode,
               "timestamp": datetime.now().isoformat(timespec="seconds"), "numpy": np.__version__, **payload}
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mission 05 Part 1 - HNSW vector index")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true", help="N = 25,000 x 768 benchmark (about 2 minutes)")
    ap.add_argument("--check-signoffs", action="store_true")
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef", type=int, default=64, help="ef_construction and default ef_search")
    ap.add_argument("--data", choices=["clustered", "random"], default="clustered",
                    help="clustered = topics with intrinsic dim 32 (default), random = isotropic 768-d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    if args.check_signoffs:
        ok, rows = check_signoffs()
        for name, ticked in rows:
            print(f"  [{'x' if ticked else ' '}] {name}")
        print("GATE PASSED - all four stations signed off." if ok else "GATE BLOCKED - missing sign-offs.")
        if not args.no_log:
            print(f"log: {write_log('signoffs', {'passed': ok, 'rows': rows})}")
        return 0 if ok else 1
    if args.selftest:
        c = self_test(seed=args.seed)
        if not args.no_log:
            print(f"log: {write_log('selftest', {'passed': c.passed, 'failed': c.failed, 'checks': c.rows})}")
        return 0 if c.failed == 0 else 1
    if args.bench:
        out = benchmark(n=args.n, dim=args.dim, M=args.M, ef_construction=args.ef, ef_search=args.ef,
                        seed=args.seed, kind=args.data)
        if not args.no_log:
            print(f"log: {write_log('bench', out)}")
        return 0
    out = demo(seed=args.seed)
    if not args.no_log:
        print(f"log: {write_log('demo', out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
