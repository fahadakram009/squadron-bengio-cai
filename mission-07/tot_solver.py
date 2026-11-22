"""
Mission 07 — Part 1: Systems Architect (Lead Engineer)
Tree-of-Thoughts (ToT) Breadth-First Beam Search & Depth-First Backtracking Solver

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
Station Officer: Fahad Akram (Systems Architect), Squadron Bengio

Pure Python 3.11+, standard library only.

Contract (kept from the instructor's starter so the harness and the other stations match):
    generator_fn(node: ThoughtNode) -> List[str]    candidate next thoughts; [] = dead end / terminal
    evaluator_fn(thought: str) -> float             step score in [0, 1], PRM-style
    one expansion = one child thought generated AND evaluated; expansion_count never exceeds max_expansions
    cumulative_score = reduction over the chain (default "min", the weakest link; "prod" = joint validity)

What this file adds on top of the starter
    1. ThoughtNode with is_pruned / is_terminal flags, trajectory and scored-trajectory reconstruction.
    2. TreeOfThoughtsSolver with ToT-BFS (beam of width B per depth) and ToT-DFS (best-first descent with
       chronological backtracking), both under a hard expansion budget, with an event trace (expand, prune,
       keep, drop, backtrack, terminal, budget) and a stats() summary.
    3. A real search problem: the Game of 24 (the ToT paper's benchmark) with an exact arithmetic
       generator, a brute-force solvability oracle, and a NOISY evaluator that mimics an imperfect PRM,
       so that beam width and budget actually matter.
    4. benchmark(): solve rate and expansions used for BFS (B = 1, 2, 3, 5) and DFS over budgets and
       evaluator noise levels; cost_model(): tokens and peak KV-cache as a function of B (doc Section A).
    5. self_test(): 38 checks, including the instructor's mock harness, budget enforcement for every
       budget, beam-width and depth invariants, backtracking, and integration with the Payload
       Specialist's PRM, the Sentinel's horizon pruner and the Protocol Officer's MCTS.

Run (from the repo root, the shared .venv; nothing to install):
    python mission-07/tot_solver.py                   # instructor harness + short benchmark
    python mission-07/tot_solver.py --selftest        # 38 checks, log in logs/
    python mission-07/tot_solver.py --bench           # full benchmark tables for doc Section A
    python mission-07/tot_solver.py --check-signoffs  # gate: Parts 1-4 ticked in README.md
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"

GeneratorFn = Callable[["ThoughtNode"], List[str]]
EvaluatorFn = Callable[[str], float]
TerminalFn = Callable[["ThoughtNode"], bool]


# =============================================================================
# 1. Thought tree
# =============================================================================

def reduce_scores(parent_cumulative: float, step_score: float, reduction: str) -> float:
    if reduction == "min":
        return min(parent_cumulative, step_score)
    if reduction == "prod":
        return parent_cumulative * step_score
    raise ValueError(f"unknown reduction {reduction!r}; use 'min' or 'prod'")


@dataclass
class ThoughtNode:
    """A discrete reasoning step z_t in a Tree of Thoughts."""
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional["ThoughtNode"] = field(default=None, repr=False)
    children: List["ThoughtNode"] = field(default_factory=list, repr=False)
    is_pruned: bool = False
    is_terminal: bool = False

    def add_child(self, child_state: str, step_score: float, reduction: str = "min") -> "ThoughtNode":
        """Create and attach a child; its cumulative score is the reduction of the parent's and its own."""
        child = ThoughtNode(
            state=child_state,
            depth=self.depth + 1,
            step_score=float(step_score),
            cumulative_score=reduce_scores(self.cumulative_score, float(step_score), reduction),
            parent=self,
        )
        self.children.append(child)
        return child

    def get_trajectory(self) -> List[str]:
        """The chain of thoughts [z_0, ..., z_t] from the root to this node."""
        path: List[str] = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def get_scored_trajectory(self) -> List[Tuple[str, float]]:
        """(state, step_score) pairs from the root to this node; the root carries score 1.0."""
        out: List[Tuple[str, float]] = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            out.append((node.state, node.step_score))
            node = node.parent
        out.reverse()
        return out

    def walk(self) -> Iterable["ThoughtNode"]:
        yield self
        for c in self.children:
            yield from c.walk()


# =============================================================================
# 2. Solver
# =============================================================================

class TreeOfThoughtsSolver:
    """System 2 search over thoughts under a strict inference compute budget.

    ToT-BFS keeps the best `beam_width` nodes per depth; ToT-DFS follows the best child and backtracks on
    dead ends. Both count one expansion per generated-and-evaluated child and never exceed
    `max_expansions`. A child whose step score is below `prune_threshold` stays attached to the tree
    (for the audit) but is marked pruned and never expanded.
    """

    def __init__(self, max_expansions: int = 16, beam_width: int = 2, prune_threshold: float = 0.45,
                 reduction: str = "min", terminal_fn: Optional[TerminalFn] = None,
                 stop_on_terminal: bool = True):
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if max_expansions < 0:
            raise ValueError("max_expansions must be >= 0")
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError("prune_threshold must be in [0, 1]")
        reduce_scores(1.0, 1.0, reduction)  # validates the name
        self.max_expansions = max_expansions
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.reduction = reduction
        self.terminal_fn = terminal_fn
        self.stop_on_terminal = stop_on_terminal
        self.expansion_count = 0
        self.pruned_count = 0
        self.backtrack_count = 0
        self.depth_reached = 0
        self.events: List[Dict[str, Any]] = []
        self.last_mode: Optional[str] = None

    # -- bookkeeping ------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.expansion_count = 0
        self.pruned_count = 0
        self.backtrack_count = 0
        self.depth_reached = 0
        self.events = []

    def _log(self, event: str, **kw: Any) -> None:
        self.events.append({"event": event, "expansions": self.expansion_count, **kw})

    def budget_left(self) -> int:
        return self.max_expansions - self.expansion_count

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.last_mode,
            "expansions": self.expansion_count,
            "max_expansions": self.max_expansions,
            "beam_width": self.beam_width,
            "prune_threshold": self.prune_threshold,
            "reduction": self.reduction,
            "pruned": self.pruned_count,
            "backtracks": self.backtrack_count,
            "depth_reached": self.depth_reached,
            "events": len(self.events),
        }

    # -- shared pieces ----------------------------------------------------------------------------------
    def _is_terminal(self, node: ThoughtNode) -> bool:
        return bool(self.terminal_fn(node)) if self.terminal_fn is not None else False

    @staticmethod
    def _better(a: ThoughtNode, b: ThoughtNode) -> bool:
        """Deeper wins; ties by cumulative score, then by step score. Deterministic."""
        return (a.depth, a.cumulative_score, a.step_score) > (b.depth, b.cumulative_score, b.step_score)

    def _expand(self, node: ThoughtNode, generator_fn: GeneratorFn, evaluator_fn: EvaluatorFn
                ) -> Tuple[List[ThoughtNode], int]:
        """Generate and score the children of `node` within the budget.

        Returns (surviving children, number of candidate thoughts the generator proposed). Every
        evaluated child costs one expansion; candidates beyond the budget are never evaluated.
        """
        candidates = list(generator_fn(node))
        survivors: List[ThoughtNode] = []
        for thought in candidates:
            if self.expansion_count >= self.max_expansions:
                self._log("budget", depth=node.depth + 1, unevaluated=len(candidates) - len(node.children))
                break
            self.expansion_count += 1
            score = float(evaluator_fn(thought))
            child = node.add_child(thought, score, self.reduction)
            child.is_terminal = self._is_terminal(child)
            self.depth_reached = max(self.depth_reached, child.depth)
            if score < self.prune_threshold:
                child.is_pruned = True
                self.pruned_count += 1
                self._log("prune", depth=child.depth, score=round(score, 4), state=thought)
            else:
                survivors.append(child)
                self._log("expand", depth=child.depth, score=round(score, 4), state=thought)
        return survivors, len(candidates)

    # -- ToT-BFS ----------------------------------------------------------------------------------------
    def bfs_beam_search(self, root: ThoughtNode, generator_fn: GeneratorFn, evaluator_fn: EvaluatorFn,
                        max_depth: int = 3) -> ThoughtNode:
        """Breadth-first beam search: at each depth expand every beam node, keep the best B survivors.

        Stops when the depth limit is reached, the frontier dies, the budget is spent, or (with
        stop_on_terminal) the best kept node is terminal; lower-ranked terminals in the beam do not stop
        the search, they simply are not expanded. Returns the best node found: deepest first,
        then highest cumulative score; the root if nothing survived.
        """
        self.reset()
        self.last_mode = "bfs"
        best = root
        beam: List[ThoughtNode] = [root]
        for depth in range(1, max_depth + 1):
            candidates: List[ThoughtNode] = []
            for node in beam:
                if node.is_terminal:
                    continue
                if self.expansion_count >= self.max_expansions:
                    self._log("budget", depth=depth, unexpanded_beam_nodes=len(beam) - beam.index(node))
                    break
                survivors, _ = self._expand(node, generator_fn, evaluator_fn)
                candidates.extend(survivors)
            if not candidates:
                self._log("frontier_empty", depth=depth)
                break
            candidates.sort(key=lambda c: (c.cumulative_score, c.step_score), reverse=True)
            beam, dropped = candidates[: self.beam_width], candidates[self.beam_width:]
            for c in beam:
                self._log("keep", depth=depth, score=round(c.cumulative_score, 4), state=c.state)
            for c in dropped:
                self._log("drop", depth=depth, score=round(c.cumulative_score, 4), state=c.state)
            if self._better(beam[0], best):
                best = beam[0]
            if beam[0].is_terminal and self.stop_on_terminal:
                self._log("terminal", depth=depth, state=beam[0].state)
                break
            if self.expansion_count >= self.max_expansions:
                break
        return best

    # -- ToT-DFS ----------------------------------------------------------------------------------------
    def dfs_backtrack_search(self, root: ThoughtNode, generator_fn: GeneratorFn, evaluator_fn: EvaluatorFn,
                             max_depth: int = 3) -> ThoughtNode:
        """Depth-first search that follows the highest-scoring surviving child and backtracks
        chronologically when a branch dead-ends (no candidates, all pruned, or budget spent below it).

        Returns the first complete trajectory (a terminal node, or a node at max_depth when no
        terminal_fn is given) if stop_on_terminal, else the best node seen; the budget is strict.
        """
        self.reset()
        self.last_mode = "dfs"
        best = root

        def visit(node: ThoughtNode) -> bool:
            nonlocal best
            if self._better(node, best):
                best = node
            complete = node.is_terminal if self.terminal_fn is not None else node.depth >= max_depth
            if complete:
                self._log("terminal", depth=node.depth, state=node.state)
                return self.stop_on_terminal
            if node.depth >= max_depth:
                return False
            if self.expansion_count >= self.max_expansions:
                return False
            survivors, n_candidates = self._expand(node, generator_fn, evaluator_fn)
            survivors.sort(key=lambda c: (c.step_score, c.cumulative_score), reverse=True)
            for child in survivors:
                if visit(child):
                    return True
                self.backtrack_count += 1
                self._log("backtrack", depth=child.depth, state=child.state)
                if self.expansion_count >= self.max_expansions:
                    return False
            if n_candidates == 0:
                self._log("dead_end", depth=node.depth, state=node.state)
            return False

        visit(root)
        return best


# =============================================================================
# 3. The Game of 24 as a real search problem
# =============================================================================

NUMS_RE = re.compile(r"\[([-\d/,\s]+)\]")  # numeric lists only; tags like [GOAL] do not match


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def parse_numbers(state: str) -> List[Fraction]:
    """Numbers still available in a state string: the last numeric [...] group, e.g. 'left: [4, 6]'."""
    groups = NUMS_RE.findall(state)
    if not groups:
        return []
    return [Fraction(tok.strip()) for tok in groups[-1].split(",") if tok.strip()]


def game24_root(numbers: Sequence[int]) -> ThoughtNode:
    return ThoughtNode(f"Game of 24: make 24 from [{', '.join(str(n) for n in numbers)}]")


def game24_generator(node: ThoughtNode) -> List[str]:
    """All single arithmetic steps on two of the remaining numbers, deduplicated by the resulting set."""
    nums = parse_numbers(node.state)
    if len(nums) <= 1:
        return []
    seen = set()
    thoughts: List[str] = []
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [n for k, n in enumerate(nums) if k not in (i, j)]
        ops: List[Tuple[Fraction, Fraction, str, Fraction]] = [(a, b, "+", a + b), (a, b, "*", a * b)]
        ops.append((a, b, "-", a - b) if a >= b else (b, a, "-", b - a))
        if b != 0:
            ops.append((a, b, "/", a / b))
        if a != 0 and a != b:
            ops.append((b, a, "/", b / a))
        for x, y, op, r in ops:
            left = sorted(rest + [r])
            key = tuple(left)
            if key in seen:
                continue
            seen.add(key)
            tag = ""
            if len(left) == 1:
                tag = " [GOAL]" if left[0] == 24 else " [FAIL]"
            thoughts.append(f"{_fmt(x)} {op} {_fmt(y)} = {_fmt(r)} (left: [{', '.join(_fmt(v) for v in left)}]){tag}")
    return thoughts


def game24_terminal(node: ThoughtNode) -> bool:
    return len(parse_numbers(node.state)) == 1


@lru_cache(maxsize=None)
def solvable24(nums: Tuple[Fraction, ...]) -> bool:
    """Brute-force oracle: can the numbers be combined with + - * / to make exactly 24."""
    if len(nums) == 1:
        return nums[0] == 24
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [n for k, n in enumerate(nums) if k not in (i, j)]
        results = {a + b, a * b, a - b, b - a}
        if b != 0:
            results.add(a / b)
        if a != 0:
            results.add(b / a)
        for r in results:
            if solvable24(tuple(sorted(rest + [r]))):
                return True
    return False


def make_evaluator(sigma: float = 0.0, seed: int = 0) -> EvaluatorFn:
    """A PRM stand-in for the Game of 24.

    Exact when sigma = 0: a step that keeps 24 reachable scores 0.9, a step that does not scores 0.2,
    a terminal step scores 1.0 or 0.0. With sigma > 0 deterministic Gaussian noise (seeded by the thought
    text) is added, so the evaluator sometimes prunes good steps and keeps bad ones, like a real PRM.
    """
    def evaluate(thought: str) -> float:
        nums = parse_numbers(thought)
        if not nums:
            return 0.0
        if len(nums) == 1:
            base = 1.0 if nums[0] == 24 else 0.0
        else:
            base = 0.9 if solvable24(tuple(sorted(nums))) else 0.2
        if sigma > 0:
            h = int(hashlib.sha256(f"{seed}|{thought}".encode()).hexdigest(), 16) % (2 ** 32)
            base += random.Random(h).gauss(0.0, sigma)
        return min(1.0, max(0.0, base))
    return evaluate


def make_puzzles(n: int = 40, seed: int = 7, lo: int = 1, hi: int = 13) -> List[Tuple[int, ...]]:
    """Random solvable 4-number puzzles, deterministic for a seed."""
    rng = random.Random(seed)
    out: List[Tuple[int, ...]] = []
    seen = set()
    while len(out) < n:
        nums = tuple(sorted(rng.randint(lo, hi) for _ in range(4)))
        if nums in seen or not solvable24(tuple(Fraction(v) for v in nums)):
            continue
        seen.add(nums)
        out.append(nums)
    return out


def solve24(numbers: Sequence[int], mode: str, beam_width: int, max_expansions: int,
            evaluator: EvaluatorFn, prune_threshold: float = 0.45) -> Tuple[bool, TreeOfThoughtsSolver, ThoughtNode]:
    solver = TreeOfThoughtsSolver(max_expansions=max_expansions, beam_width=beam_width,
                                  prune_threshold=prune_threshold, terminal_fn=game24_terminal)
    root = game24_root(numbers)
    search = solver.bfs_beam_search if mode == "bfs" else solver.dfs_backtrack_search
    best = search(root, game24_generator, evaluator, max_depth=3)
    solved = best.is_terminal and "[GOAL]" in best.state
    return solved, solver, best


# =============================================================================
# 4. Benchmark and cost model (doc Section A)
# =============================================================================

CONFIGS: List[Tuple[str, int]] = [("bfs", 1), ("bfs", 2), ("bfs", 3), ("bfs", 5), ("dfs", 1)]


def benchmark(n_puzzles: int = 40, budgets: Sequence[int] = (48, 64, 96, 192),
              sigmas: Sequence[float] = (0.0, 0.25, 0.5), seed: int = 7, quiet: bool = False) -> Dict[str, Any]:
    puzzles = make_puzzles(n_puzzles, seed=seed)
    rows: List[Dict[str, Any]] = []
    for sigma in sigmas:
        evaluator = make_evaluator(sigma=sigma, seed=seed)
        if not quiet:
            print(f"\nGame of 24, {n_puzzles} solvable puzzles, depth 3, prune_threshold 0.45, evaluator noise sigma = {sigma}")
            print(f"  {'config':<10}" + "".join(f"{'budget ' + str(b):>16}" for b in budgets) + "    (solve rate | mean expansions)")
        for mode, bw in CONFIGS:
            cells = []
            for budget in budgets:
                solved = 0
                used = 0
                for p in puzzles:
                    ok, solver, _ = solve24(p, mode, bw, budget, evaluator)
                    solved += ok
                    used += solver.expansion_count
                rate, mean_used = solved / n_puzzles, used / n_puzzles
                rows.append({"sigma": sigma, "mode": mode, "beam_width": bw, "budget": budget,
                             "solve_rate": rate, "mean_expansions": mean_used})
                cells.append(f"{rate:>6.2f} | {mean_used:>5.1f}")
            if not quiet:
                label = f"{mode.upper()} B={bw}" if mode == "bfs" else "DFS"
                print(f"  {label:<10}" + "".join(f"{c:>16}" for c in cells))
    return {"n_puzzles": n_puzzles, "budgets": list(budgets), "sigmas": list(sigmas), "rows": rows}


def cost_model(beam_widths: Sequence[int] = (1, 2, 3, 5), depth: int = 3, branching: int = 6,
               prompt_tokens: int = 200, tokens_per_thought: int = 60) -> List[Dict[str, Any]]:
    """Token and KV-cache cost of ToT-BFS as a function of beam width B (an LLM-serving view).

    Generated tokens ~ expansions x tokens_per_thought, and expansions per depth level ~ B x branching.
    The KV cache holds one context per beam node that is being expanded: B contexts of
    (prompt + depth x tokens_per_thought) tokens at the deepest level. DFS keeps one context (plus a
    stack of siblings' scores, which are numbers, not tokens).
    """
    rows = []
    for b in beam_widths:
        expansions = branching + (depth - 1) * b * branching  # root is expanded once, then B nodes per level
        generated = expansions * tokens_per_thought
        kv_peak = b * (prompt_tokens + depth * tokens_per_thought)
        rows.append({"beam_width": b, "expansions": expansions, "generated_tokens": generated,
                     "kv_peak_tokens": kv_peak, "kv_relative": b})
    return rows


# =============================================================================
# 5. Instructor harness (kept verbatim in spirit) and teammate loading
# =============================================================================

MOCK_BRANCHES = {
    "Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]": [
        "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]",
        "Step 1B: 10 + 13 = 23 (Remaining: [4, 9, 23]) [DEAD-END]",
    ],
    "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]": [
        "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]",
        "Step 2B: 4 + 4 = 8 (Remaining: [8, 10]) [DEAD-END]",
    ],
    "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]": [
        "Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]",
    ],
}


def mock_generator(node: ThoughtNode) -> List[str]:
    return MOCK_BRANCHES.get(node.state, [])


def mock_evaluator(thought: str) -> float:
    if "[GOAL]" in thought:
        return 0.98
    if "[OPTIMAL]" in thought:
        return 0.92
    if "[VALID]" in thought:
        return 0.85
    return 0.25


def instructor_harness(quiet: bool = False) -> Dict[str, Any]:
    solver = TreeOfThoughtsSolver(max_expansions=12, beam_width=2, prune_threshold=0.45)
    root_bfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_bfs = solver.bfs_beam_search(root_bfs, mock_generator, mock_evaluator, max_depth=3)
    bfs_used = solver.expansion_count
    root_dfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_dfs = solver.dfs_backtrack_search(root_dfs, mock_generator, mock_evaluator, max_depth=3)
    dfs_used = solver.expansion_count
    if not quiet:
        print("=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===")
        print(f"[ToT-BFS] Expansions Used: {bfs_used}/{solver.max_expansions}")
        print(f"[ToT-BFS] Best Terminal Score: {best_bfs.cumulative_score:.2f}")
        print(f"[ToT-BFS] Winning Trajectory: {' -> '.join(best_bfs.get_trajectory())}")
        print(f"[ToT-DFS] Expansions Used: {dfs_used}/{solver.max_expansions}")
        print(f"[ToT-DFS] Backtracks: {solver.backtrack_count}")
        print(f"[ToT-DFS] Winning Trajectory: {' -> '.join(best_dfs.get_trajectory())}")
        ok = "[GOAL]" in best_bfs.state and "[GOAL]" in best_dfs.state
        print("[OK] Part 1 (Systems Architect) verification passed." if ok else "[FAIL] Part 1 verification failed.")
    return {"bfs": {"expansions": bfs_used, "score": best_bfs.cumulative_score, "trajectory": best_bfs.get_trajectory()},
            "dfs": {"expansions": dfs_used, "score": best_dfs.cumulative_score, "trajectory": best_dfs.get_trajectory(),
                    "backtracks": solver.backtrack_count}}


def load_sibling(filename: str, module_name: str) -> Optional[Any]:
    path = HERE / filename
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        sys.modules.pop(module_name, None)
        print(f"  [notice] could not load {filename}: {type(exc).__name__}: {exc}")
        return None


# =============================================================================
# 6. Self-test
# =============================================================================

class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def passed(self) -> int:
        return sum(r["passed"] for r in self.rows)

    @property
    def failed(self) -> int:
        return len(self.rows) - self.passed


def toy_graph() -> Tuple[GeneratorFn, EvaluatorFn]:
    """A tempting dead end: A looks best, A1 looks even better, then everything under A1 collapses.
    B is the genuine chain. C is weak from the start."""
    graph = {"start": ["A", "B", "C"], "A": ["A1", "A2"], "A1": ["A1x", "A1y"], "A2": [],
             "B": ["B1", "B2"], "B1": ["B1-final"], "B2": [], "C": ["C1"], "C1": ["C1-final"]}
    scores = {"A": 0.9, "B": 0.8, "C": 0.2, "A1": 0.95, "A2": 0.4, "A1x": 0.1, "A1y": 0.05,
              "B1": 0.85, "B2": 0.5, "B1-final": 0.9, "C1": 0.9, "C1-final": 0.9}
    return (lambda node: graph.get(node.state, [])), (lambda s: scores.get(s, 0.0))


def self_test() -> Checks:
    c = Checks()
    print("=== Part 1 self-test: Tree-of-Thoughts solver ===")

    # --- node ---------------------------------------------------------------------------------------
    r = ThoughtNode("root")
    a = r.add_child("a", 0.8)
    b = a.add_child("b", 0.9)
    c.add("get_trajectory reconstructs root -> ... -> node", b.get_trajectory() == ["root", "a", "b"] and b.depth == 2)
    c.add("min reduction: cumulative is the weakest link", a.cumulative_score == 0.8 and b.cumulative_score == 0.8)
    p = ThoughtNode("root").add_child("a", 0.8, "prod").add_child("b", 0.5, "prod")
    c.add("prod reduction: cumulative is the joint validity", abs(p.cumulative_score - 0.4) < 1e-12)
    c.add("scored trajectory carries step scores", b.get_scored_trajectory() == [("root", 1.0), ("a", 0.8), ("b", 0.9)])
    c.add("starter defaults kept (16 expansions, beam 2, threshold 0.45, min)",
          (lambda s: (s.max_expansions, s.beam_width, s.prune_threshold, s.reduction) == (16, 2, 0.45, "min"))(TreeOfThoughtsSolver()))

    def raises(fn: Callable[[], Any], exc: type) -> bool:
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False
    c.add("invalid beam width / budget / threshold / reduction are refused",
          raises(lambda: TreeOfThoughtsSolver(beam_width=0), ValueError) and raises(lambda: TreeOfThoughtsSolver(max_expansions=-1), ValueError)
          and raises(lambda: TreeOfThoughtsSolver(prune_threshold=1.5), ValueError) and raises(lambda: TreeOfThoughtsSolver(reduction="mean"), ValueError))

    # --- instructor harness ---------------------------------------------------------------------------
    h = instructor_harness(quiet=True)
    c.add("instructor mock: ToT-BFS reaches [GOAL] within 12 expansions",
          "[GOAL]" in h["bfs"]["trajectory"][-1] and h["bfs"]["expansions"] <= 12, f"{h['bfs']['expansions']} expansions, score {h['bfs']['score']:.2f}")
    c.add("instructor mock: ToT-DFS reaches [GOAL] within 12 expansions",
          "[GOAL]" in h["dfs"]["trajectory"][-1] and h["dfs"]["expansions"] <= 12, f"{h['dfs']['expansions']} expansions")
    c.add("instructor mock: min-link score of the winning chain is 0.85", abs(h["bfs"]["score"] - 0.85) < 1e-12)

    # --- budget, beam, depth, pruning invariants --------------------------------------------------------
    gen, ev = toy_graph()
    ok_budget = True
    zero_root = True
    for budget in range(0, 12):
        for mode in ("bfs", "dfs"):
            s = TreeOfThoughtsSolver(max_expansions=budget, beam_width=3, prune_threshold=0.3)
            root = ThoughtNode("start")
            best = (s.bfs_beam_search if mode == "bfs" else s.dfs_backtrack_search)(root, gen, ev, max_depth=3)
            ok_budget &= s.expansion_count <= budget and sum(1 for n in root.walk()) - 1 == s.expansion_count
            if budget == 0:
                zero_root &= best is root and s.expansion_count == 0
    c.add("budget: expansions never exceed max_expansions for budgets 0..11, both searches; tree size equals expansions", ok_budget)
    c.add("budget 0 returns the root untouched", zero_root)

    ok_beam = True
    for bw in (1, 2, 3):
        s = TreeOfThoughtsSolver(max_expansions=50, beam_width=bw, prune_threshold=0.3)
        s.bfs_beam_search(ThoughtNode("start"), gen, ev, max_depth=3)
        kept_per_depth: Dict[int, int] = {}
        for e in s.events:
            if e["event"] == "keep":
                kept_per_depth[e["depth"]] = kept_per_depth.get(e["depth"], 0) + 1
        ok_beam &= all(v <= bw for v in kept_per_depth.values())
    c.add("beam width: at most B nodes kept per depth for B = 1, 2, 3", ok_beam)

    s = TreeOfThoughtsSolver(max_expansions=50, beam_width=3, prune_threshold=0.3)
    root = ThoughtNode("start")
    s.bfs_beam_search(root, gen, ev, max_depth=2)
    c.add("max_depth: no node deeper than the limit", max(n.depth for n in root.walk()) <= 2 and s.depth_reached <= 2)
    pruned = [n for n in root.walk() if n.is_pruned]
    c.add("pruning: children below the threshold are attached, flagged, never expanded, and counted",
          all(n.step_score < 0.3 and not n.children for n in pruned) and len(pruned) == s.pruned_count and s.pruned_count >= 1, f"{s.pruned_count} pruned")

    # --- behaviour on the tempting dead end -----------------------------------------------------------
    s1 = TreeOfThoughtsSolver(max_expansions=50, beam_width=1, prune_threshold=0.3)
    b1 = s1.bfs_beam_search(ThoughtNode("start"), gen, ev, max_depth=3)
    s2 = TreeOfThoughtsSolver(max_expansions=50, beam_width=2, prune_threshold=0.3)
    b2 = s2.bfs_beam_search(ThoughtNode("start"), gen, ev, max_depth=3)
    c.add("beam B = 1 is greedy and dies under the tempting branch (depth 2), B = 2 reaches the goal (depth 3)",
          b1.get_trajectory() == ["start", "A", "A1"] and b2.get_trajectory() == ["start", "B", "B1", "B1-final"])
    s3 = TreeOfThoughtsSolver(max_expansions=50, prune_threshold=0.3)
    b3 = s3.dfs_backtrack_search(ThoughtNode("start"), gen, ev, max_depth=3)
    order = [e["state"] for e in s3.events if e["event"] == "expand"]
    c.add("DFS follows A then A1 first, backtracks, and finds start -> B -> B1 -> B1-final",
          b3.get_trajectory() == ["start", "B", "B1", "B1-final"] and order[:1] == ["A"] and s3.backtrack_count >= 2, f"{s3.backtrack_count} backtracks")
    c.add("DFS needs no more expansions than BFS B = 2 on the toy graph", s3.expansion_count <= s2.expansion_count, f"{s3.expansion_count} vs {s2.expansion_count}")
    s4 = TreeOfThoughtsSolver(max_expansions=50, beam_width=2, prune_threshold=0.3)
    s5 = TreeOfThoughtsSolver(max_expansions=50, beam_width=2, prune_threshold=0.3)
    c.add("determinism: two identical runs give identical trajectories and events",
          s4.bfs_beam_search(ThoughtNode("start"), gen, ev, 3).get_trajectory() == s5.bfs_beam_search(ThoughtNode("start"), gen, ev, 3).get_trajectory()
          and s4.events == s5.events)
    s6 = TreeOfThoughtsSolver(max_expansions=50)
    r6 = ThoughtNode("nothing here")
    c.add("a root with no candidates returns the root with zero expansions",
          s6.bfs_beam_search(r6, gen, ev, 3) is r6 and s6.dfs_backtrack_search(r6, gen, ev, 3) is r6 and s6.expansion_count == 0)
    c.add("stats() and events are JSON serialisable", isinstance(json.dumps({"stats": s3.stats(), "events": s3.events}), str))

    # --- Game of 24 -----------------------------------------------------------------------------------
    c.add("oracle: [4, 9, 10, 13] is solvable, [1, 1, 1, 1] is not",
          solvable24(tuple(Fraction(v) for v in (4, 9, 10, 13))) and not solvable24(tuple(Fraction(v) for v in (1, 1, 1, 1))))
    root24 = game24_root((4, 9, 10, 13))
    kids = game24_generator(root24)
    lefts = [tuple(parse_numbers(k)) for k in kids]
    c.add("generator: every step leaves one number fewer and results are deduplicated",
          all(len(l) == 3 for l in lefts) and len(set(tuple(sorted(l)) for l in lefts)) == len(lefts) and 15 <= len(kids) <= 30, f"{len(kids)} candidates")
    exact = make_evaluator(0.0)
    c.add("exact evaluator: a step that keeps 24 reachable scores 0.9, one that does not scores 0.2",
          exact("13 - 9 = 4 (left: [4, 4, 10])") == 0.9 and exact("4 + 9 = 13 (left: [10, 13, 13])") == 0.2)
    ok, sv, best = solve24((4, 9, 10, 13), "bfs", 2, 96, exact)
    c.add("exact evaluator: BFS B = 2 solves [4, 9, 10, 13] and the last thought is a [GOAL] step",
          ok and best.depth == 3 and best.is_terminal, " -> ".join(best.get_trajectory()[1:]))
    ok_d, sv_d, best_d = solve24((4, 9, 10, 13), "dfs", 1, 96, exact)
    c.add("exact evaluator: DFS solves it with fewer expansions than BFS B = 2", ok_d and sv_d.expansion_count < sv.expansion_count,
          f"{sv_d.expansion_count} vs {sv.expansion_count}")
    noisy = make_evaluator(0.25, seed=7)
    c.add("noisy evaluator is deterministic for the same thought", noisy("13 - 9 = 4 (left: [4, 4, 10])") == noisy("13 - 9 = 4 (left: [4, 4, 10])"))
    puzzles = make_puzzles(40, seed=7)
    c.add("40 distinct solvable puzzles generated deterministically", len(set(puzzles)) == 40 and all(solvable24(tuple(Fraction(v) for v in p)) for p in puzzles)
          and make_puzzles(40, seed=7) == puzzles)
    bench = benchmark(n_puzzles=40, budgets=(48, 64, 96, 192), sigmas=(0.0, 0.25, 0.5), seed=7, quiet=True)

    def rate(sigma: float, mode: str, bw: int, budget: int) -> float:
        return next(r["solve_rate"] for r in bench["rows"] if r["sigma"] == sigma and r["mode"] == mode and r["beam_width"] == bw and r["budget"] == budget)
    c.add("exact evaluator, budget 192: BFS B = 2 and DFS solve every puzzle", rate(0.0, "bfs", 2, 192) == 1.0 and rate(0.0, "dfs", 1, 192) == 1.0)
    c.add("solve rate never decreases with budget (every config, every noise level)",
          all(rate(sg, m, bw, b1) <= rate(sg, m, bw, b2) for sg in (0.0, 0.25, 0.5) for m, bw in CONFIGS for b1, b2 in zip((48, 64, 96), (64, 96, 192))))
    c.add("mild noise (0.25), budget 48: the greedy beam is best when compute is scarce",
          rate(0.25, "bfs", 1, 48) >= max(rate(0.25, "bfs", bw, 48) for bw in (2, 3, 5)), f"B=1 {rate(0.25, 'bfs', 1, 48):.2f}")
    c.add("strong noise (0.5), budget 192: a wider beam beats the greedy beam (B = 3 > B = 1)",
          rate(0.5, "bfs", 3, 192) > rate(0.5, "bfs", 1, 192), f"B=3 {rate(0.5, 'bfs', 3, 192):.2f} vs B=1 {rate(0.5, 'bfs', 1, 192):.2f}")
    c.add("strong noise (0.5), budget 48: DFS backtracking is the most budget-efficient search",
          rate(0.5, "dfs", 1, 48) >= max(rate(0.5, "bfs", bw, 48) for bw in (1, 2, 3, 5)), f"DFS {rate(0.5, 'dfs', 1, 48):.2f}")
    used = {bw: next(r["mean_expansions"] for r in bench["rows"] if r["sigma"] == 0.5 and r["mode"] == "bfs" and r["beam_width"] == bw and r["budget"] == 192) for bw in (1, 2, 3, 5)}
    c.add("expansions spent grow with beam width at a fixed large budget", used[1] < used[2] < used[3] < used[5], ", ".join(f"B={k}: {v:.0f}" for k, v in used.items()))
    cm = cost_model()
    c.add("cost model: generated tokens grow with B and peak KV cache is B times the single-context cache",
          cm[-1]["generated_tokens"] > cm[0]["generated_tokens"] and cm[-1]["kv_peak_tokens"] == 5 * cm[0]["kv_peak_tokens"])

    # --- integration with the other stations ----------------------------------------------------------
    p3 = load_sibling("prm_evaluator.py", "part3_prm")
    if p3 is None:
        c.add("Part 3 PRM present", False, "file missing")
    else:
        try:
            prm = p3.ProcessRewardEvaluator()
            proof = {"Theorem: escape velocity at r = 3 r_s": ["Step 1: v = sqrt(2GM/r) [VALID]", "Step 1': v = -sqrt(2GM/r) SIGN_ERROR"],
                     "Step 1: v = sqrt(2GM/r) [VALID]": ["Step 2: substitute r = 3 r_s = 6GM/c^2 [VALID]", "Step 2': drop the factor 2 [FLAWED]"],
                     "Step 1': v = -sqrt(2GM/r) SIGN_ERROR": ["Step 2'': square both sides, sign cancels [VALID]"],
                     "Step 2: substitute r = 3 r_s = 6GM/c^2 [VALID]": ["Step 3: v = c / sqrt(3) [VALID]"]}
            s7 = TreeOfThoughtsSolver(max_expansions=16, beam_width=2, prune_threshold=0.45)
            r7 = ThoughtNode("Theorem: escape velocity at r = 3 r_s")
            best7 = s7.bfs_beam_search(r7, lambda n: proof.get(n.state, []), prm.step_score, max_depth=3)
            flawed = [n for n in r7.walk() if "SIGN_ERROR" in n.state or "[FLAWED]" in n.state]
            c.add("Part 3 PRM as evaluator: both flawed branches are pruned at their first bad step and the clean proof wins",
                  all(n.is_pruned and not n.children for n in flawed) and best7.get_trajectory()[-1].startswith("Step 3"))
            c.add("Part 3 reductions agree with the tree's min-link score on the winning chain",
                  abs(prm.aggregate_trajectory_min(best7.get_trajectory()[1:]) - best7.cumulative_score) < 1e-12)
        except Exception as exc:
            c.add("Part 3 PRM integration", False, f"{type(exc).__name__}: {exc}")

    p4 = load_sibling("horizon_pruner.py", "part4_pruner")
    if p4 is None:
        c.add("Part 4 pruner present", False, "file missing")
    else:
        try:
            pruner = p4.EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)
            steps, scores = zip(*best.get_scored_trajectory()[1:])
            rep = pruner.prune_trajectory(list(steps), list(scores))
            c.add("Part 4 pruner: the Game-of-24 winning trajectory survives the event horizon unsevered",
                  rep.pruned_at_step == -1 and not rep.reward_hacking_detected and len(rep.surviving_steps) == 3)
            hacked = list(steps[:1]) + ["Let us carefully verify this 100% rigorous step: " + steps[1]] + list(steps[2:])
            rep2 = pruner.prune_trajectory(hacked, list(scores))
            c.add("Part 4 pruner: a sycophantic rewrite of step 2 is penalized and severed", rep2.reward_hacking_detected and rep2.pruned_at_step == 2)
        except Exception as exc:
            c.add("Part 4 pruner integration", False, f"{type(exc).__name__}: {exc}")

    p2 = load_sibling("mcts_protocol.py", "part2_mcts")
    if p2 is None:
        c.add("Part 2 MCTS present", False, "file missing")
    else:
        try:
            engine = p2.MCTSEngine(c_puct=1.5)
            c.add("Part 2 MCTS engine present with puct_score, select_leaf, expand, simulate_rollout, backpropagate",
                  all(hasattr(engine, m) for m in ("puct_score", "select_leaf", "expand", "simulate_rollout", "backpropagate")))
        except Exception as exc:
            c.add("Part 2 MCTS integration", False, f"{type(exc).__name__}: {exc}")

    print(f"--- {c.passed} passed, {c.failed} failed ---")
    return c


# =============================================================================
# 7. Sign-off gate, logging, CLI
# =============================================================================

def check_signoffs(readme: Path = HERE / "README.md") -> Tuple[bool, List[Tuple[str, bool]]]:
    text = readme.read_text(encoding="utf-8")
    rows: List[Tuple[str, bool]] = []
    for part in range(1, 6):
        m = re.search(r"^- \[([ xX])\] \*\*Part " + str(part) + r"\b", text, re.MULTILINE)
        rows.append((f"Part {part}", bool(m and m.group(1).lower() == "x")))
    for label, ok in rows:
        print(f"  [{'x' if ok else ' '}] {label}")
    passed = all(ok for label, ok in rows[:4])
    print("GATE PASSED - Parts 1-4 signed off." if passed else "GATE BLOCKED - unticked parts above.")
    if passed and not rows[4][1]:
        print("  (Part 5, the squadron document and PDF, still open)")
    return passed, rows


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"{stamp}-ps07-{mode}.json"
    record = {"mission": "ps07", "station": "01 Systems Architect", "mode": mode,
              "timestamp": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], **payload}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"log: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AURA Subsystem 07, Part 1: Tree-of-Thoughts solver")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--check-signoffs", action="store_true")
    ap.add_argument("--puzzles", type=int, default=40)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    if args.check_signoffs:
        passed, rows = check_signoffs()
        if not args.no_log:
            write_log("signoffs", {"passed": passed, "rows": rows})
        return 0 if passed else 1
    if args.selftest:
        c = self_test()
        if not args.no_log:
            write_log("selftest", {"passed": c.passed, "failed": c.failed, "checks": c.rows})
        return 0 if c.failed == 0 else 1
    if args.bench:
        res = benchmark(n_puzzles=args.puzzles)
        print("\nLLM-serving cost model for ToT-BFS (depth 3, branching 6, prompt 200 tokens, 60 tokens per thought):")
        print(f"  {'B':>3} {'expansions':>11} {'generated tokens':>17} {'peak KV tokens':>15} {'KV relative':>12}")
        for r in cost_model():
            print(f"  {r['beam_width']:>3} {r['expansions']:>11} {r['generated_tokens']:>17} {r['kv_peak_tokens']:>15} {r['kv_relative']:>11}x")
        if not args.no_log:
            write_log("bench", {**res, "cost_model": cost_model()})
        return 0
    h = instructor_harness()
    print()
    res = benchmark(n_puzzles=min(args.puzzles, 20), budgets=(48, 192), sigmas=(0.5,))
    if not args.no_log:
        write_log("harness", {"harness": h, "short_benchmark": res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
