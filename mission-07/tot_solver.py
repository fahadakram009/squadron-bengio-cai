"""
Mission 07 - Part 1: Systems Architect
Tree-of-Thoughts solver: BFS beam search + DFS backtracking under a hard
node-expansion compute budget.  Pure Python 3.11+, no external dependencies.

Conventions
-----------
* generator_fn(state: str) -> List[str]   proposes candidate next thoughts.
  An empty list means "dead end / terminal".
* evaluator_fn(state: str) -> float       step score in [0, 1] (PRM-style).
* One *expansion* = one child thought generated AND evaluated (one evaluator
  call).  `expansion_count` never exceeds `max_expansions`.
* cumulative_score = parent.cumulative_score * step_score  (root = 1.0), i.e.
  the joint validity of the whole chain (product reduction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

GeneratorFn = Callable[[str], List[str]]
EvaluatorFn = Callable[[str], float]


# --------------------------------------------------------------------------- #
# Thought tree node
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class ThoughtNode:
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional["ThoughtNode"] = field(default=None, repr=False)
    children: List["ThoughtNode"] = field(default_factory=list, repr=False)

    def get_trajectory(self) -> List[str]:
        """Reasoning chain (list of states) from the root down to this node."""
        path: List[str] = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def add_child(self, state: str, step_score: float) -> "ThoughtNode":
        child = ThoughtNode(
            state=state,
            depth=self.depth + 1,
            step_score=step_score,
            cumulative_score=self.cumulative_score * step_score,
            parent=self,
        )
        self.children.append(child)
        return child


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class TreeOfThoughtsSolver:
    def __init__(
        self,
        beam_width: int = 3,
        max_expansions: int = 50,
        prune_threshold: float = 0.3,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if max_expansions < 0:
            raise ValueError("max_expansions must be >= 0")
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.prune_threshold = prune_threshold
        self.expansion_count = 0
        self.pruned_count = 0

    # ----------------------------- helpers -------------------------------- #
    def _reset(self) -> None:
        self.expansion_count = 0
        self.pruned_count = 0

    def _budget_left(self) -> bool:
        return self.expansion_count < self.max_expansions

    @staticmethod
    def _better(a: ThoughtNode, b: Optional[ThoughtNode]) -> bool:
        """Deeper wins; ties broken by cumulative score."""
        if b is None:
            return True
        return (a.depth, a.cumulative_score) > (b.depth, b.cumulative_score)

    def _spawn_children(
        self,
        node: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
    ) -> tuple[List[ThoughtNode], bool]:
        """
        Generate + score children of `node`, spending one expansion per child.
        Children below `prune_threshold` are discarded (not attached).
        Returns (surviving children, generator_returned_any_candidates).
        """
        candidates = generator_fn(node.state)
        survivors: List[ThoughtNode] = []
        for state in candidates:
            if not self._budget_left():
                break
            self.expansion_count += 1
            score = float(evaluator_fn(state))
            if score < self.prune_threshold:
                self.pruned_count += 1
                continue
            survivors.append(node.add_child(state, score))
        return survivors, bool(candidates)

    # ------------------------- Breadth-first beam ------------------------- #
    def bfs_beam_search(
        self,
        root: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
        max_depth: int,
    ) -> ThoughtNode:
        """
        Keep at most `beam_width` best nodes per depth level.  Stops when the
        depth limit is reached, the frontier dies, or the budget is spent.
        Returns the best terminal / deepest node found (root if nothing
        survived).
        """
        self._reset()
        best: ThoughtNode = root
        frontier: List[ThoughtNode] = [root]

        for _ in range(1, max_depth + 1):
            next_level: List[ThoughtNode] = []
            for node in frontier:
                if not self._budget_left():
                    break
                children, had_candidates = self._spawn_children(
                    node, generator_fn, evaluator_fn
                )
                if not had_candidates and self._better(node, best):
                    best = node  # genuine terminal node
                next_level.extend(children)

            if not next_level:
                break

            next_level.sort(key=lambda n: n.cumulative_score, reverse=True)
            frontier = next_level[: self.beam_width]
            if self._better(frontier[0], best):
                best = frontier[0]

            if not self._budget_left():
                break

        return best

    # ------------------------ Depth-first backtracking -------------------- #
    def dfs_backtrack_search(
        self,
        root: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
        max_depth: int,
    ) -> ThoughtNode:
        """
        Follow the highest-scoring child first.  If a branch dead-ends (no
        children, or all children fall below `prune_threshold`) we backtrack
        to the next-best sibling.  The first path reaching `max_depth` is
        returned; otherwise the best partial node seen.  Budget is strict.
        """
        self._reset()
        best: ThoughtNode = root

        def visit(node: ThoughtNode) -> bool:
            """Returns True when a full-depth solution was found."""
            nonlocal best
            if self._better(node, best):
                best = node
            if node.depth >= max_depth:
                return True
            if not self._budget_left():
                return False

            children, _ = self._spawn_children(node, generator_fn, evaluator_fn)
            children.sort(key=lambda n: n.step_score, reverse=True)
            for child in children:
                if visit(child):
                    return True
                if not self._budget_left():
                    return False
            return False  # dead end -> caller backtracks

        visit(root)
        return best


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
def _toy_problem():
    """
    Tiny reasoning graph. 'A' leads to a tempting-but-dead branch (A1 looks
    great, then collapses); 'B' is the genuinely correct chain.
    """
    graph = {
        "start": ["A", "B", "C"],
        "A": ["A1", "A2"],
        "A1": ["A1x", "A1y"],      # all children are bad -> dead end
        "A2": [],
        "B": ["B1", "B2"],
        "B1": ["B1-final"],
        "B2": [],
        "C": ["C1"],
        "C1": ["C1-final"],
    }
    scores = {
        "A": 0.9, "B": 0.8, "C": 0.2,
        "A1": 0.95, "A2": 0.4,
        "A1x": 0.1, "A1y": 0.05,
        "B1": 0.85, "B2": 0.5,
        "B1-final": 0.9,
        "C1": 0.9, "C1-final": 0.9,
    }
    return (lambda s: graph.get(s, [])), (lambda s: scores.get(s, 0.0))


if __name__ == "__main__":
    gen, ev = _toy_problem()

    # --- BFS beam ---
    bfs = TreeOfThoughtsSolver(beam_width=2, max_expansions=20, prune_threshold=0.3)
    res = bfs.bfs_beam_search(ThoughtNode("start"), gen, ev, max_depth=3)
    print("BFS  trajectory :", " -> ".join(res.get_trajectory()))
    print(f"BFS  cumulative : {res.cumulative_score:.4f} | "
          f"expansions={bfs.expansion_count} pruned={bfs.pruned_count}")
    assert res.get_trajectory() == ["start", "B", "B1", "B1-final"] or res.depth == 3
    assert bfs.expansion_count <= bfs.max_expansions

    # --- DFS backtracking (A-branch dead-ends, must backtrack to B) ---
    dfs = TreeOfThoughtsSolver(max_expansions=20, prune_threshold=0.3)
    res = dfs.dfs_backtrack_search(ThoughtNode("start"), gen, ev, max_depth=3)
    print("DFS  trajectory :", " -> ".join(res.get_trajectory()))
    print(f"DFS  cumulative : {res.cumulative_score:.4f} | "
          f"expansions={dfs.expansion_count} pruned={dfs.pruned_count}")
    assert res.get_trajectory() == ["start", "B", "B1", "B1-final"]

    # --- Hard budget enforcement ---
    for budget in (0, 1, 3, 5):
        for name in ("bfs_beam_search", "dfs_backtrack_search"):
            s = TreeOfThoughtsSolver(beam_width=3, max_expansions=budget)
            getattr(s, name)(ThoughtNode("start"), gen, ev, 3)
            assert s.expansion_count <= budget, (name, budget, s.expansion_count)
    print("Budget enforcement: OK")
    print("All Part 1 checks passed.")
