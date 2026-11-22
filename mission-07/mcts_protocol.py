"""
Mission 07 - Part 2: Communications & Protocol Officer (Madhu)
4-Phase Monte Carlo Tree Search (MCTS) & PUCT Selection Engine

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator

- PUCT(s, a) = Q(s, a) + c_puct * P(a|s) * sqrt(max(1, N(s))) / (1 + N(s, a))
- one simulation = select a leaf, expand it, roll out the new node, backpropagate the reward
- after the search the move to play is the child with the most visits
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class MCTSNode:
    """Represents a node in the Monte Carlo Reasoning Tree."""
    state: str
    prior_prob: float = 1.0
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Empirical mean action-value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSEngine:
    """
    The 4 phases of MCTS for LLM reasoning:
    1. Selection (PUCT)
    2. Expansion (candidate reasoning steps with policy priors P(a|s))
    3. Simulation / Rollout (value of the new step, e.g. from a PRM)
    4. Backpropagation (update N and W on the path back to the root)
    """

    def __init__(self, c_puct: float = 1.5):
        if c_puct < 0:
            raise ValueError("c_puct must be >= 0")
        self.c_puct = c_puct

    def puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(max(1, N(s))) / (1 + N(s, a))"""
        parent_visits = max(1, parent.visit_count)
        u_value = self.c_puct * child.prior_prob * (math.sqrt(parent_visits) / (1 + child.visit_count))
        return child.q_value + u_value

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Child of `node` with the highest PUCT score (first one wins a tie)."""
        if not node.children:
            raise ValueError("Cannot select a child from a leaf node with no children.")
        return max(node.children, key=lambda child: self.puct_score(node, child))

    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """
        Phase 1 (Selection): Traverse down the tree from `root` using `select_best_child`
        until reaching a leaf node (no children or terminal).
        """
        curr = root
        while curr.children and not curr.is_terminal:
            curr = self.select_best_child(curr)
        return curr

    def expand(self, node: MCTSNode, action_priors: Dict[str, float]) -> List[MCTSNode]:
        """Phase 2 (Expansion): one child per (state_str, prior_prob) pair."""
        if node.is_terminal:  # a finished trajectory has no next step
            return node.children
        for state_str, prior_prob in action_priors.items():
            if prior_prob < 0:
                raise ValueError(f"Negative prior for '{state_str}': {prior_prob}")
        existing = {child.state for child in node.children}
        for state_str, prior_prob in action_priors.items():
            if state_str not in existing:  # no duplicate children if expand is called twice
                node.children.append(MCTSNode(state=state_str, prior_prob=prior_prob, parent=node))
        return node.children

    def simulate_rollout(self, node: MCTSNode, rollout_evaluator: Callable[[str], float]) -> float:
        """Phase 3 (Simulation / Rollout): evaluate node.state, clamp to [0.0, 1.0]."""
        reward = float(rollout_evaluator(node.state))
        # min/max would turn NaN into 1.0, so a broken evaluator counts as reward 0
        if math.isnan(reward):
            return 0.0
        return max(0.0, min(1.0, reward))

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Phase 4 (Backpropagation): N += 1 and W += reward from `node` up to the root."""
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += reward
            curr = curr.parent

    def search(self, root: MCTSNode, n_simulations: int,
               prior_fn: Callable[[str], Dict[str, float]],
               rollout_evaluator: Callable[[str], float]) -> MCTSNode:
        """Run n_simulations of the 4 phases and return the most visited child of root."""
        for _ in range(n_simulations):
            leaf = self.select_leaf(root)
            if not leaf.is_terminal:
                priors = prior_fn(leaf.state)
                if priors:
                    self.expand(leaf, priors)
                    leaf = self.select_best_child(leaf)
                else:
                    leaf.is_terminal = True  # no next steps: dead end
            self.backpropagate(leaf, self.simulate_rollout(leaf, rollout_evaluator))
        if not root.children:
            raise ValueError("Search found no moves from the root.")
        return max(root.children, key=lambda c: (c.visit_count, c.q_value))


# --- verification -------------------------------------------------------------------

def trace_tree():
    """The PUCT trace from the brief / doc Section B: N(s)=16, A: Q=0.75 N=5 P=0.50, B: Q=0.55 N=1 P=0.40."""
    root = MCTSNode(state="Root: Sagittarius A* Ergosphere Geodesic Entry", visit_count=16)
    branch_a = MCTSNode(state="Branch A: Prograde Frame-Dragging Assist", prior_prob=0.50, parent=root,
                        visit_count=5, value_sum=0.75 * 5)
    branch_b = MCTSNode(state="Branch B: Retrograde Penrose Extraction", prior_prob=0.40, parent=root,
                        visit_count=1, value_sum=0.55 * 1)
    root.children = [branch_a, branch_b]
    return root, branch_a, branch_b


# small reasoning tree for the full search: the policy likes A best, but B leads to the safe orbit
PRIORS = {
    "Root": {"A: prograde frame-dragging assist": 0.6,
             "B: retrograde Penrose extraction": 0.3,
             "C: direct plunge": 0.1},
    "A: prograde frame-dragging assist": {"A1: tighten orbit": 0.5, "A2: boost along disk": 0.5},
    "B: retrograde Penrose extraction": {"B1: split payload in ergosphere": 0.7, "B2: hold altitude": 0.3},
}
VALUES = {
    "A: prograde frame-dragging assist": 0.40, "A1: tighten orbit": 0.20, "A2: boost along disk": 0.35,
    "B: retrograde Penrose extraction": 0.60, "B1: split payload in ergosphere": 0.90, "B2: hold altitude": 0.55,
    "C: direct plunge": 0.0,
}


def run_search(c_puct: float = 1.5, n_simulations: int = 60):
    engine = MCTSEngine(c_puct=c_puct)
    root = MCTSNode(state="Root")
    best = engine.search(root, n_simulations, lambda s: PRIORS.get(s, {}), lambda s: VALUES.get(s, 0.0))
    return root, best


def selected_for(c_puct: float) -> str:
    root, a, b = trace_tree()
    return "A" if MCTSEngine(c_puct).select_best_child(root) is a else "B"


def run_tests() -> int:
    engine = MCTSEngine(c_puct=1.5)
    root, a, b = trace_tree()
    tests = [
        ("PUCT(A) = 0.75 + 1.5*0.50*4/6 = 1.25", abs(engine.puct_score(root, a) - 1.25) < 1e-6),
        ("PUCT(B) = 0.55 + 1.5*0.40*4/2 = 1.75", abs(engine.puct_score(root, b) - 1.75) < 1e-6),
        ("under-explored Branch B is selected", engine.select_best_child(root) is b),
    ]

    engine.backpropagate(b, reward=0.95)
    tests.append(("backpropagate: N(B) 1->2, N(root) 16->17, W(B) = 0.55+0.95",
                  b.visit_count == 2 and root.visit_count == 17 and abs(b.value_sum - 1.50) < 1e-9
                  and a.visit_count == 5))

    fresh_root = MCTSNode(state="r")
    child = MCTSNode(state="c", prior_prob=0.4, parent=fresh_root)
    tests.append(("unvisited child: Q = 0 and N(s) counts as 1 -> PUCT = 1.5*0.4",
                  child.q_value == 0.0 and abs(engine.puct_score(fresh_root, child) - 0.6) < 1e-9))

    tests.append(("c_puct 0.4 picks A (exploit), 0.5 / 1.5 / 2.5 pick B (explore)",
                  [selected_for(c) for c in (0.4, 0.5, 1.5, 2.5)] == ["A", "B", "B", "B"]))

    node = MCTSNode(state="n")
    kids = engine.expand(node, {"x": 0.7, "y": 0.3})
    tests.append(("expand: one child per pair, with its prior and parent",
                  [k.state for k in kids] == ["x", "y"] and [k.prior_prob for k in kids] == [0.7, 0.3]
                  and all(k.parent is node for k in kids)))
    again = engine.expand(node, {"x": 0.7, "z": 0.2})
    tests.append(("second expand adds only the new step (no duplicate 'x')",
                  [k.state for k in again] == ["x", "y", "z"]))
    term = MCTSNode(state="t", is_terminal=True)
    bad = MCTSNode(state="b")
    try:
        engine.expand(bad, {"ok": 0.5, "neg": -0.1})
        neg_rejected = False
    except ValueError:
        neg_rejected = not bad.children
    tests.append(("terminal node is not expanded, negative prior rejected",
                  engine.expand(term, {"x": 1.0}) == [] and neg_rejected))

    clamp = [engine.simulate_rollout(node, lambda s, r=r: r) for r in (1.7, -0.2, 0.42, float("nan"))]
    tests.append(("rollout clamped to [0, 1], NaN -> 0.0", clamp == [1.0, 0.0, 0.42, 0.0]))

    t_root = MCTSNode(state="r")
    stop = MCTSNode(state="stop", parent=t_root, is_terminal=True, children=[MCTSNode(state="below")])
    t_root.children = [stop]
    tests.append(("select_leaf stops at a terminal node", engine.select_leaf(t_root) is stop))

    try:
        engine.search(MCTSNode(state="empty"), 5, lambda s: {}, lambda s: 0.5)
        empty_ok = False
    except ValueError:
        empty_ok = True
    tests.append(("search with no possible moves gives a clear error", empty_ok))

    s_root, best = run_search()
    tests.append(("full search: 60 simulations, picks B over the higher-prior A",
                  s_root.visit_count == 60 and best.state.startswith("B")
                  and sum(c.visit_count for c in s_root.children) == 60))

    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = sum(not ok for _, ok in tests)
    print(f"--- {len(tests) - failed} passed, {failed} failed ---\n")
    return failed


def report() -> None:
    root, a, b = trace_tree()
    engine = MCTSEngine(c_puct=1.5)
    print("PUCT selection trace (c_puct = 1.5):")
    for name, n in (("A", a), ("B", b)):
        u = engine.puct_score(root, n) - n.q_value
        print(f"  Branch {name}: Q={n.q_value:.2f} P={n.prior_prob:.2f} N(s)={root.visit_count} "
              f"N(s,a)={n.visit_count} U={u:.4f} PUCT={engine.puct_score(root, n):.4f}")
    print(f"  selected: {engine.select_best_child(root).state}\n")

    print("c_puct sweep on the same two branches:")
    for c in (0.4, 0.5, 1.0, 1.5, 2.5):
        e = MCTSEngine(c)
        print(f"  c_puct={c:<4} PUCT(A)={e.puct_score(root, a):.4f} PUCT(B)={e.puct_score(root, b):.4f} "
              f"-> {selected_for(c)}")
    print("  B overtakes A above c_puct = 0.2 / (0.8 - 1/3) = 0.4286\n")

    for c in (0.5, 1.5, 2.5):
        s_root, best = run_search(c_puct=c)
        visits = ", ".join(f"{ch.state.split(':')[0]}={ch.visit_count}" for ch in s_root.children)
        print(f"Full search, c_puct={c}, 60 simulations: visits {visits} -> play {best.state}")
    print()


if __name__ == "__main__":
    print("=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===")
    report()
    failed = run_tests()
    if failed == 0:
        print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")
    sys.exit(1 if failed else 0)
