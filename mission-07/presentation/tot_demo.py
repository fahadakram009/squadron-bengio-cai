"""
Week 7 lightning presentation, Topic 1: Branching Beyond Greedy Decoding (Tree-of-Thoughts)
Live demo for Squadron Bengio's tot_solver.py (Fahad Akram, Systems Architect)

    python presentation/tot_demo.py            # the riddle, three searches with tree traces, the results table
    python presentation/tot_demo.py --live     # recompute the Game-of-24 table on the spot (about 10 s)
    python presentation/tot_demo.py --step     # pause between acts (press Enter)

Runs from the mission-07 folder and imports the real solver, nothing is mocked.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tot_solver", HERE.parent / "tot_solver.py")
tot = importlib.util.module_from_spec(spec)
sys.modules["tot_solver"] = tot
spec.loader.exec_module(tot)

# ---------------------------------------------------------------------------------------------------
# The riddle: escape the Sagittarius A* accretion disk in three burns.
# A greedy pilot takes the move that LOOKS best at every step and flies into a trap.
# ---------------------------------------------------------------------------------------------------

ROOT = "Ship at 3 r_s, fuel 100, disk plasma rising"

MOVES = {
    ROOT: [
        "A: Full retrograde burn (fuel -60): fastest drop in orbital energy",
        "B: Prograde slingshot toward the companion star (fuel -40)",
        "C: Jettison cargo pods to lighten the ship",
    ],
    "A: Full retrograde burn (fuel -60): fastest drop in orbital energy": [
        "A1: Coast straight through the relativistic jet (saves fuel, looks brilliant)",
        "A2: Raise shields and drift (slow, fuel -20)",
    ],
    "A1: Coast straight through the relativistic jet (saves fuel, looks brilliant)": [
        "A1x: Jet drag cancels the burn, fuel 0, trapped on the disk",
        "A1y: Hull breach from synchrotron flux",
    ],
    "A2: Raise shields and drift (slow, fuel -20)": [],
    "B: Prograde slingshot toward the companion star (fuel -40)": [
        "B1: Gravity assist at periapsis (fuel -10)",
        "B2: Abort the slingshot and circularise",
    ],
    "B1: Gravity assist at periapsis (fuel -10)": [
        "B1*: Escape velocity reached with fuel 30 [GOAL]",
    ],
    "B2: Abort the slingshot and circularise": [],
    "C: Jettison cargo pods to lighten the ship": [
        "C1: Cargo pods ignite in the disk, radiation spike",
    ],
}

# What a (slightly myopic) verifier thinks of each move. A looks better than B. A1 looks best of all.
SCORES = {
    "A: Full retrograde burn (fuel -60): fastest drop in orbital energy": 0.90,
    "B: Prograde slingshot toward the companion star (fuel -40)": 0.80,
    "C: Jettison cargo pods to lighten the ship": 0.20,
    "A1: Coast straight through the relativistic jet (saves fuel, looks brilliant)": 0.95,
    "A2: Raise shields and drift (slow, fuel -20)": 0.40,
    "A1x: Jet drag cancels the burn, fuel 0, trapped on the disk": 0.10,
    "A1y: Hull breach from synchrotron flux": 0.05,
    "B1: Gravity assist at periapsis (fuel -10)": 0.85,
    "B2: Abort the slingshot and circularise": 0.50,
    "B1*: Escape velocity reached with fuel 30 [GOAL]": 0.90,
    "C1: Cargo pods ignite in the disk, radiation spike": 0.90,
}


def generator(node):
    return MOVES.get(node.state, [])


def evaluator(thought):
    return SCORES.get(thought, 0.0)


def is_goal(node):
    return "[GOAL]" in node.state


# ---------------------------------------------------------------------------------------------------
# Tree trace printer
# ---------------------------------------------------------------------------------------------------

def short(state: str, width: int = 58) -> str:
    return state if len(state) <= width else state[: width - 1] + "…"


def render_tree(root, solver, winner):
    """Indented ASCII tree with the solver's verdict on every node."""
    kept = {e["state"] for e in solver.events if e["event"] == "keep"}
    dropped = {e["state"] for e in solver.events if e["event"] == "drop"}
    backtracked = {e["state"] for e in solver.events if e["event"] == "backtrack"}
    path = set(winner.get_trajectory())
    lines = []

    def walk(node, prefix="", last=True):
        if node is root:
            mark, label = "", node.state
        else:
            if node.is_pruned:
                mark = "✂ pruned"
            elif "[GOAL]" in node.state and node in [winner]:
                mark = "★ GOAL"
            elif node.state in backtracked:
                mark = "↩ backtracked"
            elif node.state in dropped:
                mark = "· dropped from beam"
            elif node.state in kept:
                mark = "✔ kept"
            else:
                mark = "✔ explored"
            label = f"{short(node.state)}  [{node.step_score:.2f}]  {mark}"
            if node.state in path and not node.is_pruned:
                label = "» " + label
        lines.append(prefix + ("└─ " if last else "├─ ") + label if node is not root else label)
        kids = node.children
        for i, child in enumerate(kids):
            ext = "    " if last else "│   "
            walk(child, prefix + (ext if node is not root else ""), i == len(kids) - 1)

    walk(root)
    return "\n".join(lines)


def run(title, mode, beam_width, max_expansions=20):
    solver = tot.TreeOfThoughtsSolver(max_expansions=max_expansions, beam_width=beam_width,
                                      prune_threshold=0.45, terminal_fn=is_goal)
    root = tot.ThoughtNode(ROOT)
    search = solver.bfs_beam_search if mode == "bfs" else solver.dfs_backtrack_search
    best = search(root, generator, evaluator, max_depth=3)
    print(f"\n=== {title} ===")
    print(render_tree(root, solver, best))
    outcome = "ESCAPED" if "[GOAL]" in best.state else f"STUCK at depth {best.depth}"
    print(f"result: {outcome} | expansions {solver.expansion_count}/{max_expansions} | pruned {solver.pruned_count}"
          + (f" | backtracks {solver.backtrack_count}" if mode == "dfs" else ""))
    print("route : " + "  →  ".join(s.split(":")[0] for s in best.get_trajectory()[1:]) if best.depth else "route : none")
    return best, solver


STORED_TABLE = """
Game of 24, 40 solvable puzzles, depth 3, noisy verifier (sigma = 0.5)     solve rate | expansions
  config          budget 48     budget 96    budget 192
  BFS B=1 greedy   0.20 | 42     0.20 | 43     0.20 | 43     <- locks onto a flattered wrong step
  BFS B=2          0.07 | 48     0.42 | 61     0.42 | 61
  BFS B=3          0.00 | 48     0.47 | 79     0.47 | 79     <- width keeps the true step alive
  BFS B=5          0.00 | 48     0.25 | 95     0.47 | 114
  DFS backtrack    0.40 | 45     0.50 | 47     0.50 | 47     <- pays for siblings only when a branch dies
With an exact verifier every config reaches 1.00; the ceiling of ~0.5 is the verifier's, not the search's.
"""


def live_table():
    bench = tot.benchmark(n_puzzles=40, budgets=(48, 96, 192), sigmas=(0.5,), seed=7, quiet=False)
    return bench


def pause(enabled):
    if enabled:
        input("\n[Enter] ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--step", action="store_true")
    args = ap.parse_args()

    print("THE RIDDLE: escape the Sagittarius A* accretion disk in three burns.")
    print("Each move is scored by a verifier that is good but myopic. A looks better than B; A1 looks best of all.")
    print("Legend: [score] = verifier step score, prune threshold 0.45, budget 20 expansions.")
    pause(args.step)

    run("ACT 1  greedy pilot = beam width 1 (what a one-shot LLM does)", "bfs", 1)
    pause(args.step)
    run("ACT 2  beam width 2 keeps a second opinion alive", "bfs", 2)
    pause(args.step)
    run("ACT 3  depth-first with backtracking: dive, hit the wall, climb back out", "dfs", 1)
    pause(args.step)

    print("\n=== ACT 4  does it hold on a real problem? ===")
    if args.live:
        live_table()
    else:
        print(STORED_TABLE.strip())
    print("\nTake-away: search turns compute into alternatives; the verifier decides which alternative deserves the next expansion.")


if __name__ == "__main__":
    main()
