"""Mission 6, Part 3: MDP Trajectory Logger & Empirical Return Evaluator.

Data & Payload Specialist deliverable. Standard library only.

  1. TrajectoryLogger : records tau = (s0, a0, r0, s1, a1, r1, ..., sT)
  2. discounted_return: G0 = sum_{t=0}^{T} gamma^t * r_t   (gamma = 0.95)
  3. transition_frequency_matrix: empirical P(s' | s, a) from historical jumps

Run:  python 03_MDP_Trajectory_Logger.py
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

GAMMA = 0.95


# --------------------------------------------------------------------------
# 1. Trajectory logging
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """One MDP step: in state s the agent took action a and got reward r."""
    state: Hashable
    action: Hashable
    reward: float


@dataclass
class TrajectoryLogger:
    """Records an episode tau = (s0, a0, r0, s1, a1, r1, ..., sT).

    Usage:
        log = TrajectoryLogger(initial_state="s0")
        log.record(action="a0", reward=10, next_state="s1")
        ...
        log.finish()            # optional: seals the episode
    The terminal state s_T has no action/reward, so there is always
    one more state than there are steps.
    """
    initial_state: Hashable
    steps: list[Step] = field(default_factory=list)
    final_state: Hashable = None
    finished: bool = False

    def __post_init__(self) -> None:
        self.final_state = self.initial_state

    @property
    def current_state(self) -> Hashable:
        return self.final_state

    def record(self, action: Hashable, reward: float, next_state: Hashable) -> None:
        """Log (s_t, a_t, r_t) and advance to s_{t+1}."""
        if self.finished:
            raise RuntimeError("episode already finished; start a new logger")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) \
                or not math.isfinite(reward):
            raise ValueError(f"reward must be a finite number, got {reward!r}")
        self.steps.append(Step(self.final_state, action, float(reward)))
        self.final_state = next_state

    def finish(self) -> "TrajectoryLogger":
        self.finished = True
        return self

    # views -----------------------------------------------------------------
    @property
    def states(self) -> list:          # s_0 .. s_T  (length T+1)
        return [s.state for s in self.steps] + [self.final_state]

    @property
    def actions(self) -> list:         # a_0 .. a_{T-1}
        return [s.action for s in self.steps]

    @property
    def rewards(self) -> list[float]:  # r_0 .. r_{T-1}
        return [s.reward for s in self.steps]

    def as_tau(self) -> list:
        """Flat interleaved form (s0, a0, r0, s1, a1, r1, ..., sT)."""
        out: list = []
        for st in self.steps:
            out += [st.state, st.action, st.reward]
        out.append(self.final_state)
        return out

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return "tau = (" + ", ".join(map(str, self.as_tau())) + ")"


# --------------------------------------------------------------------------
# 2. Discounted return
# --------------------------------------------------------------------------
def discounted_return(rewards: Sequence[float] | TrajectoryLogger,
                      gamma: float = GAMMA) -> float:
    """G0 = sum_t gamma^t * r_t, accumulated by Horner's rule (stable, O(T))."""
    if isinstance(rewards, TrajectoryLogger):
        rewards = rewards.rewards
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def returns_to_go(rewards: Sequence[float], gamma: float = GAMMA) -> list[float]:
    """G_t for every t (G_t = r_t + gamma * G_{t+1}); G_0 is the first entry."""
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


# --------------------------------------------------------------------------
# 3. Transition frequency matrix  P(s' | s, a)
# --------------------------------------------------------------------------
# A historical jump record: (sector_state, action, next_sector_state)
Jump = tuple


@dataclass
class TransitionModel:
    counts: dict            # (s, a) -> Counter({s': n})
    states: list            # sorted state labels (matrix columns)

    def n(self, s, a) -> int:
        return sum(self.counts.get((s, a), {}).values())

    def prob(self, s, a, s_next) -> float:
        """Empirical P(s' | s, a); 0.0 for an unseen (s, a) pair."""
        total = self.n(s, a)
        return self.counts[(s, a)][s_next] / total if total else 0.0

    def row(self, s, a) -> dict:
        return {sp: self.prob(s, a, sp) for sp in self.states}

    def matrix(self):
        """Return (row_labels [(s, a)], column_labels, rows of probabilities)."""
        keys = sorted(self.counts, key=lambda k: (str(k[0]), str(k[1])))
        return keys, self.states, [[self.prob(s, a, sp) for sp in self.states]
                                   for s, a in keys]

    def success_rate(self, s, a, success_states: Iterable) -> float:
        """P(s' in success_states | s, a), e.g. 'navigated safely'."""
        ok = set(success_states)
        return sum(self.prob(s, a, sp) for sp in ok)

    def format(self) -> str:
        keys, cols, rows = self.matrix()
        w = max([len(str(c)) for c in cols] + [6]) + 1
        lab = max(len(f"({s}, {a})") for s, a in keys) + 2
        head = " " * lab + "".join(f"{str(c):>{w}}" for c in cols) + "      n"
        lines = [head]
        for (s, a), r in zip(keys, rows):
            lines.append(f"{f'({s}, {a})':<{lab}}" +
                         "".join(f"{p:>{w}.3f}" for p in r) + f"{self.n(s, a):>7}")
        return "\n".join(lines)


def transition_frequency_matrix(jumps: Iterable[Jump]) -> TransitionModel:
    """Maximum-likelihood estimate P(s'|s,a) = N(s,a,s') / N(s,a)."""
    counts: dict = defaultdict(Counter)
    states: set = set()
    for s, a, sp in jumps:
        counts[(s, a)][sp] += 1
        states.update((s, sp))
    if not counts:
        raise ValueError("jump dataset is empty")
    return TransitionModel(dict(counts), sorted(states, key=str))


def jumps_from_trajectory(log: TrajectoryLogger) -> list[Jump]:
    """Turn a logged episode into (s, a, s') records for the matrix."""
    s = log.states
    return [(s[t], log.actions[t], s[t + 1]) for t in range(len(log))]


# Example historical dataset of past jumps through the nebula:
# (sector, warp action, outcome sector). "SAFE" / "DESTROYED" are absorbing.
HISTORICAL_JUMPS: list[Jump] = (
    [("CALM", "WARP_SLOW", "CALM")] * 18 + [("CALM", "WARP_SLOW", "TURBULENT")] * 2
    + [("CALM", "WARP_FAST", "CALM")] * 9 + [("CALM", "WARP_FAST", "TURBULENT")] * 8
    + [("CALM", "WARP_FAST", "DESTROYED")] * 3
    + [("TURBULENT", "WARP_SLOW", "CALM")] * 11 + [("TURBULENT", "WARP_SLOW", "TURBULENT")] * 7
    + [("TURBULENT", "WARP_SLOW", "DESTROYED")] * 2
    + [("TURBULENT", "WARP_FAST", "SAFE")] * 6 + [("TURBULENT", "WARP_FAST", "TURBULENT")] * 6
    + [("TURBULENT", "WARP_FAST", "DESTROYED")] * 8
    + [("CALM", "WARP_ESCAPE", "SAFE")] * 14 + [("CALM", "WARP_ESCAPE", "TURBULENT")] * 6
)


# --------------------------------------------------------------------------
# Self-test (assert-based, matches the Master Test Suite style)
# --------------------------------------------------------------------------
def _selftest() -> None:
    # Master test 3: rewards [10, 0, -5, 20], gamma = 0.95
    rewards = [10, 0, -5, 20]
    expected = 10 + 0.95 * 0 + 0.95 ** 2 * (-5) + 0.95 ** 3 * 20   # = 22.635
    assert math.isclose(expected, 22.635, abs_tol=1e-9)
    assert math.isclose(discounted_return(rewards, GAMMA), expected, rel_tol=1e-12)
    assert math.isclose(discounted_return(rewards), 22.635, abs_tol=1e-9)

    # edge cases + consistency
    assert discounted_return([]) == 0.0
    assert discounted_return([7], 0.5) == 7.0
    assert discounted_return(rewards, 0.0) == 10
    assert discounted_return(rewards, 1.0) == sum(rewards)
    assert math.isclose(returns_to_go(rewards)[0], expected)
    assert math.isclose(returns_to_go(rewards)[-1], 20)
    try:
        discounted_return(rewards, 1.5)
        raise AssertionError("gamma > 1 must be rejected")
    except ValueError:
        pass

    # logger builds tau = (s0, a0, r0, ..., sT)
    log = TrajectoryLogger("s0")
    for a, r, sn in [("a0", 10, "s1"), ("a1", 0, "s2"),
                     ("a2", -5, "s3"), ("a3", 20, "s4")]:
        log.record(a, r, sn)
    log.finish()
    assert log.as_tau() == ["s0", "a0", 10.0, "s1", "a1", 0.0, "s2", "a2", -5.0,
                            "s3", "a3", 20.0, "s4"]
    assert len(log.states) == len(log) + 1 == 5
    assert math.isclose(discounted_return(log), expected)
    try:
        log.record("a4", 1, "s5")
        raise AssertionError("finished episode must reject new steps")
    except RuntimeError:
        pass
    try:
        TrajectoryLogger("s").record("a", float("nan"), "s2")
        raise AssertionError("NaN reward must be rejected")
    except ValueError:
        pass

    # transition matrix
    model = transition_frequency_matrix(HISTORICAL_JUMPS)
    assert model.n("CALM", "WARP_SLOW") == 20
    assert math.isclose(model.prob("CALM", "WARP_SLOW", "CALM"), 0.9)
    assert math.isclose(model.prob("CALM", "WARP_SLOW", "SAFE"), 0.0)
    assert math.isclose(model.prob("TURBULENT", "WARP_FAST", "DESTROYED"), 0.4)
    assert model.prob("NEBULA_CORE", "WARP_SLOW", "CALM") == 0.0   # unseen pair
    keys, cols, rows = model.matrix()
    assert all(math.isclose(sum(r), 1.0) for r in rows)             # rows sum to 1
    assert math.isclose(model.success_rate("CALM", "WARP_ESCAPE", {"SAFE"}), 0.7)
    assert jumps_from_trajectory(log)[0] == ("s0", "a0", "s1")


if __name__ == "__main__":
    _selftest()

    print("--- 1. Trajectory log ---")
    demo = TrajectoryLogger("CALM")
    demo.record("WARP_SLOW", 10, "CALM")
    demo.record("WARP_SLOW", 0, "TURBULENT")
    demo.record("WARP_SLOW", -5, "CALM")
    demo.record("WARP_ESCAPE", 20, "SAFE")
    demo.finish()
    print(demo)

    print("\n--- 2. Discounted return (gamma = 0.95) ---")
    print(f"rewards = {demo.rewards}")
    print(f"G0 = {discounted_return(demo):.4f}  (expected 22.6350)")

    print("\n--- 3. Empirical P(s' | s, a) from historical jumps ---")
    m = transition_frequency_matrix(HISTORICAL_JUMPS)
    print(m.format())
    print(f"\nP(reach SAFE | CALM, WARP_ESCAPE) = "
          f"{m.success_rate('CALM', 'WARP_ESCAPE', {'SAFE'}):.3f}")

    print("\nAll Part 3 checks passed.")
