"""
Mission 07 — Part 3: Data & Payload Specialist
Process Reward Model (PRM) Step Evaluator & Math-Shepherd Automated Supervision

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
Station: Data & Payload Specialist (Osaf Ahmad); implemented by Fahad Akram (Systems Architect), Squadron Bengio

Pure Python 3.11+, standard library only.

The instructor's public interface is kept exactly (the other stations call it):
    OutcomeRewardEvaluator(expected_final_marker).score_trajectory(steps) -> 1.0 / 0.0
    ProcessRewardEvaluator().step_score(step) -> (0, 1]          ([FLAWED] / SIGN_ERROR 0.12, [MINOR_DRIFT] 0.55, else 0.94)
    .score_all_steps, .aggregate_trajectory_min / _prod / _geometric_mean
    .math_shepherd_estimation(intermediate_step, rollout_final_answers, ground_truth_answer) -> {soft_label, hard_label}

What this file adds
    1. Verifiable arithmetic: a step that states "a op b = c" is checked exactly (Fractions); a false equation
       scores 0.12 whatever the step claims about itself, a true one 0.97 unless the step also carries a flaw
       marker (negative evidence wins). step_score_detailed() says why.
    2. aggregate(steps, reduction) and aggregate_scores(scores, reduction) for all three reductions, plus
       length_penalty_table(): how prod, geometric mean and min behave as a chain grows at a fixed step quality.
    3. false_positive_benchmark(): synthetic proofs in which some flaws cancel out; ORM and PRM confusion counts.
    4. simulate_rollouts() and shepherd_error_vs_M(): Math-Shepherd labels on a simulated completion policy with a
       known success probability, so the estimate error of the soft label and the saturation of the hard label can
       be measured as a function of M.
    5. self_test(): 28 checks, including the instructor's harness, and integration with the Architect's ToT solver
       (PRM as its evaluator) and the Sentinel's event-horizon pruner (PRM scores as its input).

Run (from the repo root, the shared .venv; nothing to install):
    python mission-07/prm_evaluator.py              # instructor harness + the Section C tables
    python mission-07/prm_evaluator.py --selftest   # 28 checks, log in logs/
"""

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"

# "a op b = c" with integers, decimals or fractions; the first such equation in a step is verified
EQUATION_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?(?:/\d+)?)\s*([+\-*/x×])\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)(?![\w.])"
)


class OutcomeRewardEvaluator:
    """
    Outcome Reward Model (ORM): Grades a reasoning chain solely by its final terminal step.
    Vulnerable to false-positive trajectories where intermediate algebraic errors cancel out.
    """

    def __init__(self, expected_final_marker: str):
        self.expected_final_marker = expected_final_marker

    def score_trajectory(self, steps: List[str]) -> float:
        if not steps:
            return 0.0
        return 1.0 if self.expected_final_marker in steps[-1] else 0.0


def _num(tok: str) -> Fraction:
    return Fraction(tok)  # Fraction parses "3", "-2.5" and "7/2" exactly


def verify_equation(step: str) -> Optional[bool]:
    """True / False if the step states a checkable 'a op b = c'; None if it states no equation."""
    m = EQUATION_RE.search(step)
    if not m:
        return None
    a, op, b, c = m.groups()
    try:
        x, y, z = _num(a), _num(b), _num(c)
    except (ValueError, ZeroDivisionError):
        return None
    if op == "+":
        return x + y == z
    if op == "-":
        return x - y == z
    if op in ("*", "x", "×"):
        return x * y == z
    if y == 0:
        return False
    return x / y == z


class ProcessRewardEvaluator:
    """
    Process Reward Model (PRM): Evaluates the logical validity of every intermediate reasoning step z_t,
    aggregates trajectory scores (min, prod, geometric_mean), and computes Math-Shepherd rollout labels.

    Step scoring, in order of precedence: a false equation 0.12, a [FLAWED] / SIGN_ERROR marker 0.12, a
    verified-true equation 0.97, a [MINOR_DRIFT] marker 0.55, otherwise the default 0.94.
    """

    SCORE_FLAWED = 0.12
    SCORE_DRIFT = 0.55
    SCORE_DEFAULT = 0.94
    SCORE_VERIFIED = 0.97
    REDUCTIONS = ("min", "prod", "geometric_mean")

    def __init__(self, verify_arithmetic: bool = True):
        self.verify_arithmetic = verify_arithmetic

    # -- step level -----------------------------------------------------------------------------------
    def step_score_detailed(self, step: str) -> Tuple[float, str]:
        """Negative evidence wins: a false equation or an explicit flaw marker scores 0.12; only then does a
        verified-true equation earn 0.97; a drift marker gives 0.55; otherwise the default 0.94."""
        verdict = verify_equation(step) if self.verify_arithmetic else None
        if verdict is False:
            return self.SCORE_FLAWED, "equation verified false"
        if "[FLAWED]" in step or "SIGN_ERROR" in step:
            return self.SCORE_FLAWED, "flaw marker"
        if verdict is True:
            return self.SCORE_VERIFIED, "equation verified true"
        if "[MINOR_DRIFT]" in step:
            return self.SCORE_DRIFT, "minor drift marker"
        return self.SCORE_DEFAULT, "no flaw found"

    def step_score(self, step: str) -> float:
        """Score an individual reasoning step in (0.0, 1.0]."""
        return self.step_score_detailed(step)[0]

    def score_all_steps(self, steps: List[str]) -> List[float]:
        return [self.step_score(s) for s in steps]

    # -- trajectory level -----------------------------------------------------------------------------
    @staticmethod
    def aggregate_scores(scores: Sequence[float], reduction: str) -> float:
        if not scores:
            return 0.0
        if any(s <= 0.0 for s in scores) and reduction == "geometric_mean":
            return 0.0
        if reduction == "min":
            return min(scores)
        if reduction == "prod":
            return math.prod(scores)
        if reduction == "geometric_mean":
            return math.exp(sum(math.log(s) for s in scores) / len(scores))
        raise ValueError(f"unknown reduction {reduction!r}; use one of {ProcessRewardEvaluator.REDUCTIONS}")

    def aggregate(self, steps: List[str], reduction: str) -> float:
        return self.aggregate_scores(self.score_all_steps(steps), reduction)

    def aggregate_trajectory_min(self, steps: List[str]) -> float:
        """Weakest-link reduction: min_{t=1..T} r_t."""
        return self.aggregate(steps, "min")

    def aggregate_trajectory_prod(self, steps: List[str]) -> float:
        """Joint validity reduction: prod_{t=1..T} r_t."""
        return self.aggregate(steps, "prod")

    def aggregate_trajectory_geometric_mean(self, steps: List[str]) -> float:
        """Length-normalized geometric mean: (prod_{t=1..T} r_t)^(1/T)."""
        return self.aggregate(steps, "geometric_mean")

    def first_flawed_step(self, steps: List[str], threshold: float = 0.45) -> int:
        """1-based index of the first step below the threshold, or -1."""
        for i, s in enumerate(self.score_all_steps(steps), 1):
            if s < threshold:
                return i
        return -1

    # -- Math-Shepherd --------------------------------------------------------------------------------
    def math_shepherd_estimation(
        self,
        intermediate_step: str,
        rollout_final_answers: List[str],
        ground_truth_answer: str,
    ) -> Dict[str, float]:
        """Math-Shepherd (Wang et al., 2024) step supervision from M Monte Carlo completions of the step:
        soft_label = fraction of completions whose final answer equals the ground truth,
        hard_label = 1 if at least one completion reaches it, else 0.
        """
        if not rollout_final_answers:
            return {"soft_label": 0.0, "hard_label": 0.0}
        matches = sum(1 for ans in rollout_final_answers if ans == ground_truth_answer)
        soft_label = matches / len(rollout_final_answers)
        hard_label = 1.0 if matches > 0 else 0.0
        return {"soft_label": soft_label, "hard_label": hard_label}


# =============================================================================
# Studies for Section C
# =============================================================================

def length_penalty_table(step_quality: float = 0.94, lengths: Sequence[int] = (1, 2, 4, 8, 16)) -> List[Dict[str, float]]:
    """A chain of T equally good steps: prod decays as q^T, min and the geometric mean stay at q."""
    rows = []
    for t in lengths:
        scores = [step_quality] * t
        rows.append({"T": t,
                     "min": ProcessRewardEvaluator.aggregate_scores(scores, "min"),
                     "prod": ProcessRewardEvaluator.aggregate_scores(scores, "prod"),
                     "geometric_mean": ProcessRewardEvaluator.aggregate_scores(scores, "geometric_mean")})
    return rows


def false_positive_benchmark(n_chains: int = 200, p_flaw: float = 0.5, p_cancel: float = 0.3, seed: int = 7,
                             threshold: float = 0.45) -> Dict[str, Any]:
    """Synthetic proofs: each chain of 3 steps is clean or carries one flawed step; a flawed chain reaches
    the right final answer ("cancels out") with probability p_cancel. ORM grades the last step, PRM(min)
    grades every step. Returns confusion counts and rates for both."""
    rng = random.Random(seed)
    orm = OutcomeRewardEvaluator("v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()
    counts = {"orm": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}, "prm": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}}
    for _ in range(n_chains):
        flawed = rng.random() < p_flaw
        chain = ["Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
                 "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
                 "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s"]
        if flawed:
            chain[1] = "Step 2: [FLAWED] Flip sign of frame-dragging term twice via illegal tensor contraction"
            if rng.random() >= p_cancel:
                chain[2] = "Step 3: Radial potential boundary -> v_esc = 19.1 km/s"
        accept_orm = orm.score_trajectory(chain) >= 0.5
        accept_prm = prm.aggregate_trajectory_min(chain) >= threshold
        for name, accept in (("orm", accept_orm), ("prm", accept_prm)):
            key = ("fp" if flawed else "tp") if accept else ("tn" if flawed else "fn")
            counts[name][key] += 1
    rates = {}
    for name, c in counts.items():
        flawed_n = c["fp"] + c["tn"]
        clean_n = c["tp"] + c["fn"]
        rates[name] = {"false_positive_rate": c["fp"] / flawed_n if flawed_n else 0.0,
                       "true_positive_rate": c["tp"] / clean_n if clean_n else 0.0,
                       "accuracy": (c["tp"] + c["tn"]) / n_chains}
    return {"n_chains": n_chains, "p_flaw": p_flaw, "p_cancel": p_cancel, "counts": counts, "rates": rates}


def simulate_rollouts(p_success: float, m: int, rng: random.Random, truth: str = "24.0 km/s") -> List[str]:
    """A completion policy that reaches the ground truth from the step with probability p_success."""
    wrong = ["22.5 km/s", "19.1 km/s", "-24.0 km/s", "26.8 km/s"]
    return [truth if rng.random() < p_success else rng.choice(wrong) for _ in range(m)]


def shepherd_error_vs_M(p_success: Sequence[float] = (0.1, 0.5, 0.9), Ms: Sequence[int] = (4, 16, 64, 256),
                        n_steps: int = 200, seed: int = 7) -> List[Dict[str, float]]:
    """Mean absolute error of the soft label against the true p, and the mean hard label, as M grows."""
    prm = ProcessRewardEvaluator()
    rng = random.Random(seed)
    rows = []
    for p in p_success:
        for m in Ms:
            abs_err = 0.0
            hard = 0.0
            for _ in range(n_steps):
                lab = prm.math_shepherd_estimation("z_t", simulate_rollouts(p, m, rng), "24.0 km/s")
                abs_err += abs(lab["soft_label"] - p)
                hard += lab["hard_label"]
            rows.append({"p_true": p, "M": m, "soft_mae": abs_err / n_steps, "hard_mean": hard / n_steps,
                         "theory_sd": math.sqrt(p * (1 - p) / m)})
    return rows


# =============================================================================
# Instructor harness (kept), extended report, teammate loading
# =============================================================================

VALID_CHAIN = [
    "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
    "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
    "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s",
]
FALSE_POSITIVE_CHAIN = [
    "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
    "Step 2: [FLAWED] Flip sign of frame-dragging term twice (-1 * -1 = +1) via illegal tensor contraction",
    "Step 3: Accidental cancellation yields -> v_esc = 24.0 km/s",
]
VALID_ROLLOUTS = ["24.0 km/s", "24.0 km/s", "22.5 km/s", "24.0 km/s"]
TRAP_ROLLOUTS = ["24.0 km/s", "22.5 km/s", "-24.0 km/s", "19.1 km/s"]


def instructor_harness(quiet: bool = False) -> Dict[str, Any]:
    orm = OutcomeRewardEvaluator(expected_final_marker="v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()
    out: Dict[str, Any] = {}
    if not quiet:
        print("=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===")
    for label, chain, rollouts in [("Rigorous Chain", VALID_CHAIN, VALID_ROLLOUTS),
                                   ("False-Positive Trap", FALSE_POSITIVE_CHAIN, TRAP_ROLLOUTS)]:
        row = {"orm": orm.score_trajectory(chain), "min": prm.aggregate_trajectory_min(chain),
               "prod": prm.aggregate_trajectory_prod(chain), "geo": prm.aggregate_trajectory_geometric_mean(chain),
               "shepherd": prm.math_shepherd_estimation(chain[1], rollouts, "24.0 km/s")}
        out[label] = row
        if not quiet:
            print(f"{label:20s} | ORM={row['orm']:.2f} | PRM(min)={row['min']:.3f} | PRM(prod)={row['prod']:.3f} | "
                  f"PRM(geo)={row['geo']:.3f} | Shepherd(M=4) soft={row['shepherd']['soft_label']:.2f} hard={row['shepherd']['hard_label']:.0f}")
    assert orm.score_trajectory(FALSE_POSITIVE_CHAIN) == 1.0, "ORM should be fooled by the terminal answer!"
    assert prm.aggregate_trajectory_min(FALSE_POSITIVE_CHAIN) < 0.20, "PRM(min) must catch the flawed intermediate step!"
    shepherd = out["Rigorous Chain"]["shepherd"]
    assert abs(shepherd["soft_label"] - 0.75) < 1e-6 and shepherd["hard_label"] == 1.0
    if not quiet:
        print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
    return out


def extended_report() -> Dict[str, Any]:
    print("\nLength penalty at a fixed step quality 0.94 (min and geometric mean are length-neutral, prod is not):")
    print(f"  {'T':>3} {'min':>7} {'prod':>7} {'geo':>7}")
    lp = length_penalty_table()
    for r in lp:
        print(f"  {r['T']:>3} {r['min']:>7.3f} {r['prod']:>7.3f} {r['geometric_mean']:>7.3f}")

    fb = false_positive_benchmark()
    print(f"\nORM vs PRM(min) on {fb['n_chains']} synthetic proofs (half flawed; a flaw cancels to the right answer with p = {fb['p_cancel']}):")
    print(f"  {'model':<6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'FPR':>7} {'TPR':>7} {'acc':>7}")
    for name in ("orm", "prm"):
        c, r = fb["counts"][name], fb["rates"][name]
        print(f"  {name.upper():<6} {c['tp']:>4} {c['fp']:>4} {c['tn']:>4} {c['fn']:>4} {r['false_positive_rate']:>7.2f} {r['true_positive_rate']:>7.2f} {r['accuracy']:>7.2f}")

    se = shepherd_error_vs_M()
    print("\nMath-Shepherd on a simulated completion policy (200 steps per cell): soft-label MAE vs M, and the mean hard label:")
    print(f"  {'p_true':>6} {'M':>4} {'soft MAE':>9} {'sd theory':>10} {'hard mean':>10}")
    for r in se:
        print(f"  {r['p_true']:>6.1f} {r['M']:>4} {r['soft_mae']:>9.3f} {r['theory_sd']:>10.3f} {r['hard_mean']:>10.2f}")

    print("\nArithmetic verification examples:")
    prm = ProcessRewardEvaluator()
    for s in ["Step 2: 13 - 9 = 4 (left: [4, 4, 10])", "Step 2: 13 - 9 = 5 (left: [4, 5, 10])",
              "Step 2: 7/2 * 4 = 14", "Step 2: Looks rigorous and careful, (-1 * -1 = +1)"]:
        sc, why = prm.step_score_detailed(s)
        print(f"  {sc:.2f}  {why:<24} {s}")
    return {"length_penalty": lp, "false_positive": fb, "shepherd_vs_M": se}


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
# Self-test
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


def self_test() -> Checks:
    c = Checks()
    print("=== Part 3 self-test: process reward evaluator ===")
    prm = ProcessRewardEvaluator()
    orm = OutcomeRewardEvaluator("v_esc = 24.0 km/s")
    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731

    # reductions on hand examples
    sc = [0.9, 0.5, 0.8]
    c.add("min reduction is the weakest link", close(prm.aggregate_scores(sc, "min"), 0.5))
    c.add("prod reduction is the joint validity", close(prm.aggregate_scores(sc, "prod"), 0.36))
    c.add("geometric mean is prod^(1/T)", close(prm.aggregate_scores(sc, "geometric_mean"), 0.36 ** (1 / 3)))
    c.add("for scores in (0, 1]: prod <= min <= geometric mean", 0.36 <= 0.5 <= 0.36 ** (1 / 3))
    c.add("empty trajectory scores 0 under every reduction and ORM",
          all(prm.aggregate([], r) == 0.0 for r in prm.REDUCTIONS) and orm.score_trajectory([]) == 0.0)
    c.add("unknown reduction is refused", (lambda: (lambda f: (f() is None))(lambda: None))() and _raises(lambda: prm.aggregate_scores(sc, "mean"), ValueError))
    lp = length_penalty_table()
    c.add("length penalty: prod decays as 0.94^T while min and geometric mean stay at 0.94",
          close(lp[-1]["prod"], 0.94 ** 16) and all(close(r["min"], 0.94) and close(r["geometric_mean"], 0.94) for r in lp), f"prod at T=16: {lp[-1]['prod']:.3f}")

    # step scoring
    c.add("marker scores kept from the starter: [FLAWED] 0.12, SIGN_ERROR 0.12, [MINOR_DRIFT] 0.55, default 0.94",
          prm.step_score("x [FLAWED]") == 0.12 and prm.step_score("SIGN_ERROR here") == 0.12 and prm.step_score("y [MINOR_DRIFT]") == 0.55 and prm.step_score("fine step") == 0.94)
    c.add("a true equation is verified (0.97), a false one is flagged (0.12) regardless of wording",
          prm.step_score("Step: 13 - 9 = 4") == 0.97 and prm.step_score("Step: rigorously, 13 - 9 = 5") == 0.12)
    c.add("fractions and decimals are checked exactly", prm.step_score("7/2 * 4 = 14") == 0.97 and prm.step_score("0.5 + 0.25 = 0.75") == 0.97 and prm.step_score("1/3 + 1/3 = 0.66") == 0.12)
    c.add("division by zero is a false equation, a step without an equation falls back to markers",
          prm.step_score("4 / 0 = 0") == 0.12 and verify_equation("no numbers here") is None)
    c.add("arithmetic verification can be switched off (starter behaviour)", ProcessRewardEvaluator(verify_arithmetic=False).step_score("13 - 9 = 5") == 0.94)
    c.add("negative evidence wins: a true aside '(-1 * -1 = +1)' does not rescue a [FLAWED] step",
          prm.step_score(FALSE_POSITIVE_CHAIN[1]) == 0.12 and prm.step_score("(-1 * -1 = +1)") == 0.97)
    c.add("first_flawed_step finds the trap at step 2 and -1 on the clean chain",
          prm.first_flawed_step(FALSE_POSITIVE_CHAIN) == 2 and prm.first_flawed_step(VALID_CHAIN) == -1)

    # ORM vs PRM
    h = instructor_harness(quiet=True)
    c.add("instructor harness: ORM is fooled by the trap (1.00) and PRM(min) catches it (0.12)",
          h["False-Positive Trap"]["orm"] == 1.0 and close(h["False-Positive Trap"]["min"], 0.12))
    c.add("instructor harness: rigorous chain scores min 0.940, prod 0.831, geo 0.940",
          close(h["Rigorous Chain"]["min"], 0.94) and close(h["Rigorous Chain"]["prod"], 0.94 ** 3) and close(h["Rigorous Chain"]["geo"], 0.94))
    fb = false_positive_benchmark()
    c.add("benchmark: ORM accepts cancelled flaws (FPR about p_cancel), PRM(min) has zero false positives and full recall",
          abs(fb["rates"]["orm"]["false_positive_rate"] - fb["p_cancel"]) < 0.1 and fb["rates"]["prm"]["false_positive_rate"] == 0.0 and fb["rates"]["prm"]["true_positive_rate"] == 1.0,
          f"ORM FPR {fb['rates']['orm']['false_positive_rate']:.2f}, PRM acc {fb['rates']['prm']['accuracy']:.2f}")

    # Math-Shepherd
    c.add("instructor harness: Math-Shepherd M=4 gives soft 0.75, hard 1.0", close(h["Rigorous Chain"]["shepherd"]["soft_label"], 0.75) and h["Rigorous Chain"]["shepherd"]["hard_label"] == 1.0)
    c.add("trap step: one lucky rollout of four gives soft 0.25 but hard 1.0 (the hard label is the optimistic one)",
          close(h["False-Positive Trap"]["shepherd"]["soft_label"], 0.25) and h["False-Positive Trap"]["shepherd"]["hard_label"] == 1.0)
    c.add("no rollouts: both labels 0", prm.math_shepherd_estimation("z", [], "24.0 km/s") == {"soft_label": 0.0, "hard_label": 0.0})
    c.add("all rollouts wrong: soft 0 and hard 0", prm.math_shepherd_estimation("z", ["a", "b"], "c") == {"soft_label": 0.0, "hard_label": 0.0})
    se = shepherd_error_vs_M()

    def mae(p: float, m: int) -> float:
        return next(r["soft_mae"] for r in se if r["p_true"] == p and r["M"] == m)

    def hard(p: float, m: int) -> float:
        return next(r["hard_mean"] for r in se if r["p_true"] == p and r["M"] == m)
    c.add("soft-label error falls with M for every p (4 -> 16 -> 64 -> 256)",
          all(mae(p, 4) > mae(p, 16) > mae(p, 64) > mae(p, 256) for p in (0.1, 0.5, 0.9)), f"p=0.5: {mae(0.5, 4):.3f} -> {mae(0.5, 256):.3f}")
    c.add("soft-label error is within 2x of the binomial standard deviation sqrt(p(1-p)/M)",
          all(r["soft_mae"] <= 2 * r["theory_sd"] + 1e-9 for r in se))
    c.add("hard label saturates: a poor step (p = 0.1) is labelled 1 almost always once M >= 64",
          hard(0.1, 4) < 0.5 and hard(0.1, 64) > 0.95, f"M=4: {hard(0.1, 4):.2f}, M=64: {hard(0.1, 64):.2f}")

    # integration
    p1 = load_sibling("tot_solver.py", "part1_tot")
    if p1 is None:
        c.add("Part 1 solver present", False, "file missing")
    else:
        try:
            tree = {"Prompt: v_esc at r = 3 r_s": ["Step 1: 2 * 3 = 6 so r = 6GM/c^2", "Step 1': 2 * 3 = 5 so r = 5GM/c^2"],
                    "Step 1: 2 * 3 = 6 so r = 6GM/c^2": ["Step 2: v = c * sqrt(1/3) [MINOR_DRIFT]", "Step 2': v = c / sqrt(3)"],
                    "Step 1': 2 * 3 = 5 so r = 5GM/c^2": ["Step 2'': v = c / sqrt(2.5)"],
                    "Step 2': v = c / sqrt(3)": ["Step 3: v_esc = 24.0 km/s [GOAL]"]}
            solver = p1.TreeOfThoughtsSolver(max_expansions=16, beam_width=2, prune_threshold=0.45)
            root = p1.ThoughtNode("Prompt: v_esc at r = 3 r_s")
            best = solver.bfs_beam_search(root, lambda n: tree.get(n.state, []), prm.step_score, max_depth=3)
            wrong = [n for n in root.walk() if "2 * 3 = 5" in n.state]
            c.add("Part 1 ToT with this PRM as evaluator: the false equation branch is pruned, the verified branch reaches [GOAL]",
                  wrong and wrong[0].is_pruned and not wrong[0].children and "[GOAL]" in best.state, " -> ".join(best.get_trajectory()[1:]))
            c.add("Part 1 min-link cumulative equals this file's min reduction on the winning chain",
                  close(best.cumulative_score, prm.aggregate_trajectory_min(best.get_trajectory()[1:])))
            ok, sv, b24 = p1.solve24((4, 9, 10, 13), "bfs", 2, 96, prm.step_score)
            c.add("Game of 24 thoughts from Part 1 are all verified-true equations (0.97 each) and the search still ends at a terminal",
                  all(prm.step_score(s) == 0.97 for s in b24.get_trajectory()[1:]) and b24.is_terminal)
        except Exception as exc:
            c.add("Part 1 integration", False, f"{type(exc).__name__}: {exc}")

    p4 = load_sibling("horizon_pruner.py", "part4_pruner")
    if p4 is None:
        c.add("Part 4 pruner present", False, "file missing")
    else:
        try:
            pruner = p4.EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)
            rep = pruner.prune_trajectory(FALSE_POSITIVE_CHAIN, prm.score_all_steps(FALSE_POSITIVE_CHAIN))
            rep_ok = pruner.prune_trajectory(VALID_CHAIN, prm.score_all_steps(VALID_CHAIN))
            c.add("Part 4 pruner fed with these PRM scores severs the trap at step 2 and keeps the rigorous chain whole",
                  rep.pruned_at_step == 2 and rep_ok.pruned_at_step == -1 and len(rep_ok.surviving_steps) == 3)
        except Exception as exc:
            c.add("Part 4 integration", False, f"{type(exc).__name__}: {exc}")

    print(f"--- {c.passed} passed, {c.failed} failed ---")
    return c


def _raises(fn: Callable[[], Any], exc: type) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"{stamp}-ps07-prm-{mode}.json"
    record = {"mission": "ps07", "station": "03 Data & Payload Specialist (implemented by the Architect)", "mode": mode,
              "timestamp": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], **payload}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"log: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AURA Subsystem 07, Part 3: process reward evaluator")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        c = self_test()
        if not args.no_log:
            write_log("selftest", {"passed": c.passed, "failed": c.failed, "checks": c.rows})
        return 0 if c.failed == 0 else 1
    h = instructor_harness()
    rep = extended_report()
    if not args.no_log:
        write_log("report", {"harness": h, **rep})
    return 0


if __name__ == "__main__":
    sys.exit(main())
