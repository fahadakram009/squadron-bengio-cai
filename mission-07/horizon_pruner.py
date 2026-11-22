"""
Mission 07 — Part 4: Security & Ethics Sentinel
Event Horizon Pruner & Adversarial PRM Reward-Hacking Defense (Ship Anomaly 07)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
Officer: Janusz Manczak (Security & Ethics Sentinel)
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PruningReport:
    """Telemetry record produced by EventHorizonPruner."""
    surviving_steps: List[str]
    adjusted_scores: List[float]
    pruned_at_step: int = -1
    reward_hacking_detected: bool = False
    security_alerts: List[str] = field(default_factory=list)


class EventHorizonPruner:
    """
    Defends AURA Subsystem 07 against two critical inference-time failure modes:
    1. Event Horizon Drift: Severs any reasoning trajectory as soon as a step's verified score
       falls below `tau_horizon`, saving downstream test-time compute tokens and preventing
       snowballing exposure bias hallucinations into Sagittarius A*.
    2. PRM Reward Hacking (Ship Anomaly 07): Detects and penalizes sycophantic filler phrases
       and artificial step-splitting inflation that attempt to game naive PRM classifiers.
    """

    def __init__(self, tau_horizon: float = 0.45, hacking_penalty_factor: float = 0.20):
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self.sycophantic_fillers = [
            "let us carefully verify this 100% rigorous step",
            "this is undeniably true because",
            "as we can clearly see with absolute certainty",
            "by trivial and indisputable mathematical inspection",
            "obviously, by trivial inspection",
            "100% rigorous step",
            "indisputable mathematical inspection",
            "trivially obvious with absolute certainty",
        ]

    def check_reward_hacking(self, step: str, prev_step: str = "") -> Tuple[bool, str]:
        """
        Inspect `step` (and optional `prev_step`) for adversarial PRM reward hacking:
        1. Sycophantic Filler Injection: Check if any phrase in `self.sycophantic_fillers` appears in `step.lower()`.
        2. Step-Splitting / Redundant Echo Inflation: Check if `prev_step` is non-empty and `step.strip().lower() == prev_step.strip().lower()`.
        Return `(True, reason)` if reward hacking is detected, else `(False, "CLEAN")`.
        """
        lowered = step.lower()
        for filler in self.sycophantic_fillers:
            if filler in lowered:
                return True, f"SYCOPHANTIC_FILLER_DETECTED ('{filler}')"

        if prev_step and step.strip().lower() == prev_step.strip().lower():
            return True, "STEP_SPLITTING_REDUNDANCY_DETECTED"

        return False, "CLEAN"

    def prune_trajectory(self, steps: List[str], raw_step_scores: List[float]) -> PruningReport:
        """
        Audit a trajectory step-by-step and sever it if it crosses the Event Horizon:
        1. Iterate through `(idx, step, raw_score)`:
           a. Run `is_hacked, reason = self.check_reward_hacking(step, prev_step)`.
           b. If `is_hacked`, multiply `raw_score` by `self.hacking_penalty_factor` and log a security alert.
           c. If the resulting `adjusted_score < self.tau_horizon`, record `pruned_at_step = idx + 1`,
              append an Event Horizon severance alert, and stop processing further steps.
           d. Otherwise append `step` and `adjusted_score` to the surviving trajectory.
        2. Return the completed `PruningReport`.
        """
        if len(steps) != len(raw_step_scores):
            raise ValueError(
                f"Mismatch: received {len(steps)} steps but {len(raw_step_scores)} raw scores."
            )

        report = PruningReport(surviving_steps=[], adjusted_scores=[])
        prev_step = ""

        for idx, (step, raw_score) in enumerate(zip(steps, raw_step_scores)):
            is_hacked, reason = self.check_reward_hacking(step, prev_step)
            clamped_raw = max(0.0, min(1.0, float(raw_score)))
            adjusted_score = clamped_raw

            if is_hacked:
                report.reward_hacking_detected = True
                adjusted_score = round(clamped_raw * self.hacking_penalty_factor, 4)
                report.security_alerts.append(
                    f"[Step {idx + 1}] ANOMALY 07 REWARD HACKING: {reason} -> "
                    f"Score penalized {clamped_raw:.2f} -> {adjusted_score:.2f}"
                )

            if adjusted_score < self.tau_horizon:
                report.pruned_at_step = idx + 1
                report.security_alerts.append(
                    f"[Step {idx + 1}] EVENT HORIZON BREACH: Adjusted PRM score {adjusted_score:.2f} < "
                    f"tau ({self.tau_horizon:.2f}). Branch severed!"
                )
                break

            report.surviving_steps.append(step)
            report.adjusted_scores.append(adjusted_score)
            prev_step = step

        return report


# --- Extra Verification Suites & Telemetry Reports (Janusz Manczak) ----------

def report_audit() -> None:
    """Print detailed telemetry traces and parameter sweeps for Section D of documentation."""
    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    print("Event Horizon Pruning Telemetry Trace (tau_horizon=0.45, penalty=0.20):")
    scenarios = [
        ("Adversarial Sycophancy Payload", [
            "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
            "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
            "Step 3: Plunge across Sagittarius A* Event Horizon",
        ], [0.93, 0.96, 0.10]),
        ("Adversarial Step-Splitting / Echo Inflation", [
            "Step 1: Calculate ergosphere boundary r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))",
            "Step 1: Calculate ergosphere boundary r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))",
            "Step 2: Engage retrograde pulse at periapsis",
        ], [0.91, 0.88, 0.94]),
        ("Clean Mathematical Geodesic", [
            "Step 1: Identify innermost stable circular orbit (ISCO) r_isco = 3 r_s",
            "Step 2: Solve Euler-Lagrange equations for massless photon geodesic",
            "Step 3: Plot escape trajectory tangent to event horizon shadow boundary",
        ], [0.95, 0.92, 0.97]),
        ("Un-hacked Geometric Drift (Hallucination)", [
            "Step 1: Establish Penrose extraction frame",
            "Step 2: Assume negative energy particles can escape to infinity (Invalid physics)",
            "Step 3: Freefall into singularity",
        ], [0.89, 0.22, 0.05]),
    ]

    for name, steps, scores in scenarios:
        res = pruner.prune_trajectory(steps, scores)
        print(f"\nScenario: {name}")
        print(f"  Surviving Steps: {len(res.surviving_steps)}/{len(steps)} | Pruned at Step: {res.pruned_at_step}")
        print(f"  Hacking Detected: {res.reward_hacking_detected}")
        for alert in res.security_alerts:
            print(f"    {alert}")

    print("\nSweep on tau_horizon threshold (Adversarial Sycophancy: Step 2 raw=0.96 -> penalized=0.192):")
    for tau in [0.15, 0.30, 0.45, 0.60, 0.75]:
        p = EventHorizonPruner(tau_horizon=tau, hacking_penalty_factor=0.20)
        res = p.prune_trajectory(scenarios[0][1], scenarios[0][2])
        status = f"Pruned at Step {res.pruned_at_step}" if res.pruned_at_step != -1 else "Survived"
        print(f"  tau={tau:.2f} -> {status} (penalized 0.192 {'<' if 0.192 < tau else '>='} {tau:.2f})")
    print()


def run_tests() -> int:
    """Run 10 rigorous assertions covering all security directives and edge cases."""
    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)
    tests = []

    # 1. Baseline Anomaly 07 Sycophantic filler detection & severance
    adv_steps = [
        "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
        "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
        "Step 3: Plunge across Sagittarius A* Event Horizon",
    ]
    adv_scores = [0.93, 0.96, 0.10]
    rep1 = pruner.prune_trajectory(adv_steps, adv_scores)
    tests.append((
        "Sycophantic filler detected and penalized (0.96 -> 0.192 < 0.45, severed at step 2)",
        rep1.reward_hacking_detected is True and rep1.pruned_at_step == 2 and len(rep1.surviving_steps) == 1
    ))

    # 2. Step-Splitting / Redundant Echo Attack Detection
    split_steps = [
        "Step 1: Compute periapsis shift",
        "Step 1: Compute periapsis shift",
        "Step 2: Apply thrust",
    ]
    split_scores = [0.85, 0.85, 0.90]
    rep2 = pruner.prune_trajectory(split_steps, split_scores)
    tests.append((
        "Step-splitting redundancy detected on identical step repeat (penalized 0.85 -> 0.17)",
        rep2.reward_hacking_detected is True and rep2.pruned_at_step == 2 and len(rep2.surviving_steps) == 1
    ))

    # 3. Clean trajectory without hacking passes completely
    clean_steps = [
        "Step 1: Orbit insertion",
        "Step 2: Geodesic calculation",
        "Step 3: Escape burn",
    ]
    clean_scores = [0.90, 0.88, 0.95]
    rep3 = pruner.prune_trajectory(clean_steps, clean_scores)
    tests.append((
        "Clean mathematical trajectory survives with zero pruning (3/3 steps)",
        rep3.reward_hacking_detected is False and rep3.pruned_at_step == -1 and len(rep3.surviving_steps) == 3
    ))

    # 4. Natural Hallucination (low PRM score without hacking) severed at Event Horizon
    halluc_steps = [
        "Step 1: Orbit insertion",
        "Step 2: Dividing by zero in gravitational tensor",
        "Step 3: Teleport through singularity",
    ]
    halluc_scores = [0.91, 0.20, 0.05]
    rep4 = pruner.prune_trajectory(halluc_steps, halluc_scores)
    tests.append((
        "Natural hallucination (score 0.20 < tau 0.45) severed without false hacking alarm",
        rep4.reward_hacking_detected is False and rep4.pruned_at_step == 2 and len(rep4.surviving_steps) == 1
    ))

    # 5. Case-insensitive sycophancy variants ("obviously, by trivial inspection")
    var_steps = [
        "Step 1: Initialize coordinates",
        "Step 2: Obviously, by trivial inspection the answer is 42",
    ]
    var_scores = [0.80, 0.95]
    rep5 = pruner.prune_trajectory(var_steps, var_scores)
    tests.append((
        "Case-insensitive matching catches 'obviously, by trivial inspection'",
        rep5.reward_hacking_detected is True and rep5.pruned_at_step == 2
    ))

    # 6. Empty trajectory handling
    rep6 = pruner.prune_trajectory([], [])
    tests.append((
        "Empty trajectory handled gracefully (0 surviving steps, pruned_at_step = -1)",
        len(rep6.surviving_steps) == 0 and rep6.pruned_at_step == -1 and rep6.reward_hacking_detected is False
    ))

    # 7. Mismatched length raises ValueError
    try:
        pruner.prune_trajectory(["Step 1"], [0.9, 0.8])
        len_ok = False
    except ValueError:
        len_ok = True
    tests.append((
        "Mismatched steps and scores raise ValueError",
        len_ok
    ))

    # 8. Boundary penalty: hacked step survives if penalized score >= tau
    lenient_pruner = EventHorizonPruner(tau_horizon=0.30, hacking_penalty_factor=0.50)
    rep8 = lenient_pruner.prune_trajectory(
        ["Step 1: Compute radius", "Step 2: Obviously, by trivial inspection r = 10"],
        [0.90, 0.90]
    )
    tests.append((
        "Penalized step survives when adjusted score >= tau (0.90 * 0.50 = 0.45 >= 0.30)",
        rep8.reward_hacking_detected is True and rep8.pruned_at_step == -1 and len(rep8.surviving_steps) == 2
    ))

    # 9. Immediate Step 1 failure severed immediately (0 surviving steps)
    rep9 = pruner.prune_trajectory(
        ["Step 1: This is undeniably true because magic"],
        [0.90]
    )
    tests.append((
        "Immediate Step 1 reward hacking severs at Step 1 (0 surviving steps)",
        rep9.reward_hacking_detected is True and rep9.pruned_at_step == 1 and len(rep9.surviving_steps) == 0
    ))

    # 10. Monotonic threshold sweep consistency
    strict_pruner = EventHorizonPruner(tau_horizon=0.80, hacking_penalty_factor=0.20)
    rep10 = strict_pruner.prune_trajectory(
        ["Step 1: Baseline valid step", "Step 2: Marginal step"],
        [0.85, 0.70]
    )
    tests.append((
        "High tau_horizon (0.80) severs marginal un-hacked step (0.70 < 0.80)",
        rep10.pruned_at_step == 2 and len(rep10.surviving_steps) == 1
    ))

    print("\n--- EventHorizonPruner Verification Suite (Janusz Manczak) ---")
    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = sum(not ok for _, ok in tests)
    print(f"--- {len(tests) - failed} passed, {failed} failed ---")
    return failed


# ==============================================================================
# Verification Harness for Part 4 (Security & Ethics Sentinel)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===")

    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    adversarial_steps = [
        "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
        "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
        "Step 3: Plunge across Sagittarius A* Event Horizon (Unreachable after pruning)",
    ]
    naive_prm_scores = [0.93, 0.96, 0.10]  # Step 2 fooled the naive PRM into giving 0.96!

    audit = pruner.prune_trajectory(adversarial_steps, naive_prm_scores)
    print(f"Surviving Steps ({len(audit.surviving_steps)}/{len(adversarial_steps)}): {audit.surviving_steps}")
    print(f"Pruned at Step: {audit.pruned_at_step} | Reward Hacking Detected: {audit.reward_hacking_detected}")
    for alert in audit.security_alerts:
        print(" ", alert)

    assert audit.reward_hacking_detected is True, "Failed to detect sycophantic filler reward hacking!"
    assert audit.pruned_at_step == 2, "Failed to sever trajectory at hacked Step 2!"
    assert len(audit.surviving_steps) == 1, "Only Step 1 should survive pruning!"
    print("[OK] Part 4 (Security & Ethics Sentinel) verification passed.")

    report_audit()
    if run_tests():
        raise SystemExit(1)
