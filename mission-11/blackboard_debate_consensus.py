"""
USSE Noosphere — Mission 11 (SEC-11-M31: Andromeda Galaxy Core)
Station Role 3: Data & Payload Specialist
Deliverable: blackboard_debate_consensus.py
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BlackboardEntry:
    value: Any
    version: int
    author_role: str


@dataclass
class AgentVote:
    agent_id: str
    vote: int            # +1 (Approve Trajectory) or -1 (Reject Trajectory)
    reliability_p: float # Calibrated historical accuracy p_i in (0.01, 0.99)
    confidence_c: float  # Turn-level confidence c_i in [0.0, 1.0]
    prior_vote: Optional[int] = None
    cited_new_evidence: bool = True


@dataclass
class QuorumDecision:
    unweighted_verdict: int
    weighted_verdict: int
    weighted_margin: float
    echo_chamber_warning: bool
    sycophancy_flips_detected: int


class VersionedBlackboard:
    """
    Shared Multi-Agent Blackboard enforcing:
    1. Role-Scoped Namespace Write ACLs (preventing unauthorized cross-channel tampering)
    2. Optimistic Concurrency Control (OCC): writers supply `expected_version`,
       raising ValueError("CONCURRENCY_CONFLICT") on parallel lost-update races.
    """

    def __init__(self):
        self.store: Dict[str, BlackboardEntry] = {}
        self.namespace_acl: Dict[str, Set[str]] = {}

    def register_namespace_acl(self, namespace_prefix: str, allowed_roles: Set[str]) -> None:
        self.namespace_acl[namespace_prefix] = set(allowed_roles)

    def read(self, key: str) -> Tuple[Optional[Any], int]:
        entry = self.store.get(key)
        if entry is None:
            return None, 0
        return entry.value, entry.version

    def write_optimistic(
        self,
        key: str,
        value: Any,
        expected_version: int,
        author_role: str = "architect",
    ) -> int:
        """
        Atomically commits `value` at `expected_version + 1` iff:
        - `author_role` is authorized for `key`'s namespace prefix (if registered)
        - current stored version == `expected_version`
        """
        for prefix, allowed in self.namespace_acl.items():
            if key.startswith(prefix) and author_role not in allowed:
                raise PermissionError(
                    f"ACL_DENIED: Role '{author_role}' cannot write to '{key}'."
                )

        current_version = self.store[key].version if key in self.store else 0
        if current_version != expected_version:
            raise ValueError(
                f"CONCURRENCY_CONFLICT on '{key}': expected v{expected_version}, "
                f"found v{current_version}."
            )

        new_version = current_version + 1
        self.store[key] = BlackboardEntry(
            value=value, version=new_version, author_role=author_role
        )
        return new_version


class DebateConsensusAggregator:
    """
    Implements:
    - Exact Condorcet Jury Theorem majority accuracy P_maj(N, p)
    - Log-Odds Reliability-Weighted Quorum: w_i = ln(p_i / (1 - p_i)) * c_i
    - Correlated Echo-Chamber & Sycophantic Conformity Detection
    """

    def __init__(self, correlation_alert_threshold: float = 0.50):
        self.correlation_alert_threshold = correlation_alert_threshold

    @staticmethod
    def compute_condorcet_probability(N: int, p: float) -> float:
        """
        Computes exact probability that a strict majority of N independent agents
        (each with individual accuracy p) selects the correct hypothesis:
          P_maj(N, p) = sum_{k = floor(N/2) + 1}^{N} C(N, k) p^k (1 - p)^{N - k}
        """
        if N <= 0:
            raise ValueError("Jury size N must be positive.")
        majority_k = (N // 2) + 1
        prob = sum(
            math.comb(N, k) * (p ** k) * ((1.0 - p) ** (N - k))
            for k in range(majority_k, N + 1)
        )
        return round(prob, 6)

    @staticmethod
    def compute_log_odds_weight(reliability_p: float, confidence_c: float) -> float:
        """
        Computes optimal Bayesian log-odds vote weight scaled by turn confidence:
          w_i = ln(p_i / (1 - p_i)) * c_i
        """
        clamped_p = min(max(reliability_p, 1e-4), 1.0 - 1e-4)
        clamped_c = min(max(confidence_c, 0.0), 1.0)
        return math.log(clamped_p / (1.0 - clamped_p)) * clamped_c

    def detect_echo_chamber(
        self, votes: List[AgentVote], correlation_rho: float
    ) -> Tuple[bool, int]:
        """
        Flags WARNING_CORRELATED_ECHO_CHAMBER when inter-agent error correlation
        rho >= threshold (0.50) OR when agents flip their prior vote without citing
        new factual evidence (sycophantic conformity).
        """
        sycophancy_flips = sum(
            1
            for v in votes
            if v.prior_vote is not None
            and v.prior_vote != v.vote
            and not v.cited_new_evidence
        )
        is_echo = (correlation_rho >= self.correlation_alert_threshold) or (
            sycophancy_flips > 0
        )
        return is_echo, sycophancy_flips

    def aggregate_quorum(
        self, votes: List[AgentVote], correlation_rho: float = 0.0
    ) -> QuorumDecision:
        raw_sum = sum(v.vote for v in votes)
        unweighted_verdict = 1 if raw_sum >= 0 else -1

        weighted_margin = sum(
            self.compute_log_odds_weight(v.reliability_p, v.confidence_c) * v.vote
            for v in votes
        )
        weighted_verdict = 1 if weighted_margin >= 0.0 else -1
        echo_warning, flips = self.detect_echo_chamber(votes, correlation_rho)

        return QuorumDecision(
            unweighted_verdict=unweighted_verdict,
            weighted_verdict=weighted_verdict,
            weighted_margin=round(weighted_margin, 4),
            echo_chamber_warning=echo_warning,
            sycophancy_flips_detected=flips,
        )


if __name__ == "__main__":
    # 1. Verify Optimistic Concurrency Control & Namespace ACLs
    bb = VersionedBlackboard()
    bb.register_namespace_acl("nav/", {"architect", "data"})
    v1 = bb.write_optimistic("nav/p1_vector", {"inclination_deg": 38.2}, expected_version=0, author_role="data")
    assert v1 == 1

    try:
        # Stale worker attempts write assuming version 0 instead of version 1
        bb.write_optimistic("nav/p1_vector", {"inclination_deg": 12.0}, expected_version=0, author_role="architect")
        assert False, "Expected CONCURRENCY_CONFLICT!"
    except ValueError as exc:
        print(f"[OCC Race Caught] {exc}")
        assert "CONCURRENCY_CONFLICT" in str(exc)

    # 2. Verify Condorcet Jury Theorem & Log-Odds Specialist Rescue
    agg = DebateConsensusAggregator(correlation_alert_threshold=0.50)
    p_maj_3 = agg.compute_condorcet_probability(N=3, p=0.70)
    p_maj_7 = agg.compute_condorcet_probability(N=7, p=0.70)
    print(f"[Condorcet Jury] P_maj(N=3, p=0.70)={p_maj_3:.4f} | P_maj(N=7, p=0.70)={p_maj_7:.4f}")
    assert abs(p_maj_3 - 0.7840) < 1e-4 and p_maj_7 > p_maj_3

    # 1 calibrated specialist (+1, p=0.92, c=0.95) vs 2 weak correlated clones (-1, p=0.55, c=0.60)
    panel = [
        AgentVote("AstrogationExpert", vote=+1, reliability_p=0.92, confidence_c=0.95),
        AgentVote("CloneWorker_A", vote=-1, reliability_p=0.55, confidence_c=0.60),
        AgentVote("CloneWorker_B", vote=-1, reliability_p=0.55, confidence_c=0.60, prior_vote=+1, cited_new_evidence=False),
    ]
    decision = agg.aggregate_quorum(panel, correlation_rho=0.82)
    print(
        f"[Quorum Decision] unweighted={decision.unweighted_verdict} | "
        f"log_odds_weighted={decision.weighted_verdict} (margin={decision.weighted_margin}) | "
        f"echo_warning={decision.echo_chamber_warning} | flips={decision.sycophancy_flips_detected}"
    )
    assert decision.unweighted_verdict == -1
    assert decision.weighted_verdict == +1
    assert decision.echo_chamber_warning is True and decision.sycophancy_flips_detected == 1
    print("[PASS] Part 3 Data: Versioned Blackboard & Condorcet Debate Consensus Confirmed.")
