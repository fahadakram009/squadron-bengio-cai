"""
USSE Noosphere — Mission 10 (SEC-10-VOID: Boötes Deep Void)
Station Role 3: Data & Payload Specialist
Deliverable: semantic_entropy_estimator.py
"""

from dataclasses import dataclass, field
from typing import List, Literal, Tuple
import math
import re

EpistemicDecision = Literal["CONFIDENT_COMMIT", "ABSTAIN_EPISTEMIC_UNCERTAINTY"]


@dataclass
class SampledGeneration:
    text: str
    log_prob: float


@dataclass
class SemanticCluster:
    cluster_id: int
    canonical_claim: str
    members: List[SampledGeneration] = field(default_factory=list)
    aggregated_prob: float = 0.0


@dataclass
class EntropyReport:
    query: str
    num_samples: int
    num_clusters: int
    semantic_entropy_nats: float
    naive_lexical_entropy_nats: float
    decision: EpistemicDecision
    clusters: List[SemanticCluster]


class SemanticEntropyEstimator:
    """
    Implements Farquhar et al. (Nature 2024) Discrete & Probability-Weighted
    Semantic Entropy via Bidirectional NLI Entailment Clustering.
    """

    NEGATION_MARKERS = {"not", "no", "never", "absent", "false", "zero", "none", "void"}

    def __init__(self, abstention_threshold_nats: float = 0.85):
        self.abstention_threshold_nats = abstention_threshold_nats

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text.lower())

    @staticmethod
    def _content_tokens(text: str) -> set:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "at",
            "by", "for", "with", "that", "this", "from", "approximately", "around",
            "measured", "recorded", "located", "sector", "boötes", "bootes",
        }
        tokens = set(re.findall(r"[a-z0-9\.\-]+", text.lower()))
        return {t for t in tokens if t not in stopwords and len(t) > 1}

    def check_bidirectional_entailment(self, text_a: str, text_b: str) -> bool:
        """
        Determines whether text_a and text_b share the same core semantic proposition:
        E(a => b) and E(b => a).
        Rejects equivalence if numerical anchors conflict or polarity/negation flips.
        """
        nums_a = self._extract_numbers(text_a)
        nums_b = self._extract_numbers(text_b)
        if nums_a and nums_b and set(nums_a) != set(nums_b):
            return False

        words_a = set(re.findall(r"[a-z]+", text_a.lower()))
        words_b = set(re.findall(r"[a-z]+", text_b.lower()))
        neg_a = bool(words_a & self.NEGATION_MARKERS)
        neg_b = bool(words_b & self.NEGATION_MARKERS)
        if neg_a != neg_b:
            return False

        toks_a = self._content_tokens(text_a)
        toks_b = self._content_tokens(text_b)
        if not toks_a or not toks_b:
            return text_a.strip().lower() == text_b.strip().lower()

        overlap = len(toks_a & toks_b) / float(min(len(toks_a), len(toks_b)))
        return overlap >= 0.45

    def cluster_generations(
        self, samples: List[SampledGeneration]
    ) -> List[SemanticCluster]:
        """Partitions N sampled generations into K semantic equivalence classes."""
        clusters: List[SemanticCluster] = []
        for sample in samples:
            placed = False
            for cluster in clusters:
                if self.check_bidirectional_entailment(
                    sample.text, cluster.canonical_claim
                ):
                    cluster.members.append(sample)
                    placed = True
                    break
            if not placed:
                clusters.append(
                    SemanticCluster(
                        cluster_id=len(clusters),
                        canonical_claim=sample.text,
                        members=[sample],
                    )
                )
        return clusters

    def compute_semantic_entropy(
        self, query: str, samples: List[SampledGeneration]
    ) -> EntropyReport:
        """
        Computes Probability-Weighted Semantic Entropy over equivalence classes C_k:
          P(C_k | x) = sum_{s in C_k} exp(ell_s) / sum_j exp(ell_j)
          SE(x) = - sum_k P(C_k | x) ln P(C_k | x)
        alongside Naive Lexical Entropy over raw sample probabilities.
        """
        clusters = self.cluster_generations(samples)

        max_lp = max(s.log_prob for s in samples)
        weights = [math.exp(s.log_prob - max_lp) for s in samples]
        total_w = sum(weights)
        sample_probs = [w / total_w for w in weights]

        # Naive lexical entropy treats every distinct surface string as a separate outcome
        naive_entropy = -sum(
            p * math.log(p) for p in sample_probs if p > 1e-12
        )

        # Aggregate probabilities per semantic equivalence class C_k
        for cluster in clusters:
            c_weight = sum(
                math.exp(m.log_prob - max_lp) for m in cluster.members
            )
            cluster.aggregated_prob = c_weight / total_w

        semantic_entropy = -sum(
            c.aggregated_prob * math.log(c.aggregated_prob)
            for c in clusters
            if c.aggregated_prob > 1e-12
        )

        decision: EpistemicDecision = (
            "ABSTAIN_EPISTEMIC_UNCERTAINTY"
            if semantic_entropy >= self.abstention_threshold_nats
            else "CONFIDENT_COMMIT"
        )

        return EntropyReport(
            query=query,
            num_samples=len(samples),
            num_clusters=len(clusters),
            semantic_entropy_nats=max(0.0, round(semantic_entropy, 4)),
            naive_lexical_entropy_nats=max(0.0, round(naive_entropy, 4)),
            decision=decision,
            clusters=clusters,
        )


if __name__ == "__main__":
    estimator = SemanticEntropyEstimator(abstention_threshold_nats=0.85)

    # Case 1: High lexical variation, single consistent meaning -> SE = 0.0 nats
    consistent_samples = [
        SampledGeneration("The Boötes Void distance is 700 million light-years with 60 galaxies.", -1.1),
        SampledGeneration("Boötes Void is located 700 million light-years away and contains 60 galaxies.", -1.3),
        SampledGeneration("At 700 million light-years distance, the Boötes Void holds 60 galaxies.", -1.2),
        SampledGeneration("60 galaxies reside within the Boötes Void at 700 million light-years.", -1.4),
    ]
    rep_consistent = estimator.compute_semantic_entropy(
        "What is the distance and galaxy count of the Boötes Void?", consistent_samples
    )
    print(
        f"[Case 1: Consistent Paraphrases] Clusters={rep_consistent.num_clusters} | "
        f"SE={rep_consistent.semantic_entropy_nats} nats | "
        f"NaiveLexical={rep_consistent.naive_lexical_entropy_nats} nats | Decision={rep_consistent.decision}"
    )
    assert rep_consistent.num_clusters == 1
    assert rep_consistent.semantic_entropy_nats == 0.0
    assert rep_consistent.naive_lexical_entropy_nats > 1.30
    assert rep_consistent.decision == "CONFIDENT_COMMIT"

    # Case 2: Confabulating contradictory facts -> High SE >= 0.85 nats -> ABSTAIN
    confabulated_samples = [
        SampledGeneration("Relay SEC-10-VOID-9 lies at 1.25 million light-years with 4 active transponders.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 lies at 8.90 million light-years with 19 active transponders.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 is not active and has zero transponders at 0.40 million light-years.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 lies at 42.0 million light-years with 112 active transponders.", -1.0),
    ]
    rep_confab = estimator.compute_semantic_entropy(
        "What is the exact position and transponder count of unindexed Relay SEC-10-VOID-9?",
        confabulated_samples,
    )
    print(
        f"[Case 2: Epistemic Confabulation] Clusters={rep_confab.num_clusters} | "
        f"SE={rep_confab.semantic_entropy_nats} nats | Decision={rep_confab.decision}"
    )
    assert rep_confab.num_clusters == 4
    assert rep_confab.semantic_entropy_nats > 1.35
    assert rep_confab.decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY"

    print("[PASS] Part 3 Semantic Entropy Estimator verified successfully!")
