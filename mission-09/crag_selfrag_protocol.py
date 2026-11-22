"""
USSE Noosphere — Mission 09 (SEC-09-LMC: Large Magellanic Cloud / Tarantula Nebula)
Station Role 2: Communications & Protocol Officer
Deliverable: crag_selfrag_protocol.py
"""

from dataclasses import dataclass
from typing import Dict, List, Literal
import re

CragAction = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
SupportLevel = Literal["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "NO_SUPPORT"]


@dataclass
class CragEvaluationResult:
    query: str
    action: CragAction
    max_confidence: float
    retained_strips: List[str]
    rewritten_relay_query: str


@dataclass
class SelfRagCandidateScore:
    candidate_id: str
    response_text: str
    is_relevant: bool
    support_level: SupportLevel
    utility_score: int  # 1 to 5
    composite_critique_score: float


class CragSelfRagProtocol:
    """
    Corrective RAG (CRAG) Evaluator, Decompose-then-Recompose Strip Filter,
    and Self-RAG Segment Critique Scorer.
    """

    STOPWORDS = {
        "what", "is", "the", "of", "in", "and", "how", "does", "to", "a", "an",
        "for", "with", "on", "at", "by", "from", "compare", "determine"
    }

    def __init__(
        self,
        theta_high: float = 0.65,
        theta_low: float = 0.32,
        strip_threshold: float = 0.35,
    ):
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.strip_threshold = strip_threshold

    def _tokenize_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9\-]+", text.lower())
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def score_relevance(self, query: str, text_segment: str) -> float:
        """Lightweight lexical + entity overlap evaluator E_psi(q, segment) in [0, 1]."""
        q_kws = set(self._tokenize_keywords(query))
        s_kws = set(self._tokenize_keywords(text_segment))
        if not q_kws or not s_kws:
            return 0.0
        overlap = len(q_kws.intersection(s_kws)) / len(q_kws)
        return round(min(1.0, overlap * 1.25), 3)

    def decompose_into_strips(self, document: str) -> List[str]:
        """Splits a document into atomic 1-sentence knowledge strips."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", document.strip())
        return [s.strip() for s in raw_sentences if len(s.strip()) > 12]

    def rewrite_for_deep_space_relay(self, query: str) -> str:
        """Reformulates query into compact high-recall keywords for external LMC Relay search."""
        kws = self._tokenize_keywords(query)
        top_kws = " ".join(kws[:8])
        return f"SEC-09-LMC ASTRO-RELAY SEARCH: {top_kws} 2026 telemetry"

    def evaluate_and_filter(self, query: str, documents: List[str]) -> CragEvaluationResult:
        """
        Executes CRAG three-way branch (CORRECT / AMBIGUOUS / INCORRECT)
        and Decompose-then-Recompose strip pruning.
        """
        doc_scores = [self.score_relevance(query, doc) for doc in documents]
        max_conf = max(doc_scores) if doc_scores else 0.0

        retained_strips: List[str] = []
        for doc in documents:
            for strip in self.decompose_into_strips(doc):
                if self.score_relevance(query, strip) >= self.strip_threshold:
                    retained_strips.append(strip)

        if max_conf >= self.theta_high:
            action: CragAction = "CORRECT"
            relay_query = ""
        elif max_conf < self.theta_low:
            action = "INCORRECT"
            retained_strips = []  # Discard noisy local chunks completely
            relay_query = self.rewrite_for_deep_space_relay(query)
        else:
            action = "AMBIGUOUS"
            relay_query = self.rewrite_for_deep_space_relay(query)

        return CragEvaluationResult(
            query=query,
            action=action,
            max_confidence=max_conf,
            retained_strips=retained_strips,
            rewritten_relay_query=relay_query,
        )

    def score_self_rag_candidates(
        self,
        query: str,
        evidence_strips: List[str],
        candidates: List[Dict[str, str]],
        w_rel: float = 1.0,
        w_sup: float = 2.0,
        w_use: float = 0.5,
    ) -> List[SelfRagCandidateScore]:
        """
        Scores candidate generation segments using Self-RAG reflection tokens:
        [ISREL], [ISSUP], and [ISUSE], returning candidates sorted best-first.
        """
        combined_evidence = " ".join(evidence_strips)
        ev_kws = set(self._tokenize_keywords(combined_evidence))
        scored: List[SelfRagCandidateScore] = []

        for cand in candidates:
            cid = cand["id"]
            text = cand["text"]
            cand_kws = set(self._tokenize_keywords(text))

            is_rel = self.score_relevance(query, text) >= 0.30
            if not ev_kws or not cand_kws:
                entailment_ratio = 0.0
            else:
                entailment_ratio = len(cand_kws.intersection(ev_kws)) / len(cand_kws)

            # Check for ungrounded numbers not present in evidence
            cand_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
            ev_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", combined_evidence))
            unverified_nums = cand_nums - ev_nums

            if entailment_ratio >= 0.55 and len(unverified_nums) == 0:
                support: SupportLevel = "FULLY_SUPPORTED"
                sup_val = 1.0
                utility = 5 if is_rel else 3
            elif entailment_ratio >= 0.30:
                support = "PARTIALLY_SUPPORTED"
                sup_val = 0.3
                utility = 3
            else:
                support = "NO_SUPPORT"
                sup_val = -1.0
                utility = 1

            rel_val = 1.0 if is_rel else -0.5
            use_val = (utility - 3.0) / 2.0  # Normalized in [-1, 1]

            composite = round(w_rel * rel_val + w_sup * sup_val + w_use * use_val, 3)
            scored.append(
                SelfRagCandidateScore(
                    candidate_id=cid,
                    response_text=text,
                    is_relevant=is_rel,
                    support_level=support,
                    utility_score=utility,
                    composite_critique_score=composite,
                )
            )

        return sorted(scored, key=lambda x: x.composite_critique_score, reverse=True)


if __name__ == "__main__":
    protocol = CragSelfRagProtocol()
    q = "What is the bolometric luminosity and mass of Wolf-Rayet star R136a1 in 30 Doradus?"
    docs = [
        "R136a1 in the Tarantula Nebula (30 Doradus) is a Wolf-Rayet star with a bolometric luminosity of 4.7 million solar luminosities and a mass of approximately 196 solar masses. The cafeteria on Deck 3 serves hydroponic algae soup on Tuesdays.",
    ]
    res = protocol.evaluate_and_filter(q, docs)
    print(f"[CRAG Action] {res.action} (max_conf={res.max_confidence}) | Retained Strips: {len(res.retained_strips)}")
    assert res.action == "CORRECT"
    assert len(res.retained_strips) == 1  # Irrelevant cafeteria strip pruned!

    cands = [
        {
            "id": "cand_hallucinated",
            "text": "R136a1 in 30 Doradus has a bolometric luminosity of 99.9 million solar luminosities and mass of 850 solar masses.",
        },
        {
            "id": "cand_grounded",
            "text": "Wolf-Rayet star R136a1 in 30 Doradus has a bolometric luminosity of 4.7 million solar luminosities and a mass of 196 solar masses.",
        },
    ]
    ranked = protocol.score_self_rag_candidates(q, res.retained_strips, cands)
    print(f"[Self-RAG Winner] {ranked[0].candidate_id} | Support={ranked[0].support_level} | Score={ranked[0].composite_critique_score}")
    assert ranked[0].candidate_id == "cand_grounded"
    assert ranked[0].support_level == "FULLY_SUPPORTED"
    print("[PASS] Part 2 CRAG & Self-RAG Protocol verified successfully!")
