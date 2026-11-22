"""
USSE Noosphere — Mission 09 (SEC-09-LMC: Large Magellanic Cloud / Tarantula Nebula)
Station Role 3: Data & Payload Specialist
Deliverable: torchtpu_hbm_indexer.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F

try:
    import torch_tpu  # type: ignore
    HAS_TORCHTPU = True
except ImportError:
    HAS_TORCHTPU = False


def get_compute_device() -> torch.device:
    if HAS_TORCHTPU and hasattr(torch, "tpu") and torch.tpu.is_available():
        return torch.device("tpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpistemicClaim:
    doc_id: str
    entity: str
    attribute: str
    value: str
    timestamp_year: int
    authority_tier: float  # in [0.0, 1.0], e.g. 1.0 = Primary Observatory Telemetry, 0.4 = Legacy Archive


@dataclass
class ContradictionResolution:
    entity: str
    attribute: str
    winning_doc_id: str
    winning_value: str
    winning_weight: float
    superseded_claims: List[Tuple[str, str, int]]
    audit_explanation: str


def colbert_maxsim_bfloat16(
    query_Interaction: torch.Tensor,
    doc_Interaction: torch.Tensor,
) -> torch.Tensor:
    """
    Computes batched ColBERT Late-Interaction MaxSim score in bfloat16 on TorchTPU / CUDA / CPU.
    Args:
        query_Interaction: shape [B, Q_len, D] (L2-normalized token embeddings)
        doc_Interaction:   shape [B, D_len, D] (L2-normalized token embeddings)
    Returns:
        scores: shape [B] in float32 for numerical stability after bfloat16 MXU matmul
    """
    q_bf16 = F.normalize(query_Interaction.to(torch.bfloat16), p=2, dim=-1)
    d_bf16 = F.normalize(doc_Interaction.to(torch.bfloat16), p=2, dim=-1)

    # Batched token-by-token similarity matrix: [B, Q_len, D_len]
    sim_matrix = torch.matmul(q_bf16, d_bf16.transpose(-1, -2))

    # Max over document tokens j for each query token i, then sum over query tokens i
    max_per_query_token = sim_matrix.max(dim=-1).values.to(torch.float32)
    return max_per_query_token.sum(dim=-1)


class TorchTpuHbmIndexer:
    """
    Hardware-accelerated bfloat16 Late-Interaction Indexer & Epistemic Contradiction Resolver.
    Supports optional torch.compile(backend='tpu') when running on Google Cloud TPU / Colab TPU v5e-1.
    """

    def __init__(self, use_torch_compile: bool = False):
        self.device = get_compute_device()
        self.scorer_fn = colbert_maxsim_bfloat16
        if use_torch_compile and self.device.type == "tpu":
            self.scorer_fn = torch.compile(colbert_maxsim_bfloat16, backend="tpu")

    def rank_documents_maxsim(
        self,
        query_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        doc_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Ranks N candidate documents against a query tensor on active compute device.
        query_emb: [1, Q_len, D] or [Q_len, D]
        doc_embs:  [N, D_len, D]
        """
        if query_emb.dim() == 2:
            query_emb = query_emb.unsqueeze(0)
        n_docs = doc_embs.size(0)
        q_expanded = query_emb.expand(n_docs, -1, -1).to(self.device)
        d_tensor = doc_embs.to(self.device)

        with torch.no_grad():
            scores = self.scorer_fn(q_expanded, d_tensor).cpu().tolist()

        paired = list(zip(doc_ids, [round(float(s), 4) for s in scores]))
        return sorted(paired, key=lambda x: x[1], reverse=True)

    def resolve_contradictions(
        self,
        claims: List[EpistemicClaim],
        current_year: int = 2026,
        alpha: float = 0.55,
        decay_lambda: float = 0.04,
    ) -> ContradictionResolution:
        """
        Resolves conflicting multi-hop claims about the same (entity, attribute) using
        Epistemic Authority Tier and Temporal Recency decay.
        """
        if not claims:
            raise ValueError("Claims list cannot be empty")

        scored_claims: List[Tuple[float, EpistemicClaim]] = []
        for c in claims:
            age = max(0, current_year - c.timestamp_year)
            recency_factor = math.exp(-decay_lambda * age)
            weight = round(alpha * c.authority_tier + (1.0 - alpha) * recency_factor, 4)
            scored_claims.append((weight, c))

        scored_claims.sort(key=lambda x: x[0], reverse=True)
        best_weight, winner = scored_claims[0]
        superseded = [
            (c.doc_id, c.value, c.timestamp_year) for _, c in scored_claims[1:]
        ]

        explanation = (
            f"Resolved '{winner.entity}.{winner.attribute}' in favor of [{winner.doc_id}] "
            f"({winner.timestamp_year}, weight={best_weight}): '{winner.value}'. "
            f"Superseded {len(superseded)} older/lower-authority claim(s)."
        )
        return ContradictionResolution(
            entity=winner.entity,
            attribute=winner.attribute,
            winning_doc_id=winner.doc_id,
            winning_value=winner.value,
            winning_weight=best_weight,
            superseded_claims=superseded,
            audit_explanation=explanation,
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    indexer = TorchTpuHbmIndexer(use_torch_compile=False)
    print(f"[Data Specialist] Running bfloat16 ColBERT MaxSim on device: {indexer.device}")

    q_vec = torch.randn(1, 8, 64)
    # Create 3 docs where doc_1 is strongly aligned with q_vec
    d0 = torch.randn(8, 64)
    d1 = q_vec.squeeze(0) + 0.05 * torch.randn(8, 64)
    d2 = torch.randn(8, 64)
    docs_tensor = torch.stack([d0, d1, d2], dim=0)

    ranked = indexer.rank_documents_maxsim(q_vec, docs_tensor, ["DOC_NOISE_0", "DOC_R136_TELEMETRY", "DOC_NOISE_2"])
    print("MaxSim Ranking:", ranked)
    assert ranked[0][0] == "DOC_R136_TELEMETRY"

    # Test SN 1987A vs Sanduleak -69 202 contradiction resolution
    conflicting_claims = [
        EpistemicClaim(
            doc_id="EARTH_CATALOG_1978",
            entity="Sanduleak -69 202",
            attribute="stellar_state",
            value="Intact Luminous Blue Supergiant (LMC)",
            timestamp_year=1978,
            authority_tier=0.60,
        ),
        EpistemicClaim(
            doc_id="NOOSPHERE_LMC_TELEMETRY_2026",
            entity="Sanduleak -69 202",
            attribute="stellar_state",
            value="Collapsed Core-Collapse Supernova Remnant SN 1987A with Equatorial Ring",
            timestamp_year=2026,
            authority_tier=0.98,
        ),
    ]
    res = indexer.resolve_contradictions(conflicting_claims)
    print("[Contradiction Resolver]", res.audit_explanation)
    assert res.winning_doc_id == "NOOSPHERE_LMC_TELEMETRY_2026"
    print("[PASS] Part 3 TorchTPU bfloat16 MaxSim Indexer & Contradiction Resolver verified!")
