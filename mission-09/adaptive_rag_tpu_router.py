"""
USSE Noosphere — Mission 09 (SEC-09-LMC: Large Magellanic Cloud / Tarantula Nebula)
Station Role 1: Systems Architect
Deliverable: adaptive_rag_tpu_router.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import re
import time
import torch

# Optional TorchTPU import (available on Google Colab TPU v5e-1 / Kaggle / Cloud TPU VMs)
try:
    import torch_tpu  # type: ignore
    HAS_TORCHTPU = True
except ImportError:
    HAS_TORCHTPU = False

RouteType = Literal["NO_RETRIEVAL", "SINGLE_STEP_RAG", "MULTI_HOP_AGENT"]


@dataclass
class HardwareTelemetry:
    device_type: str
    device_name: str
    dtype: str
    is_tpu_native: bool
    torch_compile_backend: str


@dataclass
class RoutingDecision:
    query: str
    route: RouteType
    complexity_score: float
    planned_hops: int
    sub_queries: List[str]
    estimated_ttft_ms: float
    estimated_token_cost: int
    rationale: str


def get_compute_device() -> torch.device:
    """
    Detects Open-Source TorchTPU ('tpu') first, then 'cuda', then 'cpu'.
    Ensures identical execution across Colab Free TPU v5e-1 and local laptops.
    """
    if HAS_TORCHTPU and hasattr(torch, "tpu") and torch.tpu.is_available():
        return torch.device("tpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hardware_telemetry() -> HardwareTelemetry:
    """Returns diagnostic telemetry for AURA Subsystem 09 hardware status."""
    dev = get_compute_device()
    if dev.type == "tpu":
        return HardwareTelemetry(
            device_type="tpu",
            device_name="Google Cloud TPU (torch_tpu OpenXLA/Pallas)",
            dtype="torch.bfloat16",
            is_tpu_native=True,
            torch_compile_backend="tpu",
        )
    elif dev.type == "cuda":
        return HardwareTelemetry(
            device_type="cuda",
            device_name=torch.cuda.get_device_name(0),
            dtype="torch.bfloat16",
            is_tpu_native=False,
            torch_compile_backend="inductor",
        )
    return HardwareTelemetry(
        device_type="cpu",
        device_name="Host CPU (bfloat16 Emulation Fallback)",
        dtype="torch.bfloat16",
        is_tpu_native=False,
        torch_compile_backend="eager",
    )


class AdaptiveRagTpuRouter:
    """
    Adaptive RAG Query Complexity Classifier & Execution Topology Router (Jeong et al., 2024).
    """

    MULTI_HOP_INDICATORS = (
        "compare", "contrast", "difference between", "correlate", "both",
        "how does", "impact of", "versus", "vs", "across", "reconcile",
        "before and after", "pre-collapse", "multi-hop", "and determine"
    )
    DOMAIN_ENTITY_INDICATORS = (
        "r136", "sn 1987a", "1987a", "30 doradus", "tarantula", "sanduleak",
        "sec-09-lmc", "lmc", "magellanic", "wolf-rayet", "bfloat16", "torchtpu",
        "aura", "noosphere", "neutrino", "ring"
    )

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.device = get_compute_device()

    def decompose_multi_hop_query(self, query: str) -> List[str]:
        """Splits a complex multi-hop query into bounded atomic sub-queries (<= max_hops)."""
        parts = re.split(r"\b(?:and determine|and compare|while|versus|vs\.?|as well as|and how)\b", query, flags=re.IGNORECASE)
        cleaned = [p.strip(" ,?.") for p in parts if len(p.strip()) > 8]
        if len(cleaned) <= 1:
            # Fallback conjunction split
            conj_parts = [p.strip(" ,?.") for p in re.split(r"\band\b", query, flags=re.IGNORECASE) if len(p.strip()) > 10]
            if len(conj_parts) > 1:
                cleaned = conj_parts
        return cleaned[: self.max_hops] if len(cleaned) > 1 else [query]

    def route_query(self, query: str) -> RoutingDecision:
        """
        Classifies query complexity into NO_RETRIEVAL, SINGLE_STEP_RAG, or MULTI_HOP_AGENT
        and estimates latency/token cost on TorchTPU vs. standard execution.
        """
        q_lower = query.lower().strip()

        multi_hits = sum(1 for kw in self.MULTI_HOP_INDICATORS if kw in q_lower)
        entity_hits = sum(1 for kw in self.DOMAIN_ENTITY_INDICATORS if kw in q_lower)
        clause_count = len(self.decompose_multi_hop_query(query))

        # Complexity score in [0, 1]
        raw_score = 0.35 * multi_hits + 0.25 * entity_hits + 0.20 * max(0, clause_count - 1)
        complexity_score = round(min(1.0, raw_score), 3)

        # Speedup factor when running on native TPU HBM
        tpu_accel = 1.0 if self.device.type == "tpu" else 1.35

        if multi_hits >= 1 and (entity_hits >= 2 or clause_count >= 2):
            sub_qs = self.decompose_multi_hop_query(query)
            hops = min(self.max_hops, max(2, len(sub_qs)))
            return RoutingDecision(
                query=query,
                route="MULTI_HOP_AGENT",
                complexity_score=max(0.70, complexity_score),
                planned_hops=hops,
                sub_queries=sub_qs,
                estimated_ttft_ms=round(185.0 * hops * tpu_accel, 1),
                estimated_token_cost=420 + 680 * hops,
                rationale=(
                    f"Detected compositional multi-hop structure ({multi_hits} relational operators, "
                    f"{entity_hits} domain entities). Routing to iterative CRAG + Self-RAG loop ({hops} hops)."
                ),
            )
        elif entity_hits >= 1 or "?" in q_lower and len(q_lower.split()) > 9:
            return RoutingDecision(
                query=query,
                route="SINGLE_STEP_RAG",
                complexity_score=max(0.35, min(0.65, complexity_score)),
                planned_hops=1,
                sub_queries=[query],
                estimated_ttft_ms=round(140.0 * tpu_accel, 1),
                estimated_token_cost=580,
                rationale=(
                    f"Single-hop domain lookup ({entity_hits} domain entities). "
                    "Routing to single-pass TorchTPU HBM ColBERT index."
                ),
            )
        else:
            return RoutingDecision(
                query=query,
                route="NO_RETRIEVAL",
                complexity_score=min(0.25, complexity_score),
                planned_hops=0,
                sub_queries=[],
                estimated_ttft_ms=round(55.0 * tpu_accel, 1),
                estimated_token_cost=160,
                rationale="Self-contained parametric or mathematical query; bypassing retrieval index (0 hops).",
            )


if __name__ == "__main__":
    hw = get_hardware_telemetry()
    print(f"[Architect Hardware Check] Device={hw.device_type} ({hw.device_name}) | Backend={hw.torch_compile_backend}")

    router = AdaptiveRagTpuRouter(max_hops=3)
    test_queries = [
        "What is the speed of light c in km/s?",
        "What is the bolometric luminosity of Wolf-Rayet star R136a1 in 30 Doradus?",
        "Compare the pre-collapse progenitor Sanduleak -69 202 with SN 1987A ring kinematics and determine how R136 UV flux ionizes the surrounding LMC filaments.",
    ]
    expected_routes: List[RouteType] = ["NO_RETRIEVAL", "SINGLE_STEP_RAG", "MULTI_HOP_AGENT"]

    for q, exp in zip(test_queries, expected_routes):
        dec = router.route_query(q)
        print(f"Query: {q[:68]}...")
        print(f" -> Route: {dec.route} (score={dec.complexity_score}, hops={dec.planned_hops}, TTFT={dec.estimated_ttft_ms}ms)")
        assert dec.route == exp, f"Expected {exp}, got {dec.route}"

    print("[PASS] Part 1 Adaptive RAG TorchTPU Router verified successfully!")
