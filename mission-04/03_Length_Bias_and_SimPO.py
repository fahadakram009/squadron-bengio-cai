"""
03_Length_Bias_and_SimPO.py

Task 3 — Data & Payload Specialist
Length Bias Diagnostics & Reference-Free SimPO

Requirements implemented:
1. Pearson length-bias diagnostic:
       r_len = corr(|y_w| - |y_l|, Delta_w - Delta_l)
   where Delta_w / Delta_l are supplied model preference scores (for example,
   policy-vs-reference sequence log-ratio scores). The function reports the
   coefficient and p-value.
2. SimPOLoss(nn.Module): reference-free, length-normalized SimPO objective
   using average per-token completion log-probabilities, beta=2.0, gamma=0.5.
3. VRAM benchmark: compares Policy+Reference model storage (DPO) against
   Policy-only storage (SimPO), and optionally measures CUDA peak allocation
   around caller-provided training-step functions.

Install:
    pip install torch pandas scipy

Run smoke tests:
    python 03_Length_Bias_and_SimPO.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Length-correlation diagnostic
# ---------------------------------------------------------------------------

@dataclass
class LengthCorrelationResult:
    n_pairs: int
    r_len: float
    p_value: float
    mean_length_diff: float
    mean_preference_margin: float


def length_correlation_diagnostic(
    chosen_lengths,
    rejected_lengths,
    chosen_scores,
    rejected_scores,
) -> LengthCorrelationResult:
    """
    Compute Pearson correlation between pairwise length differential and
    model preference margin:

        length_diff = |y_w| - |y_l|
        preference_margin = Delta_w - Delta_l
        r_len = Pearson(length_diff, preference_margin)

    Inputs may be Python sequences, NumPy arrays, or 1-D torch tensors.
    Delta values must be on the same score scale for chosen and rejected.
    Positive r_len indicates that pairs with a larger chosen-vs-rejected
    length advantage tend to have a larger model preference margin. This is
    a diagnostic association, not evidence of causation.
    """
    import numpy as np
    from scipy.stats import pearsonr

    def to_numpy_1d(value, name):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return arr

    cw = to_numpy_1d(chosen_lengths, "chosen_lengths")
    rl = to_numpy_1d(rejected_lengths, "rejected_lengths")
    sw = to_numpy_1d(chosen_scores, "chosen_scores")
    sl = to_numpy_1d(rejected_scores, "rejected_scores")

    if not (len(cw) == len(rl) == len(sw) == len(sl)):
        raise ValueError("All four inputs must contain the same number of pairs")
    if len(cw) < 2:
        raise ValueError("At least two preference pairs are required")
    if (cw < 0).any() or (rl < 0).any():
        raise ValueError("Response lengths must be non-negative")

    length_diff = cw - rl
    preference_margin = sw - sl

    # Pearson is undefined if either variable is constant.
    if np.ptp(length_diff) == 0 or np.ptp(preference_margin) == 0:
        r_len, p_value = float("nan"), float("nan")
    else:
        result = pearsonr(length_diff, preference_margin)
        r_len, p_value = float(result.statistic), float(result.pvalue)

    return LengthCorrelationResult(
        n_pairs=len(cw),
        r_len=r_len,
        p_value=p_value,
        mean_length_diff=float(length_diff.mean()),
        mean_preference_margin=float(preference_margin.mean()),
    )


# Backwards-friendly alias for notebooks/scripts.
analyze_length_bias = length_correlation_diagnostic


# ---------------------------------------------------------------------------
# 2. Token log-probabilities and reference-free SimPO
# ---------------------------------------------------------------------------

def completion_average_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return mean next-token log probability per sequence and scored token count.

    logits: [batch, sequence, vocab], unshifted causal-LM logits.
    labels: [batch, sequence]; prompt positions should be ignore_index.
    The causal shift is applied internally: logits[:, :-1] predicts labels[:, 1:].
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels batch/sequence dimensions must match")

    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)

    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must have shape [B,T]")
        valid = valid & attention_mask[:, 1:].bool()

    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_logps = token_logps * valid.to(token_logps.dtype)
    counts = valid.sum(dim=-1)
    avg_logps = token_logps.sum(dim=-1) / counts.clamp_min(1).to(token_logps.dtype)
    return avg_logps, counts


class SimPOLoss(nn.Module):
    """
    Reference-free SimPO loss.

    Given average per-token log probabilities:
        r_w = beta * log pi(y_w|x) / |y_w|
        r_l = beta * log pi(y_l|x) / |y_l|
        loss = -log sigmoid(r_w - r_l - gamma)

    Defaults follow the task specification: beta=2.0, gamma=0.5.
    No reference model or reference log probabilities are accepted/needed.
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(
        self,
        chosen_avg_logps: torch.Tensor,
        rejected_avg_logps: torch.Tensor,
        return_metrics: bool = False,
    ):
        if chosen_avg_logps.shape != rejected_avg_logps.shape:
            raise ValueError("Chosen and rejected log-probabilities must match in shape")

        chosen_avg_logps = chosen_avg_logps.float()
        rejected_avg_logps = rejected_avg_logps.float()

        logits = self.beta * (chosen_avg_logps - rejected_avg_logps) - self.gamma
        per_pair = -(
            (1.0 - self.label_smoothing) * F.logsigmoid(logits)
            + self.label_smoothing * F.logsigmoid(-logits)
        )

        if self.reduction == "mean":
            loss = per_pair.mean()
        elif self.reduction == "sum":
            loss = per_pair.sum()
        else:
            loss = per_pair

        if not return_metrics:
            return loss

        with torch.no_grad():
            metrics = {
                "reward_accuracy": (logits > 0).float().mean(),
                "logit_margin": logits.mean(),
                "chosen_reward": (self.beta * chosen_avg_logps).mean(),
                "rejected_reward": (self.beta * rejected_avg_logps).mean(),
                "reward_margin": (
                    self.beta * (chosen_avg_logps - rejected_avg_logps)
                ).mean(),
            }
        return loss, metrics

    def forward_from_logits(
        self,
        chosen_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        return_metrics: bool = False,
    ):
        """Convenience path from causal-LM logits and completion-only labels."""
        chosen_logps, chosen_counts = completion_average_log_probs(
            chosen_logits, chosen_labels, chosen_attention_mask, ignore_index
        )
        rejected_logps, rejected_counts = completion_average_log_probs(
            rejected_logits, rejected_labels, rejected_attention_mask, ignore_index
        )
        if (chosen_counts == 0).any() or (rejected_counts == 0).any():
            raise ValueError("Each chosen/rejected completion needs >=1 scored token")
        result = self.forward(chosen_logps, rejected_logps, return_metrics)
        return result


# ---------------------------------------------------------------------------
# 3. VRAM / memory benchmark: DPO (Policy + Ref) vs SimPO (Policy only)
# ---------------------------------------------------------------------------

def _module_storage_bytes(model: nn.Module) -> int:
    """Count unique parameter and buffer storage bytes for a model."""
    seen = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def benchmark_model_memory(
    policy_model: nn.Module,
    reference_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Compare model-weight storage for DPO (policy + reference) and SimPO
    (policy only). This is an exact parameter/buffer storage comparison for
    the supplied model instances, not a measured training-step peak.

    For a fair like-for-like comparison, policy_model and reference_model
    should be equivalent model architectures/dtypes. The reported reduction
    is relative to the DPO model-storage total.
    """
    policy_bytes = _module_storage_bytes(policy_model)
    if reference_model is None:
        # Assumes the DPO reference is a same-sized copy of the policy.
        reference_bytes = policy_bytes
        reference_assumed = True
    else:
        reference_bytes = _module_storage_bytes(reference_model)
        reference_assumed = False

    dpo_bytes = policy_bytes + reference_bytes
    simpo_bytes = policy_bytes
    reduction = 100.0 * (dpo_bytes - simpo_bytes) / dpo_bytes if dpo_bytes else 0.0

    return {
        "policy_model_mb": policy_bytes / (1024 ** 2),
        "reference_model_mb": reference_bytes / (1024 ** 2),
        "dpo_two_model_storage_mb": dpo_bytes / (1024 ** 2),
        "simpo_one_model_storage_mb": simpo_bytes / (1024 ** 2),
        "vram_reduction_percent": reduction,
        "reference_size_assumed_equal": float(reference_assumed),
    }


def benchmark_cuda_peak_memory(
    dpo_step: Callable[[], Any],
    simpo_step: Callable[[], Any],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Optionally measure CUDA peak allocated memory for actual training steps.

    Callbacks must perform one representative forward/backward/optimizer step
    for each method on the selected CUDA device. The caller is responsible for
    arranging model residency so DPO has Policy+Ref and SimPO has Policy only.
    Returns peak allocated MB for each callback. CUDA is required.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use benchmark_model_memory instead")
    device = device or torch.device("cuda")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")

    results = {}
    for name, step_fn in (("dpo", dpo_step), ("simpo", simpo_step)):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        results[f"{name}_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    dpo_peak = results["dpo_peak_allocated_mb"]
    simpo_peak = results["simpo_peak_allocated_mb"]
    results["peak_memory_reduction_percent"] = (
        100.0 * (dpo_peak - simpo_peak) / dpo_peak if dpo_peak else 0.0
    )
    return results


# ---------------------------------------------------------------------------
# 4. Smoke tests
# ---------------------------------------------------------------------------

def _smoke_test():
    torch.manual_seed(7)

    # Diagnostic test: length differential and preference margin co-vary.
    diag = length_correlation_diagnostic(
        chosen_lengths=[120, 140, 160, 180, 200],
        rejected_lengths=[100, 100, 100, 100, 100],
        chosen_scores=[1.2, 1.4, 1.6, 1.8, 2.0],
        rejected_scores=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    assert abs(diag.r_len - 1.0) < 1e-6
    print("Length diagnostic:", asdict(diag))

    # SimPO loss + gradient test.
    chosen = torch.tensor([-0.7, -0.8, -0.6], requires_grad=True)
    rejected = torch.tensor([-1.0, -1.1, -0.9], requires_grad=True)
    criterion = SimPOLoss(beta=2.0, gamma=0.5)
    loss, metrics = criterion(chosen, rejected, return_metrics=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert chosen.grad is not None and rejected.grad is not None
    print("SimPO loss:", float(loss.detach()), {
        k: float(v) for k, v in metrics.items()
    })

    # Memory comparison test with a tiny model. If both models are equal-sized,
    # model-weight storage should show the expected 50% reduction.
    policy = nn.Linear(16, 16)
    reference = nn.Linear(16, 16)
    memory = benchmark_model_memory(policy, reference)
    assert abs(memory["vram_reduction_percent"] - 50.0) < 1e-6
    print("Memory benchmark (model storage MB):", memory)

    print("All Task 3 smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
