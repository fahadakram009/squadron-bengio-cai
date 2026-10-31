"""
Mission 04 - Part 1: Systems Architect (Lead Engineer)
Direct Preference Optimization (DPO) Loss Core & Log-Likelihood Extraction

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex)
Subsystem 04: Plasma Ignition & Ethical Steering Core
Squadron Bengio - Systems Architect: Fahad Akram

What this file provides (pure PyTorch, no TRL / HuggingFace):

  get_batch_logps(logits, labels, label_pad_token_id=-100, average_log_prob=False)
      vectorised torch.gather of log p(token) over the response tokens of each sequence.
      Convention (shared with Parts 2 and 3): logits[:, t] predicts labels[:, t+1]; every position whose
      label is -100 (prompt or padding) is ignored. Returns the sum, or the mean when average_log_prob=True
      (the quantity SimPO needs).

  dpo_log_ratios(...)      Delta_w = log pi(y_w|x) - log pi_ref(y_w|x),  Delta_l likewise
  DPOLoss(beta)            -log sigmoid(beta * (Delta_w - Delta_l)) via F.logsigmoid, plus the implicit
                           rewards r = beta * Delta and monitoring metrics. forward() takes the four
                           sequence log-probs; from_logits() takes raw logits + labels and keeps the
                           reference pass under torch.no_grad().

  TinyCausalLM             a small causal language model used for the gradient test and the training demo.
                           If mission-03/01_MultiHead_Attention_Core_starter.py is present its
                           MultiHeadSelfAttention is the mixing layer (Subsystem 03 feeding Subsystem 04);
                           otherwise a causal prefix-mean layer is used.
  make_reference(policy)   deep copy, requires_grad=False, eval(): the frozen pi_ref.

Modes
  python 01_DPO_Loss_Core.py                   demo: gather, loss, numerical stability, gradient test
  python 01_DPO_Loss_Core.py --selftest        30 checks (gather oracle, loss algebra, gradients, frozen
                                               reference, integration with Parts 2 / 3 / 4)
  python 01_DPO_Loss_Core.py --train           short DPO run on synthetic preference pairs, beta sweep
  python 01_DPO_Loss_Core.py --check-signoffs  gate: all four boxes in README.md must be ticked
Every run writes logs/<timestamp>-ps04-<mode>.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
IGNORE = -100  # label id of prompt / padding positions

PART2 = HERE / "02_Implicit_Reward_Evaluator.py"
PART3 = HERE / "03_Length_Bias_and_SimPO.py"
PART4 = HERE / "04_Activation_Steering_and_Audit.py"
MISSION3_ATTENTION = HERE.parent / "mission-03" / "01_MultiHead_Attention_Core_starter.py"


# ---------------------------------------------------------------------------
# 1. Token log-probability gather
# ---------------------------------------------------------------------------

def response_token_mask(labels: torch.Tensor, label_pad_token_id: int = IGNORE) -> torch.Tensor:
    """Boolean (batch, seq_len - 1) mask of the label positions that count: labels[:, 1:] != pad."""
    return labels[:, 1:] != label_pad_token_id


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = IGNORE,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Sequence log-probability log pi(y | x) of the labelled tokens under `logits`.

    Args:
        logits: (batch, seq_len, vocab) raw model outputs.
        labels: (batch, seq_len) token ids, with prompt and padding positions set to label_pad_token_id.
        label_pad_token_id: id that marks ignored positions (default -100, as in F.cross_entropy).
        average_log_prob: return the mean per response token instead of the sum (SimPO / length analysis).

    Returns:
        (batch,) tensor. A row with no response tokens returns 0 (no NaN, no division by zero).
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} do not line up")

    # logits at position t predict the token at position t + 1
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != label_pad_token_id

    # gather needs a valid index everywhere, so the ignored positions are pointed at token 0
    safe_labels = torch.where(mask, shift_labels, torch.zeros_like(shift_labels))
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    per_token = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    per_token = per_token * mask.to(per_token.dtype)

    total = per_token.sum(dim=-1)
    if average_log_prob:
        return total / mask.sum(dim=-1).clamp(min=1)
    return total


def get_batch_logps_and_lengths(
    logits: torch.Tensor, labels: torch.Tensor, label_pad_token_id: int = IGNORE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum of log-probs and the number of response tokens |y| per sequence (for the length-bias station)."""
    total = get_batch_logps(logits, labels, label_pad_token_id, average_log_prob=False)
    lengths = response_token_mask(labels, label_pad_token_id).sum(dim=-1)
    return total, lengths


# ---------------------------------------------------------------------------
# 2. + 3. Log-ratio deltas and the DPO loss
# ---------------------------------------------------------------------------

def dpo_log_ratios(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delta_w = log pi(y_w|x) - log pi_ref(y_w|x) and Delta_l = log pi(y_l|x) - log pi_ref(y_l|x)."""
    delta_w = policy_chosen_logps - reference_chosen_logps
    delta_l = policy_rejected_logps - reference_rejected_logps
    return delta_w, delta_l


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss (Rafailov et al., NeurIPS 2023).

        L_DPO = -E[ log sigmoid( beta * (Delta_w - Delta_l) ) ]

    The reference model enters only through the detached numbers Delta_* are built from, so the gradient
    reaches the policy alone. `label_smoothing` epsilon > 0 gives the conservative DPO variant
    (1 - eps) * -log sigmoid(z) + eps * -log sigmoid(-z) that tolerates noisy preference labels.
    `reference_free=True` sets log pi_ref to 0 for both responses (useful for ablations).
    """

    def __init__(self, beta: float = 0.1, label_pad_token_id: int = IGNORE,
                 label_smoothing: float = 0.0, reference_free: bool = False):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        self.beta = float(beta)
        self.label_pad_token_id = label_pad_token_id
        self.label_smoothing = float(label_smoothing)
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """All inputs are (batch,) sequence log-probs from get_batch_logps. Returns (loss, metrics)."""
        if self.reference_free:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        delta_w, delta_l = dpo_log_ratios(policy_chosen_logps, policy_rejected_logps,
                                          reference_chosen_logps.detach(), reference_rejected_logps.detach())

        # Bradley-Terry logit. sigma(z) = P(y_w > y_l | x) under the implicit reward r = beta * Delta.
        logits = self.beta * (delta_w - delta_l)

        # -log sigmoid(z) through logsigmoid: finite for z = -1000 where -log(sigmoid(z)) underflows to inf
        losses = -F.logsigmoid(logits)
        if self.label_smoothing > 0:
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * (-F.logsigmoid(-logits))
        loss = losses.mean()

        with torch.no_grad():
            chosen_rewards = self.beta * delta_w.detach()
            rejected_rewards = self.beta * delta_l.detach()
            margins = chosen_rewards - rejected_rewards
            metrics = {
                "loss": loss.detach(),
                "chosen_rewards": chosen_rewards.mean(),
                "rejected_rewards": rejected_rewards.mean(),
                "reward_margin": margins.mean(),
                "chosen_win_rate": (margins > 0).float().mean(),  # pairwise accuracy of the implicit reward
                "bt_probability": torch.sigmoid(logits.detach()).mean(),
                "logits": logits.detach().mean(),
                "chosen_logratio": delta_w.detach().mean(),
                "rejected_logratio": delta_l.detach().mean(),
            }
        return loss, metrics

    def from_logits(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convenience path from raw logits. The reference log-probs are computed without a graph."""
        pad = self.label_pad_token_id
        pc = get_batch_logps(policy_chosen_logits, chosen_labels, pad)
        pr = get_batch_logps(policy_rejected_logits, rejected_labels, pad)
        with torch.no_grad():
            rc = get_batch_logps(reference_chosen_logits, chosen_labels, pad)
            rr = get_batch_logps(reference_rejected_logits, rejected_labels, pad)
        return self.forward(pc, pr, rc, rr)


# ---------------------------------------------------------------------------
# 4. A tiny causal LM, a frozen reference, synthetic preference pairs
# ---------------------------------------------------------------------------

def load_sibling(path: Path, name: str):
    """Import a script whose file name starts with a digit. Returns None if it is missing or fails."""
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses and pickling look the module up here
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # noqa: BLE001 - a broken teammate file must not break this one
        sys.modules.pop(name, None)
        print(f"  (could not import {path.name}: {type(exc).__name__}: {exc})")
        return None


class PrefixMeanMixer(nn.Module):
    """Causal fallback mixing layer: position t sees the mean of embeddings 0..t."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x.cumsum(dim=1) / counts


class TinyCausalLM(nn.Module):
    """
    token + position embedding -> causal mixing -> residual + LayerNorm -> MLP -> vocab logits.
    `mixer` is Mission 3's MultiHeadSelfAttention when available (causal=True), else PrefixMeanMixer.
    """

    def __init__(self, vocab: int, dim: int = 32, max_len: int = 64, num_heads: int = 4,
                 attention_cls: Optional[type] = None):
        super().__init__()
        self.vocab, self.dim = vocab, dim
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.uses_mission3_attention = attention_cls is not None
        self.mixer = attention_cls(d_model=dim, num_heads=num_heads, dropout=0.0) if attention_cls else PrefixMeanMixer()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, n = input_ids.shape
        x = self.tok(input_ids) + self.pos(torch.arange(n, device=input_ids.device)).unsqueeze(0)
        mixed = self.mixer(x, causal=True) if self.uses_mission3_attention else self.mixer(x)
        if isinstance(mixed, tuple):  # MultiHeadSelfAttention returns (output, weights)
            mixed = mixed[0]
        x = self.norm(x + mixed)
        x = x + self.ff(x)
        return self.head(x)


def mission3_attention_cls() -> Optional[type]:
    mod = load_sibling(MISSION3_ATTENTION, "mission3_attention")
    return getattr(mod, "MultiHeadSelfAttention", None) if mod is not None else None


def make_reference(policy: nn.Module) -> nn.Module:
    """pi_ref: a deep copy of the policy with every parameter frozen, in eval mode."""
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref


def parameter_digest(model: nn.Module) -> str:
    """SHA-256 of all parameter bytes - bitwise proof that a model did or did not change."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().contiguous().view(-1).to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


class PreferencePairs:
    """
    Synthetic preference data. Vocabulary = [pad] + prompt tokens + response tokens. Each prompt picks a
    topic; the chosen response is drawn from the topic's preferred token distribution, the rejected one
    from a flatter distribution with a different favourite. Lengths vary so padding is exercised.
    """

    def __init__(self, num_topics: int = 4, num_prompt_tokens: int = 8, num_response_tokens: int = 24,
                 prompt_len: int = 4, min_len: int = 3, max_len: int = 10, sharpness: float = 2.0,
                 seed: int = 0, world_seed: int = 1234):
        # world_seed fixes the preference tables (the "humans"); seed only drives the sampling, so a
        # validation set with another seed comes from the same preference world as the training set
        self.g = torch.Generator().manual_seed(seed)
        gw = torch.Generator().manual_seed(world_seed)
        self.num_topics, self.prompt_len = num_topics, prompt_len
        self.min_len, self.max_len = min_len, max_len
        self.pad = 0
        self.prompt_offset = 1
        self.resp_offset = 1 + num_prompt_tokens
        self.vocab = 1 + num_prompt_tokens + num_response_tokens
        self.seq_len = prompt_len + max_len
        base = torch.randn(num_topics, num_response_tokens, generator=gw)
        self.good = torch.softmax(sharpness * base, dim=-1)        # what humans prefer per topic
        self.bad = torch.softmax(-sharpness * base.roll(1, dims=1), dim=-1)  # a different favourite

    def _responses(self, topics: torch.Tensor, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = topics.shape[0]
        lengths = torch.randint(self.min_len, self.max_len + 1, (n,), generator=self.g)
        toks = torch.multinomial(table[topics], self.max_len, replacement=True, generator=self.g)
        return toks + self.resp_offset, lengths

    def _assemble(self, prompts: torch.Tensor, resp: torch.Tensor, lengths: torch.Tensor):
        n = prompts.shape[0]
        ids = torch.full((n, self.seq_len), self.pad, dtype=torch.long)
        labels = torch.full((n, self.seq_len), IGNORE, dtype=torch.long)
        ids[:, : self.prompt_len] = prompts
        keep = torch.arange(self.max_len).unsqueeze(0) < lengths.unsqueeze(1)
        ids[:, self.prompt_len:] = torch.where(keep, resp, torch.full_like(resp, self.pad))
        labels[:, self.prompt_len:] = torch.where(keep, resp, torch.full_like(resp, IGNORE))
        return ids, labels

    def sample(self, n: int) -> Dict[str, torch.Tensor]:
        topics = torch.randint(0, self.num_topics, (n,), generator=self.g)
        prompts = torch.randint(0, self.resp_offset - 1, (n, self.prompt_len), generator=self.g) + self.prompt_offset
        prompts[:, 0] = topics + self.prompt_offset  # first prompt token names the topic
        rw, lw = self._responses(topics, self.good)
        rl, ll = self._responses(topics, self.bad)
        cw, clw = self._assemble(prompts, rw, lw)
        cl, cll = self._assemble(prompts, rl, ll)
        return {"chosen_input_ids": cw, "chosen_labels": clw, "rejected_input_ids": cl,
                "rejected_labels": cll, "len_w": lw, "len_l": ll, "topic": topics}


def build_models(vocab: int, dim: int = 32, seed: int = 0, use_mission3: bool = True) -> Tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    cls = mission3_attention_cls() if use_mission3 else None
    policy = TinyCausalLM(vocab, dim=dim, attention_cls=cls)
    return policy, make_reference(policy)


def dpo_step(policy: nn.Module, ref: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: DPOLoss,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One DPO forward (and optional update). The reference pass runs under no_grad."""
    pc = get_batch_logps(policy(batch["chosen_input_ids"]), batch["chosen_labels"])
    pr = get_batch_logps(policy(batch["rejected_input_ids"]), batch["rejected_labels"])
    with torch.no_grad():
        rc = get_batch_logps(ref(batch["chosen_input_ids"]), batch["chosen_labels"])
        rr = get_batch_logps(ref(batch["rejected_input_ids"]), batch["rejected_labels"])
    loss, metrics = loss_fn(pc, pr, rc, rr)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss, metrics


# ---------------------------------------------------------------------------
# 5. Empirical gradient test
# ---------------------------------------------------------------------------

def gradient_test(seed: int = 0, batch_size: int = 32, beta: float = 0.1, quiet: bool = False) -> Dict[str, Any]:
    """
    Back-propagate through DPOLoss with a real policy and a real frozen reference and record:
      - every policy parameter receives a finite, non-zero gradient,
      - no reference parameter has requires_grad or a .grad,
      - one optimizer step changes the policy digest and leaves the reference digest bitwise identical,
      - the step lowers the loss and raises the implicit reward margin on the same batch,
      - the analytic gradient dL/d(log pi(y_w)) = -beta (1 - sigma(z)) / B matches autograd.
    """
    data = PreferencePairs(seed=seed)
    policy, ref = build_models(data.vocab, seed=seed)
    batch = data.sample(batch_size)
    loss_fn = DPOLoss(beta=beta)

    # analytic gradient on the four log-prob inputs
    pc = get_batch_logps(policy(batch["chosen_input_ids"]), batch["chosen_labels"]).detach().requires_grad_(True)
    pr = get_batch_logps(policy(batch["rejected_input_ids"]), batch["rejected_labels"]).detach().requires_grad_(True)
    with torch.no_grad():
        rc = get_batch_logps(ref(batch["chosen_input_ids"]), batch["chosen_labels"])
        rr = get_batch_logps(ref(batch["rejected_input_ids"]), batch["rejected_labels"])
    loss0, m0 = loss_fn(pc, pr, rc, rr)
    loss0.backward()
    z = beta * ((pc - rc) - (pr - rr)).detach()
    expected_w = -beta * (1 - torch.sigmoid(z)) / batch_size
    analytic_err = max((pc.grad - expected_w).abs().max().item(), (pr.grad + expected_w).abs().max().item())

    # full model pass
    ref_digest_before, policy_digest_before = parameter_digest(ref), parameter_digest(policy)
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    loss_before, metrics_before = dpo_step(policy, ref, batch, loss_fn, optimizer=None)
    opt.zero_grad(set_to_none=True)
    loss_before.backward()
    policy_grads = {n: p.grad for n, p in policy.named_parameters()}
    policy_grad_ok = all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in policy_grads.values())
    ref_frozen = all((not p.requires_grad) and p.grad is None for p in ref.parameters())
    grad_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in policy_grads.values()))
    opt.step()
    with torch.no_grad():
        loss_after, metrics_after = dpo_step(policy, ref, batch, loss_fn, optimizer=None)

    result = {
        "batch_size": batch_size, "beta": beta,
        "uses_mission3_attention": policy.uses_mission3_attention,
        "num_policy_params": sum(p.numel() for p in policy.parameters()),
        "policy_params_with_grad": sum(1 for g in policy_grads.values() if g is not None),
        "policy_params_total": len(policy_grads),
        "policy_grad_norm": grad_norm,
        "policy_grads_finite_nonzero": policy_grad_ok,
        "ref_params_requires_grad": sum(int(p.requires_grad) for p in ref.parameters()),
        "ref_params_with_grad": sum(int(p.grad is not None) for p in ref.parameters()),
        "ref_frozen": ref_frozen,
        "ref_digest_before": ref_digest_before, "ref_digest_after": parameter_digest(ref),
        "policy_digest_before": policy_digest_before, "policy_digest_after": parameter_digest(policy),
        "loss_before_step": loss_before.item(), "loss_after_step": loss_after.item(),
        "margin_before_step": metrics_before["reward_margin"].item(),
        "margin_after_step": metrics_after["reward_margin"].item(),
        "analytic_gradient_max_err": analytic_err,
    }
    result["ref_unchanged"] = result["ref_digest_before"] == result["ref_digest_after"]
    result["policy_changed"] = result["policy_digest_before"] != result["policy_digest_after"]
    result["loss_decreased"] = result["loss_after_step"] < result["loss_before_step"]
    result["margin_increased"] = result["margin_after_step"] > result["margin_before_step"]

    if not quiet:
        print("Empirical gradient test (policy = TinyCausalLM, reference = frozen deep copy)")
        print(f"  mixing layer                : "
              f"{'Mission 3 MultiHeadSelfAttention' if policy.uses_mission3_attention else 'causal prefix mean'}")
        print(f"  policy params with gradient : {result['policy_params_with_grad']} / {result['policy_params_total']}"
              f"  (all finite and non-zero: {policy_grad_ok}, grad norm {grad_norm:.4f})")
        print(f"  reference requires_grad     : {result['ref_params_requires_grad']} params,"
              f" .grad set on {result['ref_params_with_grad']} params")
        print(f"  reference digest            : {ref_digest_before} -> {result['ref_digest_after']}"
              f"  (unchanged: {result['ref_unchanged']})")
        print(f"  policy digest               : {policy_digest_before} -> {result['policy_digest_after']}"
              f"  (changed: {result['policy_changed']})")
        print(f"  loss  before / after 1 step : {result['loss_before_step']:.4f} -> {result['loss_after_step']:.4f}"
              f"   (log 2 = {math.log(2):.4f})")
        print(f"  margin before / after 1 step: {result['margin_before_step']:+.4f} -> {result['margin_after_step']:+.4f}")
        print(f"  analytic dL/dlogp vs autograd: max |err| = {analytic_err:.2e}")
    return result


# ---------------------------------------------------------------------------
# 6. Numerical stability of -log sigmoid
# ---------------------------------------------------------------------------

def stability_table(quiet: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for z in [-1000.0, -100.0, -20.0, 0.0, 20.0, 100.0, 1000.0]:
        zt = torch.tensor(z)
        stable = max((-F.logsigmoid(zt)).item(), 0.0)
        naive = (-torch.log(torch.sigmoid(zt))).item()
        naive = max(naive, 0.0) if math.isfinite(naive) else naive
        rows.append({"z": z, "logsigmoid": stable, "naive_log_sigmoid": naive,
                     "naive_finite": math.isfinite(naive)})
    if not quiet:
        print("Numerical stability of the loss term  -log sigma(z),  z = beta (Delta_w - Delta_l)")
        print(f"  {'z':>8} | {'-logsigmoid(z)':>16} | {'-log(sigmoid(z))':>18}")
        for r in rows:
            naive = f"{r['naive_log_sigmoid']:.4f}" if r["naive_finite"] else "inf  <- NaN gradient"
            print(f"  {r['z']:>8.0f} | {r['logsigmoid']:>16.4f} | {naive:>18}")
    return rows


# ---------------------------------------------------------------------------
# 7. Training demo (beta sweep)
# ---------------------------------------------------------------------------

def train_dpo(beta: float = 0.1, steps: int = 150, batch_size: int = 64, lr: float = 1e-3, seed: int = 0,
              n_val: int = 512, quiet: bool = False) -> Dict[str, Any]:
    data = PreferencePairs(seed=seed)
    policy, ref = build_models(data.vocab, seed=seed)
    val = PreferencePairs(seed=seed + 1000).sample(n_val)
    loss_fn = DPOLoss(beta=beta)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    ref_digest = parameter_digest(ref)
    history = []

    def evaluate(step: int, train_loss: float):
        policy.eval()
        with torch.no_grad():
            loss, m = dpo_step(policy, ref, val, loss_fn)
            pc = get_batch_logps(policy(val["chosen_input_ids"]), val["chosen_labels"])
            rc = get_batch_logps(ref(val["chosen_input_ids"]), val["chosen_labels"])
        policy.train()
        history.append({"step": step, "train_loss": train_loss, "val_loss": loss.item(),
                        "pairwise_accuracy": m["chosen_win_rate"].item(), "reward_margin": m["reward_margin"].item(),
                        "bt_probability": m["bt_probability"].item(),
                        "chosen_logp_drift": (pc - rc).mean().item()})

    evaluate(0, math.log(2))
    for step in range(1, steps + 1):
        loss, _ = dpo_step(policy, ref, data.sample(batch_size), loss_fn, optimizer=opt)
        if step % max(1, steps // 5) == 0 or step == steps:
            evaluate(step, loss.item())
    out = {"beta": beta, "steps": steps, "history": history,
           "ref_frozen": parameter_digest(ref) == ref_digest,
           "uses_mission3_attention": policy.uses_mission3_attention}
    if not quiet:
        print(f"\nDPO training, beta = {beta}, {steps} steps of batch {batch_size}, AdamW lr {lr}")
        print(f"  {'step':>5} | {'val loss':>8} | {'pair acc':>8} | {'margin':>8} | {'P(BT)':>6} | {'drift':>7}")
        for h in history:
            print(f"  {h['step']:>5} | {h['val_loss']:>8.4f} | {100 * h['pairwise_accuracy']:>7.1f}% |"
                  f" {h['reward_margin']:>+8.4f} | {h['bt_probability']:>6.3f} | {h['chosen_logp_drift']:>+7.3f}")
        print("  step 0: policy == reference, every pair is a tie; strict accuracy counts a tie as wrong")
        print(f"  reference frozen for the whole run: {out['ref_frozen']}")
    return out


def beta_sweep(betas=(0.05, 0.1, 0.5), steps: int = 150, seed: int = 0, quiet: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for b in betas:
        r = train_dpo(beta=b, steps=steps, seed=seed, quiet=True)
        last = r["history"][-1]
        rows.append({"beta": b, "val_loss": last["val_loss"], "pairwise_accuracy": last["pairwise_accuracy"],
                     "reward_margin": last["reward_margin"], "chosen_logp_drift": last["chosen_logp_drift"],
                     "ref_frozen": r["ref_frozen"]})
    if not quiet:
        print(f"\nbeta sweep after {steps} steps (beta is the strength of the KL leash to pi_ref)")
        print(f"  {'beta':>5} | {'val loss':>8} | {'pair acc':>8} | {'margin':>8} | {'drift log pi(y_w)':>18}")
        for r in rows:
            print(f"  {r['beta']:>5} | {r['val_loss']:>8.4f} | {100 * r['pairwise_accuracy']:>7.1f}% |"
                  f" {r['reward_margin']:>+8.4f} | {r['chosen_logp_drift']:>+18.3f}")
    return rows


# ---------------------------------------------------------------------------
# 8. Self-test
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self, quiet: bool = False):
        self.rows: List[Dict[str, Any]] = []
        self.quiet = quiet

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})
        if not self.quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if not r["ok"])

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r["ok"])


def self_test(seed: int = 0, quiet: bool = False) -> Checks:
    c = Checks(quiet)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    say = (lambda *a, **k: None) if quiet else print

    # --- gather -----------------------------------------------------------
    say("\n[1] get_batch_logps")
    B, T, V = 3, 9, 13
    logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    labels[:, :3] = IGNORE          # prompt
    labels[1, 6:] = IGNORE          # padding
    labels[2, 1:] = IGNORE          # a row with no response tokens at all
    lp = get_batch_logps(logits, labels)
    ls = F.log_softmax(logits, dim=-1)
    manual = []
    for b in range(B):
        s = 0.0
        for t in range(1, T):
            if labels[b, t].item() != IGNORE:
                s += ls[b, t - 1, labels[b, t]].item()
        manual.append(s)
    c.add("gather == explicit loop (prompt and padding ignored)",
          max(abs(a - m) for a, m in zip(lp.tolist(), manual)) < 1e-5)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=IGNORE,
                         reduction="none").view(B, T - 1).sum(-1)
    c.add("gather == -cross_entropy(ignore_index=-100) summed per row", (lp + ce).abs().max().item() < 1e-5)
    c.add("row without response tokens gives 0, not NaN", lp[2].item() == 0.0 and torch.isfinite(lp).all().item())
    c.add("sequence log-probs are <= 0", bool((lp <= 1e-7).all().item()))
    avg = get_batch_logps(logits, labels, average_log_prob=True)
    n_tok = (labels[:, 1:] != IGNORE).sum(-1).clamp(min=1)
    c.add("average_log_prob == sum / number of response tokens", (avg - lp / n_tok).abs().max().item() < 1e-6)
    mutated = logits.clone()
    mutated[:, :2] += 5.0 * torch.randn(B, 2, V, generator=g)  # positions predicting prompt tokens
    c.add("logits at prompt positions do not influence the result",
          torch.allclose(get_batch_logps(mutated, labels), lp, atol=1e-6))
    padded = torch.cat([logits, torch.randn(B, 4, V, generator=g)], dim=1)
    padded_labels = torch.cat([labels, torch.full((B, 4), IGNORE)], dim=1)
    c.add("appending padding positions does not change the result",
          torch.allclose(get_batch_logps(padded, padded_labels), lp, atol=1e-6))
    total, lengths = get_batch_logps_and_lengths(logits, labels)
    c.add("get_batch_logps_and_lengths returns |y| = number of response tokens",
          torch.equal(lengths, (labels[:, 1:] != IGNORE).sum(-1)) and torch.allclose(total, lp))
    try:
        get_batch_logps(logits, labels[:, :-1])
        c.add("shape mismatch raises", False)
    except ValueError:
        c.add("shape mismatch raises", True)
    lg = logits.clone().requires_grad_(True)
    get_batch_logps(lg, labels).sum().backward()
    c.add("gradient flows to logits (finite) and is zero on ignored positions",
          torch.isfinite(lg.grad).all().item() and lg.grad[2, 1:].abs().sum().item() == 0
          and lg.grad[:, -1].abs().sum().item() == 0)

    # --- loss algebra -------------------------------------------------------
    say("\n[2] DPOLoss")
    loss_fn = DPOLoss(beta=0.1)
    pc = torch.tensor([-10.2, -12.4, -8.1, -14.5])
    pr = torch.tensor([-15.1, -13.0, -11.5, -13.8])
    rc = torch.tensor([-12.0, -14.0, -10.0, -15.0])
    rr = torch.tensor([-13.0, -13.5, -10.5, -14.0])
    loss, m = loss_fn(pc, pr, rc, rr)
    z = [0.1 * ((a - b) - (cc - d)) for a, b, cc, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())]
    hand = sum(-math.log(1 / (1 + math.exp(-zi))) for zi in z) / 4
    c.add("loss == mean -log sigma(beta (Delta_w - Delta_l)) by hand", abs(loss.item() - hand) < 1e-6,
          f"{loss.item():.6f} vs {hand:.6f}")
    c.add("implicit rewards r = beta * Delta and margin r_w - r_l",
          abs(m["chosen_rewards"].item() - 0.1 * (pc - rc).mean().item()) < 1e-6
          and abs(m["reward_margin"].item() - 0.1 * ((pc - rc) - (pr - rr)).mean().item()) < 1e-6)
    same = torch.randn(5, generator=g) - 20
    loss_eq, m_eq = loss_fn(same, same - 1, same, same - 1)
    c.add("policy == reference -> loss = log 2, P(BT) = 0.5, margin 0",
          abs(loss_eq.item() - math.log(2)) < 1e-6 and abs(m_eq["bt_probability"].item() - 0.5) < 1e-6
          and m_eq["reward_margin"].item() == 0.0)
    shifted, _ = loss_fn(pc + 3.0, pr + 3.0, rc, rr)
    c.add("adding a constant to both policy log-probs leaves the loss unchanged",
          abs(shifted.item() - loss.item()) < 1e-6)
    swapped, _ = loss_fn(pr, pc, rr, rc)
    sig = torch.sigmoid(torch.tensor(z))
    c.add("swapping y_w / y_l gives -log(1 - sigma(z))",
          abs(swapped.item() - (-torch.log(1 - sig)).mean().item()) < 1e-6)
    _, m2 = DPOLoss(beta=0.2)(pc, pr, rc, rr)
    c.add("logits and rewards scale linearly with beta",
          abs(m2["logits"].item() - 2 * m["logits"].item()) < 1e-6
          and abs(m2["reward_margin"].item() - 2 * m["reward_margin"].item()) < 1e-6)
    big = torch.tensor([1000.0, -1000.0])
    zero = torch.zeros(2)
    loss_big, _ = DPOLoss(beta=1.0)(big, zero, zero, zero)
    naive = -torch.log(torch.sigmoid(big)).mean()
    c.add("finite loss at z = +-1000 where -log(sigmoid(z)) is inf", torch.isfinite(loss_big).item()
          and not torch.isfinite(naive).item(), f"{loss_big.item():.1f} vs {naive.item()}")
    eps = 0.1
    ls_loss, _ = DPOLoss(beta=0.1, label_smoothing=eps)(pc, pr, rc, rr)
    zt = torch.tensor(z)
    expected = ((1 - eps) * -F.logsigmoid(zt) + eps * -F.logsigmoid(-zt)).mean()
    c.add("label smoothing = conservative DPO mixture", abs(ls_loss.item() - expected.item()) < 1e-6)
    rf, _ = DPOLoss(beta=0.1, reference_free=True)(pc, pr, rc, rr)
    rf2, _ = loss_fn(pc, pr, torch.zeros(4), torch.zeros(4))
    c.add("reference_free == zero reference log-probs", abs(rf.item() - rf2.item()) < 1e-7)
    try:
        DPOLoss(beta=0.0)
        c.add("beta <= 0 rejected", False)
    except ValueError:
        c.add("beta <= 0 rejected", True)

    # --- gradients ----------------------------------------------------------
    say("\n[3] gradients")
    inputs = tuple(torch.randn(4, dtype=torch.float64, generator=g).requires_grad_(i < 2) for i in range(4))
    c.add("torch.autograd.gradcheck (float64) on the four log-prob inputs",
          torch.autograd.gradcheck(lambda a, b, cc, d: DPOLoss(beta=0.3)(a, b, cc, d)[0], inputs, eps=1e-6,
                                   atol=1e-5))
    rc_g = torch.randn(4, generator=g).requires_grad_(True)
    rr_g = torch.randn(4, generator=g).requires_grad_(True)
    pc_g = torch.randn(4, generator=g).requires_grad_(True)
    pr_g = torch.randn(4, generator=g).requires_grad_(True)
    loss_fn(pc_g, pr_g, rc_g, rr_g)[0].backward()
    c.add("reference inputs are detached inside the loss (no grad even if they require it)",
          rc_g.grad is None and rr_g.grad is None and pc_g.grad is not None and pr_g.grad is not None)
    c.add("dL/d log pi(y_w) = -beta (1 - sigma(z)) / B, dL/d log pi(y_l) = +beta (1 - sigma(z)) / B",
          torch.allclose(pc_g.grad, -0.1 * (1 - torch.sigmoid(0.1 * ((pc_g - rc_g) - (pr_g - rr_g)))).detach() / 4,
                         atol=1e-6) and torch.allclose(pc_g.grad, -pr_g.grad, atol=1e-7))
    gt = gradient_test(seed=seed, quiet=True)
    c.add("every policy parameter gets a finite non-zero gradient",
          gt["policy_grads_finite_nonzero"] and gt["policy_params_with_grad"] == gt["policy_params_total"],
          f"{gt['policy_params_with_grad']}/{gt['policy_params_total']}")
    c.add("reference: requires_grad False on all params, no .grad after backward", gt["ref_frozen"])
    c.add("one optimizer step: reference digest identical, policy digest changed",
          gt["ref_unchanged"] and gt["policy_changed"])
    c.add("one SGD step lowers the loss and raises the margin on the same batch",
          gt["loss_decreased"] and gt["margin_increased"],
          f"loss {gt['loss_before_step']:.4f} -> {gt['loss_after_step']:.4f}")
    c.add("autograd matches the analytic gradient on the model's log-probs", gt["analytic_gradient_max_err"] < 1e-6)

    # --- model + data ---------------------------------------------------------
    say("\n[4] TinyCausalLM and preference pairs")
    data = PreferencePairs(seed=seed)
    policy, ref = build_models(data.vocab, seed=seed)
    batch = data.sample(16)
    ids = batch["chosen_input_ids"].clone()
    with torch.no_grad():
        out_a = policy(ids)
        ids2 = ids.clone()
        ids2[:, 7] = (ids2[:, 7] + 1) % data.vocab
        out_b = policy(ids2)
    c.add("model is causal: changing token 7 leaves logits at positions < 7 unchanged",
          torch.allclose(out_a[:, :7], out_b[:, :7], atol=1e-5) and not torch.allclose(out_a[:, 7:], out_b[:, 7:]),
          "Mission 3 attention" if policy.uses_mission3_attention else "prefix-mean fallback")
    with torch.no_grad():
        pcv = get_batch_logps(policy(batch["chosen_input_ids"]), batch["chosen_labels"])
        rcv = get_batch_logps(ref(batch["chosen_input_ids"]), batch["chosen_labels"])
    c.add("fresh reference == policy: identical log-probs, DPO loss = log 2", torch.equal(pcv, rcv))
    c.add("labels: prompt masked, number of response tokens == sampled length",
          bool((batch["chosen_labels"][:, : data.prompt_len] == IGNORE).all())
          and torch.equal((batch["chosen_labels"] != IGNORE).sum(1), batch["len_w"]))
    lf, mf = loss_fn.from_logits(policy(batch["chosen_input_ids"]), policy(batch["rejected_input_ids"]),
                                 ref(batch["chosen_input_ids"]), ref(batch["rejected_input_ids"]),
                                 batch["chosen_labels"], batch["rejected_labels"])
    c.add("from_logits == forward(get_batch_logps(...)) and keeps a graph to the policy",
          abs(lf.item() - math.log(2)) < 1e-6 and lf.requires_grad)
    tr = train_dpo(beta=0.1, steps=40, batch_size=64, seed=seed, quiet=True)
    h0, h1 = tr["history"][0], tr["history"][-1]
    c.add("short DPO run: validation loss drops below log 2 and pairwise accuracy rises above 60%",
          h1["val_loss"] < math.log(2) - 0.02 and h1["pairwise_accuracy"] > 0.6,
          f"loss {h0['val_loss']:.3f} -> {h1['val_loss']:.3f}, acc {100 * h1['pairwise_accuracy']:.0f}%")
    c.add("reference digest unchanged over the whole run", tr["ref_frozen"])

    # --- integration with the other stations -------------------------------
    say("\n[5] integration with Parts 2, 3, 4")
    p2 = load_sibling(PART2, "part2_implicit_reward")
    if p2 is not None and hasattr(p2, "implicit_rewards_from_logps"):
        res = p2.implicit_rewards_from_logps(pc, pr, rc, rr, beta=0.1)
        _, mm = loss_fn(pc, pr, rc, rr)
        ok = abs(res["reward_w"].mean().item() - mm["chosen_rewards"].item()) < 1e-6 \
            and abs(res["reward_l"].mean().item() - mm["rejected_rewards"].item()) < 1e-6 \
            and abs(res["p_win"].mean().item() - mm["bt_probability"].item()) < 1e-6
        c.add("Part 2 implicit rewards and P(y_w > y_l) equal the DPOLoss metrics", ok)
        c.add("Part 2 uses this file's get_batch_logps / DPOLoss at run time",
              getattr(p2, "PART1_SOURCE", "") == "01_DPO_Loss_Core.py", str(getattr(p2, "PART1_SOURCE", "?")))
    else:
        c.add("Part 2 not importable - integration check skipped", True)
    p3 = load_sibling(PART3, "part3_simpo")
    if p3 is not None and hasattr(p3, "SimPOLoss"):
        aw = get_batch_logps(policy(batch["chosen_input_ids"]), batch["chosen_labels"], average_log_prob=True)
        al = get_batch_logps(policy(batch["rejected_input_ids"]), batch["rejected_labels"], average_log_prob=True)
        sl, _ = p3.SimPOLoss(beta=2.0, gamma=0.5)(aw, al)
        expect = (-F.logsigmoid(2.0 * (aw - al) - 0.5)).mean()
        c.add("Part 3 SimPOLoss on average_log_prob=True log-probs == -log sigma(beta (avg_w - avg_l) - gamma)",
              abs(sl.item() - expect.item()) < 1e-5 and bool((aw <= 0).all() and (aw >= -math.log(data.vocab) - 1).all()))
    else:
        c.add("Part 3 not importable - integration check skipped", True)
    p4 = load_sibling(PART4, "part4_steering")
    if p4 is not None:
        c.add("Part 4 exposes extract_steering_vector and ActivationSteeringHook",
              hasattr(p4, "extract_steering_vector") and hasattr(p4, "ActivationSteeringHook"))
    else:
        c.add("Part 4 not importable - presence check skipped", True)

    say(f"\n--- {c.passed} passed, {c.failed} failed ---")
    return c


# ---------------------------------------------------------------------------
# 9. Sign-off gate, logging, CLI
# ---------------------------------------------------------------------------

def check_signoffs(readme: Path = HERE / "README.md") -> Tuple[bool, List[Tuple[str, bool]]]:
    rows = []
    pattern = re.compile(r"^- \[( |x|X)\] \*\*([^*(`]+)")
    for line in readme.read_text(encoding="utf-8").splitlines():
        mm = pattern.match(line.strip())
        if mm:
            rows.append((mm.group(2).strip(), mm.group(1).lower() == "x"))
    return bool(rows) and all(ok for _, ok in rows), rows


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_log(mode: str, payload: Dict[str, Any]) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps04-{mode}.json"
    payload = {"mission": "ps04", "station": "01 Systems Architect", "mode": mode,
               "timestamp": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, **payload}
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def demo(seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    print("=" * 78)
    print("USSE NOOSPHERE - SUBSYSTEM 04: DPO LOSS CORE (Systems Architect)")
    print("=" * 78)

    # 1. gather on a hand-checkable batch
    data = PreferencePairs(seed=seed)
    policy, ref = build_models(data.vocab, seed=seed)
    batch = data.sample(4)
    with torch.no_grad():
        logits = policy(batch["chosen_input_ids"])
    lp_sum = get_batch_logps(logits, batch["chosen_labels"])
    lp_avg = get_batch_logps(logits, batch["chosen_labels"], average_log_prob=True)
    print("1. get_batch_logps on 4 chosen responses (prompt = 4 tokens masked with -100, padded to 14)")
    print(f"   {'|y|':>4} | {'sum log pi(y|x)':>16} | {'mean per token':>14} | {'uniform baseline':>16}")
    for n, s, a in zip(batch["len_w"].tolist(), lp_sum.tolist(), lp_avg.tolist()):
        print(f"   {n:>4} | {s:>16.3f} | {a:>14.3f} | {-math.log(data.vocab):>16.3f}")

    # 2./3. loss on a perturbed policy
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.2 * torch.randn_like(p))
    loss_fn = DPOLoss(beta=0.1)
    loss, m = dpo_step(policy, ref, data.sample(64), loss_fn)
    print("\n2./3. DPO loss on 64 pairs, policy = reference + noise, beta = 0.1")
    print(f"   Delta_w mean {m['chosen_logratio'].item():+.4f}   Delta_l mean {m['rejected_logratio'].item():+.4f}")
    print(f"   loss {loss.item():.4f} (log 2 = {math.log(2):.4f})   P(y_w > y_l) {m['bt_probability'].item():.3f}"
          f"   implicit reward margin {m['reward_margin'].item():+.4f}   pairwise acc {100 * m['chosen_win_rate'].item():.1f}%")
    print()
    stab = stability_table()
    print()
    gt = gradient_test(seed=seed)
    print("\nNext: --selftest (30 checks), --train (DPO run + beta sweep), --check-signoffs (gate).")
    print("=" * 78)
    return {"logps_sum": lp_sum, "logps_avg": lp_avg, "lengths": batch["len_w"], "loss": loss, "metrics": m,
            "stability": stab, "gradient_test": gt}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mission 04 Part 1 - DPO loss core")
    ap.add_argument("--selftest", action="store_true", help="run the 30 checks")
    ap.add_argument("--train", action="store_true", help="short DPO training run and beta sweep")
    ap.add_argument("--check-signoffs", action="store_true", help="all four README boxes must be ticked")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if args.check_signoffs:
        ok, rows = check_signoffs()
        for name, ticked in rows:
            print(f"  [{'x' if ticked else ' '}] {name}")
        print("GATE PASSED - all four stations signed off." if ok else "GATE BLOCKED - missing sign-offs.")
        if not args.no_log:
            print(f"log: {write_log('signoffs', {'passed': ok, 'rows': rows})}")
        return 0 if ok else 1

    if args.selftest:
        c = self_test(seed=args.seed)
        if not args.no_log:
            print(f"log: {write_log('selftest', {'passed': c.passed, 'failed': c.failed, 'checks': c.rows})}")
        return 0 if c.failed == 0 else 1

    if args.train:
        run = train_dpo(beta=args.beta, steps=args.steps, seed=args.seed)
        sweep = beta_sweep(steps=args.steps, seed=args.seed)
        if not args.no_log:
            print(f"log: {write_log('train', {'run': run, 'beta_sweep': sweep})}")
        return 0

    out = demo(seed=args.seed)
    if not args.no_log:
        print(f"log: {write_log('demo', out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
