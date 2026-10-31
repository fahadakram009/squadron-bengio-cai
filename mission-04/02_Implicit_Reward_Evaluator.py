"""
Mission 04 - Part 2: Communications & Protocol Officer (Madhu)
Implicit Reward Extraction & Bradley-Terry Evaluation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-04-ORI (Betelgeuse)
Subsystem 04: Plasma Ignition & Ethical Steering Core

What is in this file (ps04_briefing.md, Part 2):
  1. evaluate_implicit_rewards(policy, ref, batch): implicit rewards r(x, y_w) and r(x, y_l) for validation pairs
  2. pairwise_accuracy(): Accuracy = 1/N * sum_i 1[r(x_i, y_w,i) > r(x_i, y_l,i)]  (= pairwise win-rate),
     plus the margin separation delta_r = r(x, y_w) - r(x, y_l) (mean, std, mean/std)
  3. reliability diagram: predicted Bradley-Terry probability sigma(r_w - r_l) vs. how often the human
     annotators actually preferred y_w, per probability bin -> figures/bt_calibration.png

Formulas (DPO, Rafailov et al. 2023):
    r(x, y)        = beta * (log pi_theta(y|x) - log pi_ref(y|x))
    P(y_w > y_l|x) = sigma(r(x, y_w) - r(x, y_l))          (Bradley-Terry)
The log Z(x) term of the DPO derivation only cancels if pi_ref stays exactly the same the whole time,
so the reference model is frozen here and the self-test checks that its weights never change.

Why there is a small "preference world" in here
-----------------------------------------------
To check calibration you need human labels that are NOT all "y_w wins". If the chosen answer is always put
first (like in the starter demo), every label is 1 and the reliability diagram can only say "predict 1".
So I generate a toy dataset where I know the true preference:
  - prompts have a topic, responses are short token sequences
  - a hidden utility table says how much humans like each token for each topic
  - 5 simulated annotators vote on every pair with P(a wins) = sigma(kappa * (u(a) - u(b)))  (Bradley-Terry)
  - the majority vote decides which answer is "chosen" for DPO training
Then a tiny language model is trained with DPO (Part 1's DPOLoss) and the implicit rewards are checked
against the annotators' vote shares on held-out pairs, before and during training.

Part 1 (01_DPO_Loss_Core.py) is loaded at run time for get_batch_logps and DPOLoss. If it can't be loaded,
the small copies in this file are used instead (the self-test checks that both give the same numbers).

How to run:
    python 02_Implicit_Reward_Evaluator.py                 # everything (self-test, training, calibration, figure)
    python 02_Implicit_Reward_Evaluator.py --selftest      # only the tests
    python 02_Implicit_Reward_Evaluator.py --epochs 20     # train longer
A json log is saved in logs/ (use --no-log to skip it).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DPO_FILE = HERE / "01_DPO_Loss_Core.py"
DEFAULT_FIGURE = HERE / "figures" / "bt_calibration.png"
IGNORE = -100   # label for prompt / padding positions (same as Part 1)


# ---------------------------------------------------------------------------
# 0. Part 1 functions (loaded from 01_DPO_Loss_Core.py if possible)
# ---------------------------------------------------------------------------

def local_get_batch_logps(logits: torch.Tensor, labels: torch.Tensor,
                          label_pad_token_id: int = IGNORE, average_log_prob: bool = False) -> torch.Tensor:
    """Sum (or mean) of log p(token) over the response tokens. Same convention as Part 1:
    logits[:, t] predicts labels[:, t+1], positions with label -100 are ignored."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != label_pad_token_id
    safe = torch.where(mask, labels, torch.zeros_like(labels))
    logp = torch.gather(F.log_softmax(logits, dim=-1), 2, safe.unsqueeze(2)).squeeze(2)
    logp = logp * mask.float()
    if average_log_prob:
        return logp.sum(-1) / mask.sum(-1).clamp(min=1)
    return logp.sum(-1)


class LocalDPOLoss(nn.Module):
    """-log sigmoid(beta * (delta_w - delta_l)), same interface as Part 1's DPOLoss."""

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, pc, pr, rc, rr):
        logits = self.beta * ((pc - rc) - (pr - rr))
        loss = -F.logsigmoid(logits).mean()
        return loss, {"loss": loss.detach()}


def load_sibling(path: Path, name: str):
    """Import a teammate's script (file names start with a digit, so normal import does not work)."""
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"  (could not import {path.name}: {type(exc).__name__}: {exc})")
        return None


_part1 = load_sibling(DPO_FILE, "dpo_core")
if _part1 is not None and hasattr(_part1, "get_batch_logps") and hasattr(_part1, "DPOLoss"):
    get_batch_logps, DPOLoss, PART1_SOURCE = _part1.get_batch_logps, _part1.DPOLoss, DPO_FILE.name
else:
    get_batch_logps, DPOLoss, PART1_SOURCE = local_get_batch_logps, LocalDPOLoss, "local copy"


# ---------------------------------------------------------------------------
# 1. implicit rewards + pairwise accuracy
# ---------------------------------------------------------------------------

def implicit_rewards_from_logps(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                                ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
                                beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """r = beta * (log pi - log pi_ref) for chosen and rejected, plus the Bradley-Terry probability."""
    reward_w = beta * (policy_chosen_logps - ref_chosen_logps)
    reward_l = beta * (policy_rejected_logps - ref_rejected_logps)
    delta_r = reward_w - reward_l
    return {"reward_w": reward_w, "reward_l": reward_l, "delta_r": delta_r, "p_win": torch.sigmoid(delta_r)}


def pairwise_accuracy(reward_w: torch.Tensor, reward_l: torch.Tensor) -> Tuple[float, int]:
    """Accuracy = 1/N sum 1[r_w > r_l] (strict, like the formula in the brief). Also returns the number of ties,
    because at the start of training policy == ref and every reward is exactly 0."""
    acc = (reward_w > reward_l).float().mean().item()
    ties = int((reward_w == reward_l).sum().item())
    return acc, ties


@torch.no_grad()
def sequence_logps(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """log pi(y|x) for every sequence, in mini-batches."""
    out = []
    for i in range(0, input_ids.shape[0], batch_size):
        out.append(get_batch_logps(model(input_ids[i:i + batch_size]), labels[i:i + batch_size]))
    return torch.cat(out)


def evaluate_implicit_rewards(policy: nn.Module, ref: nn.Module, batch: Dict[str, torch.Tensor],
                              beta: float = 0.1) -> Dict[str, Any]:
    """Implicit rewards of the chosen / rejected answers on a validation batch.

    batch needs: chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels   (labels -100 = ignore)
    Returns tensors reward_w, reward_l, delta_r, p_win and the numbers pairwise_accuracy, ties, mean_margin,
    mean_p_win. The models are put in eval mode for this and switched back afterwards.
    """
    modes = (policy.training, ref.training)
    policy.eval()
    ref.eval()
    try:
        pc = sequence_logps(policy, batch["chosen_input_ids"], batch["chosen_labels"])
        pr = sequence_logps(policy, batch["rejected_input_ids"], batch["rejected_labels"])
        rc = sequence_logps(ref, batch["chosen_input_ids"], batch["chosen_labels"])
        rr = sequence_logps(ref, batch["rejected_input_ids"], batch["rejected_labels"])
    finally:
        policy.train(modes[0])
        ref.train(modes[1])
    res: Dict[str, Any] = implicit_rewards_from_logps(pc, pr, rc, rr, beta)
    acc, ties = pairwise_accuracy(res["reward_w"], res["reward_l"])
    res.update({"pairwise_accuracy": acc, "ties": ties, "mean_p_win": res["p_win"].mean().item()})
    res.update(margin_separation(res["reward_w"], res["reward_l"]))
    return res


def margin_separation(reward_w: torch.Tensor, reward_l: torch.Tensor) -> Dict[str, float]:
    """How well the rewards separate chosen from rejected: mean / std of delta_r = r_w - r_l.
    separation = mean / std (like a signal-to-noise ratio, 0 = no separation)."""
    d = (reward_w - reward_l).double()
    std = d.std().item() if d.numel() > 1 else 0.0
    return {"mean_reward_w": reward_w.double().mean().item(), "mean_reward_l": reward_l.double().mean().item(),
            "mean_margin": d.mean().item(), "std_margin": std,
            "separation": d.mean().item() / std if std > 0 else 0.0}


# ---------------------------------------------------------------------------
# 2. calibration
# ---------------------------------------------------------------------------

def compute_calibration_curve(predicted_probs: Sequence[float], ground_truth_labels: Sequence[float],
                              num_bins: int = 10) -> List[Dict[str, Any]]:
    """Reliability curve. Labels can be 0/1 or a vote share in [0, 1] (fraction of annotators preferring y_w).
    Equal-width bins, the last bin includes p = 1.0."""
    p = torch.as_tensor(predicted_probs, dtype=torch.float64).flatten()
    y = torch.as_tensor(ground_truth_labels, dtype=torch.float64).flatten()
    idx = (p * num_bins).floor().clamp(0, num_bins - 1).long()
    curve = []
    for b in range(num_bins):
        sel = idx == b
        n = int(sel.sum().item())
        lo, hi = b / num_bins, (b + 1) / num_bins
        if n == 0:
            curve.append({"bin": f"[{lo:.2f}, {hi:.2f})", "bin_lo": lo, "bin_hi": hi, "count": 0,
                          "avg_predicted_prob": (lo + hi) / 2, "empirical_accuracy": None, "calibration_error": None})
            continue
        avg_p = p[sel].mean().item()
        emp = y[sel].mean().item()
        curve.append({"bin": f"[{lo:.2f}, {hi:.2f})", "bin_lo": lo, "bin_hi": hi, "count": n,
                      "avg_predicted_prob": avg_p, "empirical_accuracy": emp, "calibration_error": abs(avg_p - emp)})
    return curve


def expected_calibration_error(curve: List[Dict[str, Any]]) -> float:
    """ECE = sum over bins of (bin size / N) * |mean predicted - mean empirical|. Empty bins are skipped."""
    total = sum(c["count"] for c in curve)
    if total == 0:
        return float("nan")
    return sum(c["count"] / total * c["calibration_error"] for c in curve if c["count"] > 0)


def brier_score(p: torch.Tensor, y: torch.Tensor) -> float:
    return ((p.double() - y.double()) ** 2).mean().item()


def soft_nll(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Cross-entropy against the vote share y (log loss in nats)."""
    p = p.double().clamp(eps, 1 - eps)
    y = y.double()
    return (-(y * p.log() + (1 - y) * (1 - p).log())).mean().item()


def symmetrize(p_win: torch.Tensor, vote_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use every pair in both orders: (P(w>l), share for w) and (P(l>w), share for l) = (1-p, 1-v).
    Otherwise the empirical side is always >= 0.5 (the chosen answer is the majority one) and half of the
    diagram stays empty. The ECE of the symmetric version is the same as for one fixed order."""
    return torch.cat([p_win, 1 - p_win]), torch.cat([vote_w, 1 - vote_w])


def fit_temperature(delta_r: torch.Tensor, vote_w: torch.Tensor) -> float:
    """Temperature scaling: find tau so that sigma(delta_r / tau) best fits the votes (grid search on log tau).
    tau > 1 means the rewards are overconfident; beta / tau is the 'calibrated beta'."""
    best_tau, best = 1.0, float("inf")
    for k in range(-200, 201):
        tau = 10 ** (k / 100)
        nll = soft_nll(torch.sigmoid(delta_r.double() / tau), vote_w)
        if nll < best:
            best_tau, best = tau, nll
    return best_tau


def calibration_summary(p_win: torch.Tensor, vote_w: torch.Tensor, num_bins: int = 10) -> Dict[str, Any]:
    ps, vs = symmetrize(p_win, vote_w)
    curve = compute_calibration_curve(ps.tolist(), vs.tolist(), num_bins)
    return {"curve": curve, "ece": expected_calibration_error(curve), "brier": brier_score(p_win, vote_w),
            "nll": soft_nll(p_win, vote_w)}


# ---------------------------------------------------------------------------
# 3. toy preference world + tiny language model
# ---------------------------------------------------------------------------

PAD, BOS = 0, 1


class PreferenceWorld:
    """Prompts = [BOS, topic, filler, filler], responses = 4..10 content tokens.
    Human utility of a response = mean over its tokens of utility[topic, token]."""

    def __init__(self, num_topics: int = 4, num_content: int = 26, kappa: float = 3.0,
                 annotators: int = 5, seed: int = 0):
        self.g = torch.Generator().manual_seed(seed)
        self.num_topics, self.num_content = num_topics, num_content
        self.topic0 = 2
        self.content0 = 2 + num_topics
        self.vocab = self.content0 + num_content
        self.kappa, self.annotators = kappa, annotators
        self.utility = torch.randn(num_topics, num_content, generator=self.g)
        self.prompt_len, self.min_len, self.max_len = 4, 4, 10

    def true_utility(self, topic: torch.Tensor, resp: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        u = self.utility[topic.unsqueeze(1).expand_as(resp), (resp - self.content0).clamp(min=0)]
        mask = torch.arange(0, resp.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        return (u * mask.float()).sum(1) / lengths.float()

    def sample_responses(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.randint(self.min_len, self.max_len + 1, (n,), generator=self.g)
        resp = torch.randint(self.content0, self.vocab, (n, self.max_len), generator=self.g)
        return resp, lengths

    def to_sequences(self, prompts: torch.Tensor, resp: torch.Tensor, lengths: torch.Tensor):
        """Concatenate prompt + response, pad with PAD, labels = -100 on prompt and padding."""
        ids = torch.cat([prompts, resp], dim=1)
        pos = torch.arange(0, self.max_len).unsqueeze(0)
        pad = pos >= lengths.unsqueeze(1)
        ids[:, self.prompt_len:] = torch.where(pad, torch.full_like(resp, PAD), resp)
        labels = ids.clone()
        labels[:, :self.prompt_len] = IGNORE
        labels[:, self.prompt_len:] = torch.where(pad, torch.full_like(resp, IGNORE), resp)
        return ids, labels

    def make_pairs(self, n: int) -> Dict[str, torch.Tensor]:
        """n prompts, two responses each, 5 annotator votes. y_w = majority winner."""
        topic = torch.randint(0, self.num_topics, (n,), generator=self.g)
        filler = torch.randint(self.content0, self.vocab, (n, 2), generator=self.g)
        prompts = torch.cat([torch.full((n, 1), BOS, dtype=torch.long), (topic + self.topic0).unsqueeze(1), filler], 1)
        ra, la = self.sample_responses(n)
        rb, lb = self.sample_responses(n)
        ua, ub = self.true_utility(topic, ra, la), self.true_utility(topic, rb, lb)
        p_true = torch.sigmoid(self.kappa * (ua - ub))                       # true P(a > b)
        votes = (torch.rand(n, self.annotators, generator=self.g) < p_true.unsqueeze(1)).float()
        share_a = votes.mean(1)
        a_wins = share_a > 0.5                                                 # odd number of annotators, no ties
        ids_a, lab_a = self.to_sequences(prompts, ra, la)
        ids_b, lab_b = self.to_sequences(prompts, rb, lb)
        w = a_wins.unsqueeze(1)
        return {
            "chosen_input_ids": torch.where(w, ids_a, ids_b), "chosen_labels": torch.where(w, lab_a, lab_b),
            "rejected_input_ids": torch.where(w, ids_b, ids_a), "rejected_labels": torch.where(w, lab_b, lab_a),
            "vote_w": torch.where(a_wins, share_a, 1 - share_a),              # share of annotators preferring y_w
            "p_true_w": torch.where(a_wins, p_true, 1 - p_true),              # Bradley-Terry truth for y_w
            "len_w": torch.where(a_wins, la, lb), "len_l": torch.where(a_wins, lb, la),
        }


class TinyPolicyLM(nn.Module):
    """Very small LM: next token depends on the current token and the topic token (position 1).
    logits_t = W tanh(E[x_t] + C[x_1]) + b"""

    def __init__(self, vocab: int, dim: int = 32):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.ctx = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.tok(input_ids) + self.ctx(input_ids[:, 1]).unsqueeze(1))
        return self.out(h)


# ---------------------------------------------------------------------------
# 4. DPO training with checkpoints
# ---------------------------------------------------------------------------

def checkpoint_report(policy, ref, val, beta) -> Dict[str, Any]:
    ev = evaluate_implicit_rewards(policy, ref, val, beta)
    cal = calibration_summary(ev["p_win"], val["vote_w"])
    return {"pairwise_accuracy": ev["pairwise_accuracy"], "ties": ev["ties"], "mean_margin": ev["mean_margin"],
            "mean_p_win": ev["mean_p_win"], "separation": ev["separation"],
            "mean_reward_w": ev["mean_reward_w"], "mean_reward_l": ev["mean_reward_l"], "ece": cal["ece"], "brier": cal["brier"], "nll": cal["nll"],
            "curve": cal["curve"], "_p_win": ev["p_win"], "_delta_r": ev["delta_r"]}


def train_dpo(policy: nn.Module, ref: nn.Module, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor],
              epochs: int = 12, beta: float = 0.1, lr: float = 0.05, batch_size: int = 64, seed: int = 0,
              checkpoints: Sequence[int] = (0, 1, 3), quiet: bool = False) -> Dict[str, Any]:
    say = (lambda *a: None) if quiet else print
    g = torch.Generator().manual_seed(seed)
    loss_fn = DPOLoss(beta=beta)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    # reference is frozen, so its log-probs only need to be computed once
    ref_c = sequence_logps(ref, train["chosen_input_ids"], train["chosen_labels"])
    ref_r = sequence_logps(ref, train["rejected_input_ids"], train["rejected_labels"])
    n = train["chosen_input_ids"].shape[0]
    history, snaps = [], {}

    def record(epoch, train_loss):
        rep = checkpoint_report(policy, ref, val, beta)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{k: rep[k] for k in ("pairwise_accuracy", "ties", "mean_margin", "separation", "mean_reward_w",
                                                  "mean_reward_l", "ece", "brier", "nll")}})
        if epoch in checkpoints or epoch == epochs:
            snaps[epoch] = rep
        say(f"  epoch {epoch:>3} | train loss {train_loss:.4f} | val acc {rep['pairwise_accuracy'] * 100:5.1f}% "
            f"(ties {rep['ties']}) | margin {rep['mean_margin']:+.3f} (sep {rep['separation']:.2f}) | ECE {rep['ece']:.3f} | "
            f"Brier {rep['brier']:.4f} | NLL {rep['nll']:.4f}")

    record(0, math.log(2))   # policy == ref -> every logit is 0 -> loss = log 2
    policy.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pc = get_batch_logps(policy(train["chosen_input_ids"][idx]), train["chosen_labels"][idx])
            pr = get_batch_logps(policy(train["rejected_input_ids"][idx]), train["rejected_labels"][idx])
            out = loss_fn(pc, pr, ref_c[idx], ref_r[idx])
            loss = out[0] if isinstance(out, tuple) else out   # Part 1 returns (loss, metrics)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
            count += idx.shape[0]
        record(epoch, total / count)
    return {"history": history, "snapshots": snaps}


# ---------------------------------------------------------------------------
# 5. experiment + figure
# ---------------------------------------------------------------------------

def run_experiment(seed: int = 0, epochs: int = 12, beta: float = 0.1, n_train: int = 2000, n_val: int = 1000,
                   quiet: bool = False) -> Dict[str, Any]:
    say = (lambda *a: None) if quiet else print
    torch.manual_seed(seed)
    world = PreferenceWorld(seed=seed)
    train, val = world.make_pairs(n_train), world.make_pairs(n_val)
    ref = TinyPolicyLM(world.vocab)
    ref.requires_grad_(False)
    policy = copy.deepcopy(ref)
    policy.requires_grad_(True)
    ref_before = [p.detach().clone() for p in ref.parameters()]

    agree = val["vote_w"].mean().item()
    oracle = calibration_summary(val["p_true_w"], val["vote_w"])
    oracle_acc = (val["p_true_w"] > 0.5).float().mean().item()   # true BT ranking agrees with the majority
    say(f"=== Preference data: {n_train} train / {n_val} val pairs, {world.annotators} annotators, "
        f"kappa={world.kappa} ===")
    say(f"  annotators agree with the majority on {agree * 100:.1f}% of votes; "
        f"true Bradley-Terry P(y_w > y_l) averages {val['p_true_w'].mean().item() * 100:.1f}%")
    say(f"  oracle (true BT probabilities): accuracy {oracle_acc * 100:.1f}%, ECE {oracle['ece']:.3f}, "
        f"Brier {oracle['brier']:.4f}")
    say("  -> about the best any model can reach, because 5 noisy votes sometimes pick the worse answer")
    say(f"  DPO loss / log-probs from: {PART1_SOURCE}\n")
    say(f"=== DPO training, beta={beta}, evaluated on the validation pairs ===")
    run = train_dpo(policy, ref, train, val, epochs=epochs, beta=beta, seed=seed,
                    checkpoints=(0, 1, 3), quiet=quiet)

    ref_frozen = all(torch.equal(a, b) for a, b in zip(ref_before, ref.parameters()))
    final = run["snapshots"][epochs]
    # temperature scaling: fit on the first half of the validation pairs, check on the second half
    half = n_val // 2
    tau = fit_temperature(final["_delta_r"][:half], val["vote_w"][:half])
    raw_half = calibration_summary(final["_p_win"][half:], val["vote_w"][half:])
    scaled_half = calibration_summary(torch.sigmoid(final["_delta_r"][half:] / tau), val["vote_w"][half:])
    say(f"\n=== Temperature scaling of the final rewards (fit on {half} pairs, tested on {n_val - half}) ===")
    say(f"  tau = {tau:.2f}  -> calibrated beta = beta / tau = {beta / tau:.3f}")
    say(f"  held-out ECE {raw_half['ece']:.3f} -> {scaled_half['ece']:.3f}, "
        f"NLL {raw_half['nll']:.4f} -> {scaled_half['nll']:.4f}")
    say(f"  reference model unchanged after training: {ref_frozen}\n")

    say(f"=== Bradley-Terry reliability table (final epoch {epochs}, both orders of every pair) ===")
    say(f"  {'bin':<14}{'count':>7}{'pred P(y_w>y_l)':>18}{'human share':>14}{'|gap|':>8}")
    for c in final["curve"]:
        if c["count"]:
            say(f"  {c['bin']:<14}{c['count']:>7}{c['avg_predicted_prob']:>18.3f}{c['empirical_accuracy']:>14.3f}"
                f"{c['calibration_error']:>8.3f}")
    say("")

    hist = run["history"]
    best = max(hist[1:], key=lambda r: r["pairwise_accuracy"])
    best_nll = min(hist[1:], key=lambda r: r["nll"])
    say("=== Summary ===")
    say(f"  - pairwise accuracy: {best['pairwise_accuracy'] * 100:.1f}% at best (epoch {best['epoch']}), "
        f"{hist[-1]['pairwise_accuracy'] * 100:.1f}% at epoch {epochs}; oracle ceiling {oracle_acc * 100:.1f}%")
    fin = hist[-1]
    say(f"  - margin separation at epoch {epochs}: r(y_w) {fin['mean_reward_w']:+.3f} vs r(y_l) "
        f"{fin['mean_reward_l']:+.3f}, delta_r {fin['mean_margin']:+.3f} (mean/std {fin['separation']:.2f})")
    say("  - epoch 0 has ECE 0 only because every prediction is exactly 0.5 (calibrated but useless);")
    say("    Brier / NLL show the real improvement after training")
    say(f"  - lowest validation NLL at epoch {best_nll['epoch']}; after that the reward margins keep growing,")
    say(f"    ECE rises to {hist[-1]['ece']:.3f} and the predictions are overconfident (curve flatter than the diagonal)")
    say(f"  - dividing the rewards by tau = {tau:.2f} (i.e. beta {beta} -> {beta / tau:.3f}) fixes most of it: "
        f"held-out ECE {raw_half['ece']:.3f} -> {scaled_half['ece']:.3f}")
    say("  - so implicit DPO rewards rank answers well, but their Bradley-Terry probabilities should not be")
    say("    read as real win rates without checking calibration (or early stopping on validation NLL)\n")
    return {"world": {"n_train": n_train, "n_val": n_val, "annotators": world.annotators, "kappa": world.kappa,
                      "annotator_agreement": agree},
            "oracle": {"accuracy": oracle_acc, **{k: oracle[k] for k in ("ece", "brier", "nll", "curve")}},
            "best_epoch_accuracy": best["epoch"], "best_epoch_nll": best_nll["epoch"],
            "history": run["history"], "snapshots": run["snapshots"],
            "temperature": {"tau": tau, "calibrated_beta": beta / tau, "ece_before": raw_half["ece"],
                            "ece_after": scaled_half["ece"], "nll_before": raw_half["nll"],
                            "nll_after": scaled_half["nll"], "curve_after": scaled_half["curve"],
                            "curve_before": raw_half["curve"]},
            "ref_frozen": ref_frozen, "beta": beta, "epochs": epochs, "part1_source": PART1_SOURCE}


def plot_calibration(res: Dict[str, Any], path: Path = DEFAULT_FIGURE) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed - `pip install matplotlib` to draw the reliability diagram.")
        return None

    def xy(curve):
        pts = [(c["avg_predicted_prob"], c["empirical_accuracy"]) for c in curve if c["count"]]
        return [p[0] for p in pts], [p[1] for p in pts]

    snaps = res["snapshots"]
    epochs = sorted(snaps)
    palette = ["#8c959f", "#54aeff", "#0969da", "#cf222e", "#8250df", "#bf8700"]
    color = {e: palette[i % len(palette)] for i, e in enumerate(epochs)}
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    a = ax[0, 0]
    a.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="perfect calibration")
    ox, oy = xy(res["oracle"]["curve"])
    a.plot(ox, oy, color="#1a7f37", lw=1.2, ls=":", marker="s", ms=4,
           label=f"oracle (true BT probability): ECE {res['oracle']['ece']:.3f}")
    for e in epochs:
        x, y = xy(snaps[e]["curve"])
        a.plot(x, y, marker="o", color=color[e], lw=1.6, label=f"epoch {e}: ECE {snaps[e]['ece']:.3f}")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xlabel("predicted  P(y_w > y_l) = sigma(r_w - r_l)")
    a.set_ylabel("share of annotators preferring y_w")
    a.set_title("(a) Bradley-Terry reliability diagram during DPO training")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper left")

    b = ax[0, 1]
    h = res["history"]
    ep = [r["epoch"] for r in h]
    l1 = b.plot(ep[1:], [r["pairwise_accuracy"] * 100 for r in h[1:]], color="#0969da", marker="o", ms=3,
                label="pairwise accuracy (%)")
    b.axhline(res["oracle"]["accuracy"] * 100, color="#0969da", lw=0.8, ls=":")
    b.text(0.02, 0.03, "epoch 0: policy = ref, every pair is a tie (accuracy 0%)", transform=b.transAxes, fontsize=8)
    b.set_xlabel("epoch")
    b.set_ylabel("validation pairwise accuracy (%)", color="#0969da")
    b.tick_params(axis="y", colors="#0969da")
    b.grid(alpha=0.3)
    b2 = b.twinx()
    l2 = b2.plot(ep, [r["ece"] for r in h], color="#cf222e", marker="s", ms=3, label="ECE")
    l3 = b2.plot(ep, [r["brier"] for r in h], color="#bf8700", marker="^", ms=3, label="Brier score")
    b2.axhline(res["oracle"]["ece"], color="#cf222e", lw=0.8, ls=":")
    b2.axhline(res["oracle"]["brier"], color="#bf8700", lw=0.8, ls=":")
    b2.set_ylabel("ECE / Brier  (dotted lines = oracle)", color="#cf222e")
    b2.tick_params(axis="y", colors="#cf222e")
    lines = l1 + l2 + l3
    b.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    b.set_title("(b) Validation accuracy and calibration per epoch")

    c_ = ax[1, 0]
    bins = [i / 20 for i in range(21)]
    for e in epochs[1:]:
        c_.hist(snaps[e]["_p_win"].tolist(), bins=bins, alpha=0.45, color=color[e], label=f"epoch {e}")
    c_.set_xlabel("predicted  P(y_w > y_l)  (chosen answer first)")
    c_.set_ylabel("number of validation pairs")
    c_.set_title("(c) Predicted win probabilities spread out as reward margins grow")
    c_.grid(alpha=0.3)
    c_.legend(fontsize=8)

    d = ax[1, 1]
    t = res["temperature"]
    d.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
    x, y = xy(t["curve_before"])
    d.plot(x, y, marker="o", color="#cf222e", label=f"raw rewards: ECE {t['ece_before']:.3f}")
    x, y = xy(t["curve_after"])
    d.plot(x, y, marker="o", color="#1a7f37", label=f"divided by tau={t['tau']:.2f}: ECE {t['ece_after']:.3f}")
    d.set_xlim(0, 1)
    d.set_ylim(0, 1)
    d.set_xlabel("predicted  P(y_w > y_l)")
    d.set_ylabel("share of annotators preferring y_w")
    d.set_title(f"(d) Temperature scaling on held-out pairs (calibrated beta = {t['calibrated_beta']:.3f})")
    d.grid(alpha=0.3)
    d.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Mission 04 / Part 2 - implicit DPO rewards vs. human preferences "
                 f"(beta={res['beta']}, {res['world']['n_val']} validation pairs, "
                 f"{res['world']['annotators']} annotators each)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. self test
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self, quiet: bool = False):
        self.items: List[Dict[str, Any]] = []
        self.quiet = quiet

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        if not self.quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(not c["ok"] for c in self.items)

    @property
    def passed(self) -> int:
        return sum(c["ok"] for c in self.items)


def self_test(seed: int = 0, quiet: bool = False) -> Checks:
    c = Checks(quiet)
    say = (lambda *a: None) if quiet else print
    say("=== Part 2 self-test: implicit rewards & Bradley-Terry calibration ===")
    g = torch.Generator().manual_seed(seed)

    # reward formula on the starter's hand-made numbers
    p_w = torch.tensor([-10.2, -12.4, -8.1, -14.5])
    p_l = torch.tensor([-15.1, -13.0, -11.5, -13.8])
    r_w = torch.tensor([-12.0, -14.0, -10.0, -15.0])
    r_l = torch.tensor([-13.0, -13.5, -10.5, -14.0])
    res = implicit_rewards_from_logps(p_w, p_l, r_w, r_l, beta=0.1)
    exp_w = [0.1 * (a - b) for a, b in zip(p_w.tolist(), r_w.tolist())]
    exp_l = [0.1 * (a - b) for a, b in zip(p_l.tolist(), r_l.tolist())]
    ok = all(abs(x - y) < 1e-6 for x, y in zip(res["reward_w"].tolist() + res["reward_l"].tolist(), exp_w + exp_l))
    c.add("r(x, y) = beta * (log pi - log pi_ref) on hand-computed values", ok)
    pw = res["p_win"].tolist()
    ok = all(abs(pv - 1 / (1 + math.exp(-(a - b)))) < 1e-6 for pv, a, b in zip(pw, exp_w, exp_l))
    c.add("P(y_w > y_l) = sigma(r_w - r_l)", ok)
    swap = implicit_rewards_from_logps(p_l, p_w, r_l, r_w, beta=0.1)["p_win"]
    c.add("swapping y_w and y_l gives 1 - P  (Bradley-Terry symmetry)",
          (swap + res["p_win"] - 1).abs().max().item() < 1e-6)
    sep = margin_separation(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))
    c.add("margin separation: mean / std of delta_r", abs(sep["mean_margin"] - 2) < 1e-6
          and abs(sep["std_margin"] - 1) < 1e-6 and abs(sep["separation"] - 2) < 1e-6)
    acc, ties = pairwise_accuracy(torch.tensor([1.0, 0.0, 2.0, -1.0]), torch.tensor([0.0, 0.0, 3.0, -2.0]))
    c.add("pairwise accuracy is strict (ties count as wrong) and ties are reported", acc == 0.5 and ties == 1,
          f"acc {acc}, ties {ties}")

    # log-prob gather
    logits = torch.randn(3, 7, 11, generator=g)
    labels = torch.randint(0, 11, (3, 7), generator=g)
    labels[:, :3] = IGNORE
    labels[1, 5:] = IGNORE
    lp = local_get_batch_logps(logits, labels)
    ls = F.log_softmax(logits, dim=-1)
    manual = []
    for b in range(3):
        s = 0.0
        for t in range(1, 7):
            if labels[b, t].item() != IGNORE:
                s += ls[b, t - 1, labels[b, t].item()].item()
        manual.append(s)
    c.add("get_batch_logps == explicit loop over tokens (prompt / padding ignored)",
          max(abs(a - b) for a, b in zip(lp.tolist(), manual)) < 1e-4)
    if PART1_SOURCE != "local copy":
        diff = (get_batch_logps(logits, labels) - lp).abs().max().item()
        c.add(f"Part 1 get_batch_logps ({PART1_SOURCE}) gives the same numbers", diff < 1e-5, f"max diff {diff:.1e}")
    else:
        c.add("Part 1 not found, using the local copy of get_batch_logps", True)

    # evaluate_implicit_rewards on a real model
    world = PreferenceWorld(seed=seed)
    data = world.make_pairs(64)
    ref = TinyPolicyLM(world.vocab)
    policy = copy.deepcopy(ref)
    ev = evaluate_implicit_rewards(policy, ref, data, beta=0.1)
    c.add("policy == ref -> all rewards 0, P = 0.5, every pair is a tie",
          ev["reward_w"].abs().max().item() == 0 and abs(ev["mean_p_win"] - 0.5) < 1e-7 and ev["ties"] == 64)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    before = [p.detach().clone() for p in policy.parameters()]
    policy.train()
    ref.eval()
    ev = evaluate_implicit_rewards(policy, ref, data, beta=0.1)
    with torch.no_grad():
        pc = get_batch_logps(policy(data["chosen_input_ids"]), data["chosen_labels"])
        rc = get_batch_logps(ref(data["chosen_input_ids"]), data["chosen_labels"])
    c.add("evaluate_implicit_rewards matches beta * (log pi - log pi_ref) computed by hand",
          (ev["reward_w"] - 0.1 * (pc - rc)).abs().max().item() < 1e-5)
    c.add("evaluation does not change the weights and restores train / eval mode",
          all(torch.equal(a, b) for a, b in zip(before, policy.parameters())) and policy.training and not ref.training)
    c.add("rewards from evaluation carry no gradient", not ev["reward_w"].requires_grad)
    ev2 = evaluate_implicit_rewards(policy, ref, data, beta=0.2)
    c.add("rewards scale linearly with beta", (ev2["reward_w"] - 2 * ev["reward_w"]).abs().max().item() < 1e-5)

    # data checks
    ok_mask = bool((data["chosen_labels"][:, :world.prompt_len] == IGNORE).all().item())
    n_resp = (data["chosen_labels"] != IGNORE).sum(1)
    c.add("labels: prompt masked, number of response tokens == response length",
          ok_mask and torch.equal(n_resp, data["len_w"]))
    c.add("y_w is the majority vote (share for y_w > 0.5 on every pair)", bool((data["vote_w"] > 0.5).all().item()))

    # calibration helpers
    n = 40000
    p = torch.rand(n, generator=g)
    y = (torch.rand(n, generator=g) < p).float()
    ece_good = expected_calibration_error(compute_calibration_curve(p.tolist(), y.tolist(), 10))
    p_over = torch.where(p > 0.5, torch.full_like(p, 0.99), torch.full_like(p, 0.01))
    ece_bad = expected_calibration_error(compute_calibration_curve(p_over.tolist(), y.tolist(), 10))
    c.add("ECE ~ 0 for calibrated predictions, large for overconfident ones", ece_good < 0.02 and ece_bad > 0.2,
          f"{ece_good:.3f} vs {ece_bad:.3f}")
    curve = compute_calibration_curve([0.0, 0.05, 0.5, 0.999, 1.0], [0, 0, 1, 1, 1], 10)
    c.add("calibration bins: counts add up, p = 1.0 goes in the last bin",
          sum(b["count"] for b in curve) == 5 and curve[-1]["count"] == 2 and curve[0]["count"] == 2)
    starter_style = compute_calibration_curve([0.6, 0.7, 0.8], [1, 1, 1], 4)
    c.add("starter-style labels (all 1) only fill the top of the diagram -> need real votes",
          all(b["empirical_accuracy"] in (None, 1.0) for b in starter_style))
    ps, vs = symmetrize(torch.tensor([0.8, 0.3]), torch.tensor([0.6, 0.6]))
    c.add("symmetrize keeps the ECE the same as one fixed order",
          abs(expected_calibration_error(compute_calibration_curve(ps.tolist(), vs.tolist(), 10)) -
              expected_calibration_error(compute_calibration_curve([0.8, 0.3], [0.6, 0.6], 10))) < 1e-12)
    true_tau = 2.5
    d = torch.randn(20000, generator=g) * 3
    v = (torch.rand(20000, generator=g) < torch.sigmoid(d / true_tau)).float()
    tau = fit_temperature(d, v)
    c.add("fit_temperature recovers a known temperature", abs(tau - true_tau) < 0.25, f"true 2.5, fitted {tau:.2f}")

    # short training run: does DPO learn the human preferences and keep ref frozen?
    small = run_experiment(seed=seed, epochs=3, n_train=600, n_val=300, quiet=True)
    h = small["history"]
    c.add("DPO training raises validation accuracy above chance",
          h[-1]["pairwise_accuracy"] > 0.6,
          f"{h[0]['pairwise_accuracy'] * 100:.0f}% -> {h[-1]['pairwise_accuracy'] * 100:.0f}%")
    c.add("DPO training loss goes down from log 2", h[-1]["train_loss"] < math.log(2) - 0.01,
          f"{h[0]['train_loss']:.3f} -> {h[-1]['train_loss']:.3f}")
    c.add("reference model stays frozen during training", small["ref_frozen"])

    say(f"--- {c.passed} passed, {c.failed} failed ---\n")
    return c


# ---------------------------------------------------------------------------
# 7. command line
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def write_log(mode: str, payload: Dict) -> Path:
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps04-rewards-{mode}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mission 04 Part 2 - implicit rewards & Bradley-Terry calibration")
    p.add_argument("--selftest", action="store_true", help="run the tests only")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_FIGURE, help="figure path")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-log", action="store_true", help="do not write logs/*.json")
    args = p.parse_args(argv)

    print("=" * 78)
    print("USSE NOOSPHERE - IMPLICIT REWARD & BRADLEY-TERRY CALIBRATION ENGINE (Part 2)")
    print("=" * 78)
    checks = self_test(args.seed, quiet=args.quiet)
    payload: Dict[str, Any] = {"torch": torch.__version__, "checks": checks.items,
                               "passed": checks.passed, "failed": checks.failed}
    if not args.selftest:
        res = run_experiment(seed=args.seed, epochs=args.epochs, beta=args.beta, quiet=args.quiet)
        payload["experiment"] = res
        fig = plot_calibration(res, args.out)
        if fig is not None:
            payload["figure"] = str(fig)
            if not args.quiet:
                print(f"Reliability diagram written to {fig}")
    if not args.no_log:
        log = write_log("selftest" if args.selftest else "all", payload)
        if not args.quiet:
            print(f"Log written to {log.relative_to(HERE)}")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
