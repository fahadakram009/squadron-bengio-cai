"""
Mission 03: Multi-Head Self-Attention from first principles - Squadron Bengio solution
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Subsystem 03 (Spatial Attention Matrix & KV-Cache Transceiver)
Systems Architect: Fahad

Keeps the starter's public surface (class MultiHeadSelfAttention, forward(x, causal, attention_mask)
-> (output, attention_weights), verify_variance_preservation) and implements Part 1 of the briefing
in pure PyTorch, i.e. tensor algebra only - no nn.MultiheadAttention, no F.scaled_dot_product_attention
(both are used *only* as independent oracles inside the self-test):

1. Scaled dot-product attention   softmax(Q K^T / sqrt(d_k)) V written out step by step
   (scaled_dot_product_attention), with an explicit boolean mask and NaN-safe fully masked rows.
2. Learned projections            W_Q, W_K, W_V, W_O in R^{d_model x d_model} (nn.Linear, bias-free).
3. Head splitting                 (b, n, d_model) -> (b, n, h, d_head) -> (b, h, n, d_head) and the inverse
   merge, with a per-head loop in the self-test proving the vectorised version is exactly h independent heads.
4. Causal masking                 strictly lower-triangular boolean mask, -inf above the diagonal before
   softmax, cached per sequence length; combinable with a (batch, seq_len) key-padding mask.
5. Variance-preservation test     empirical check over d_k in {16, 64, 256, 1024} that unit-variance inputs
   give Var(q.k) = d_k unscaled and Var(q.k / sqrt(d_k)) = 1, plus the softmax-saturation consequence
   (attention entropy collapses to ~0 bits without the scale).

Integration points for the other stations (read at run time, skipped if the file is absent):
  * Part 2 (Protocol Officer, 02_RoPE_Positional_Engine_starter.py): forward(..., rope=fn) applies a rotary
    hook to Q and K after the head split; the self-test imports apply_rotary_emb/precompute_freqs_cis and
    verifies that attention weights are invariant to a global position shift.
  * Part 3 (Data & Payload Specialist, 03_GQA_and_KVCache_Benchmark_starter.py): with H_KV == H_Q a
    GQAAttention layer must equal this module; the self-test copies the weights across and checks it.

Usage
-----
  python 01_MultiHead_Attention_Core_starter.py                 # variance test + demo forward pass (starter behaviour)
  python 01_MultiHead_Attention_Core_starter.py --selftest      # full verification suite, offline, ~20 checks
  python 01_MultiHead_Attention_Core_starter.py --bench         # O(N^2) time/memory scaling of naive attention
  python 01_MultiHead_Attention_Core_starter.py --check-signoffs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROPE_FILE = HERE / "02_RoPE_Positional_Engine_starter.py"
GQA_FILE = HERE / "03_GQA_and_KVCache_Benchmark_starter.py"
README_FILE = HERE / "README.md"

RopeHook = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


# --------------------------------------------------------------------------------------------------
# 1. Scaled dot-product attention, written out
# --------------------------------------------------------------------------------------------------
def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask, True where a query may attend: key position <= query position."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, for tensors (..., n_q, d_k), (..., n_k, d_k), (..., n_k, d_v).

    mask: boolean, broadcastable to (..., n_q, n_k); True = may attend, False = -inf before the softmax.
    A query row that is masked everywhere would give softmax(-inf, ..., -inf) = NaN; such rows are returned
    as all-zero weights (and therefore a zero context vector) instead.
    Returns (context (..., n_q, d_v), weights (..., n_q, n_k)).
    """
    d_k = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale          # (..., n_q, n_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = torch.nan_to_num(weights, nan=0.0)                # fully masked rows -> 0, not NaN
    if dropout is not None:
        weights = dropout(weights)
    context = torch.matmul(weights, v)                              # (..., n_q, d_v)
    return context, weights


# --------------------------------------------------------------------------------------------------
# 2.-4. Multi-head self-attention module
# --------------------------------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Pure-PyTorch multi-head self-attention (Vaswani et al., 2017, Sec. 3.2.2).

        head_i = Attention(X W_Q^i, X W_K^i, X W_V^i),   MHA(X) = Concat(head_1..head_h) W_O

    The h per-head projections are stored as one d_model x d_model matrix each (W_Q, W_K, W_V) and split
    into heads by reshaping; W_O mixes the concatenated heads back into d_model.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # causal mask cache (not a parameter, not saved in state_dict); grown on demand
        self.register_buffer("_causal_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    # ---- head bookkeeping -------------------------------------------------------------------------
    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)."""
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model); inverse of split_heads."""
        b, _, n, _ = t.shape
        return t.transpose(1, 2).contiguous().view(b, n, self.d_model)

    def project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learned projections followed by the head split."""
        return self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))

    # ---- masks ------------------------------------------------------------------------------------
    def _causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_cache.shape[0] < seq_len or self._causal_cache.device != device:
            self._causal_cache = causal_mask(max(seq_len, self._causal_cache.shape[0]), device)
        return self._causal_cache[:seq_len, :seq_len]

    def build_mask(
        self, seq_len: int, causal: bool, attention_mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Combine the causal mask (seq_len, seq_len) with an optional key-padding mask (batch, seq_len)
        (True = real token) into one boolean mask broadcastable to (batch, heads, seq_len, seq_len).
        """
        mask = None
        if causal and seq_len > 1:
            mask = self._causal(seq_len, device)[None, None]            # (1, 1, n, n)
        if attention_mask is not None:
            pad = attention_mask.to(torch.bool).to(device)[:, None, None, :]   # (b, 1, 1, n): masks keys
            mask = pad if mask is None else (mask & pad)
        return mask

    # ---- forward ----------------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        rope: Optional[RopeHook] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              (batch, seq_len, d_model)
            causal:         lower-triangular mask so position i only attends to positions <= i
            attention_mask: optional boolean (batch, seq_len), True for real tokens, False for padding (keys)
            rope:           optional hook (q, k) -> (q, k) applied after the head split, e.g.
                            lambda q, k: apply_rotary_emb(q, k, freqs_cis[:seq_len])   (Part 2)
        Returns:
            output  (batch, seq_len, d_model)
            weights (batch, num_heads, seq_len, seq_len), rows sum to 1 (0 for fully masked rows)
        """
        b, n, d = x.shape
        assert d == self.d_model, f"expected last dim {self.d_model}, got {d}"

        q, k, v = self.project_qkv(x)                                   # 1. projections  2. head split
        if rope is not None:
            q, k = rope(q, k)                                           # relative position (Part 2 hook)
        mask = self.build_mask(n, causal, attention_mask, x.device)     # 3. causal / padding mask
        context, weights = scaled_dot_product_attention(                # 4. softmax(QK^T/sqrt(d_k)) V
            q, k, v, mask=mask, scale=self.scale, dropout=self.dropout if self.training else None
        )
        output = self.w_o(self.merge_heads(context))                    # 5. concat heads, W_O
        return output, weights

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}"


# --------------------------------------------------------------------------------------------------
# 5. Variance preservation
# --------------------------------------------------------------------------------------------------
def softmax_entropy_bits(scores: torch.Tensor) -> float:
    """Mean Shannon entropy (bits) of the softmax rows of a score tensor."""
    p = torch.softmax(scores, dim=-1)
    h = -(p * torch.log2(p.clamp_min(1e-30))).sum(-1)
    return h.mean().item()


def variance_study(d_ks=(16, 64, 256, 1024), seq_len: int = 128, batch_size: int = 32, seed: int = 42) -> List[Dict]:
    """
    For q_i, k_i i.i.d. with E=0, Var=1:  Var(q.k) = sum_i Var(q_i k_i) = d_k, so Var(q.k / sqrt(d_k)) = 1.
    Returns one row per d_k with measured variances and the softmax entropy with and without the scale.
    """
    torch.manual_seed(seed)
    rows = []
    for d_k in d_ks:
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        unscaled = q @ k.transpose(-2, -1)
        scaled = unscaled / math.sqrt(d_k)
        rows.append({
            "d_k": d_k,
            "mean_unscaled": unscaled.mean().item(), "var_unscaled": unscaled.var().item(),
            "mean_scaled": scaled.mean().item(), "var_scaled": scaled.var().item(),
            "entropy_bits_unscaled": softmax_entropy_bits(unscaled),
            "entropy_bits_scaled": softmax_entropy_bits(scaled),
            "max_entropy_bits": math.log2(seq_len),
        })
    return rows


def verify_variance_preservation(quiet: bool = False) -> List[Dict]:
    """
    Cadet Verification: with zero-mean, unit-variance inputs the raw dot product has variance d_k
    (so softmax saturates and its gradient p(1-p) vanishes); scaling by 1/sqrt(d_k) restores unit variance.
    """
    rows = variance_study()
    if not quiet:
        print("=== Subsystem 03: Attention Variance Preservation Verification ===")
        print(f"{'d_k':>5} | {'Var unscaled':>12} (expect d_k) | {'Var scaled':>10} (expect 1) | "
              f"{'H unscaled':>10} | {'H scaled':>8} | max H")
        print("-" * 92)
        for r in rows:
            print(f"{r['d_k']:>5} | {r['var_unscaled']:>12.2f}              | {r['var_scaled']:>10.4f}            | "
                  f"{r['entropy_bits_unscaled']:>7.3f} b  | {r['entropy_bits_scaled']:>6.3f} b | {r['max_entropy_bits']:.2f} b")
    for r in rows:
        assert abs(r["var_scaled"] - 1.0) < 0.1, f"variance preservation failed for d_k={r['d_k']}"
        assert abs(r["var_unscaled"] / r["d_k"] - 1.0) < 0.1, f"unscaled variance != d_k for d_k={r['d_k']}"
    if not quiet:
        print("[PASS] Var(q.k) grows like d_k; Var(q.k / sqrt(d_k)) = 1 for every d_k. "
              "Without the scale the softmax entropy collapses towards 0 bits (one-hot attention, vanishing gradient).")
    return rows


# --------------------------------------------------------------------------------------------------
# Teammate modules (optional integration)
# --------------------------------------------------------------------------------------------------
def load_sibling(path: Path, name: str):
    """Import a sibling script whose file name starts with a digit. Returns the module or None."""
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # teammate file present but not importable (e.g. plotting at import time)
        print(f"  (could not import {path.name}: {type(exc).__name__}: {exc})")
        return None


# --------------------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------------------
class Checks:
    def __init__(self, quiet: bool = False):
        self.results: List[Dict] = []
        self.quiet = quiet

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False):
        status = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        self.results.append({"check": name, "status": status, "detail": detail})
        if not self.quiet:
            print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(r["status"] == "FAIL" for r in self.results)

    @property
    def passed(self) -> int:
        return sum(r["status"] == "PASS" for r in self.results)


def self_test(seed: int = 42, quiet: bool = False, device: str = "cpu") -> Checks:
    torch.manual_seed(seed)
    dev = torch.device(device)
    c = Checks(quiet)
    b, n, d, h = 2, 16, 256, 4
    mha = MultiHeadSelfAttention(d_model=d, num_heads=h, dropout=0.1).to(dev).eval()
    x = torch.randn(b, n, d, device=dev)

    if not quiet:
        print("=== Subsystem 03: Multi-Head Attention self-test ===")
        print("-- shapes, softmax, masking --")
    with torch.no_grad():
        out, w = mha(x, causal=True)
    c.add("output shape (b, n, d_model)", tuple(out.shape) == (b, n, d), str(tuple(out.shape)))
    c.add("weights shape (b, h, n, n)", tuple(w.shape) == (b, h, n, n), str(tuple(w.shape)))
    c.add("softmax rows sum to 1", torch.allclose(w.sum(-1), torch.ones(b, h, n, device=dev), atol=1e-5))
    upper = w.masked_select(torch.ones(n, n, dtype=torch.bool, device=dev).triu(1))
    c.add("causal: zero weight above the diagonal", bool((upper == 0).all()), f"max={upper.max().item():.1e}")
    c.add("no NaN/Inf in output", bool(torch.isfinite(out).all()))

    # future tokens must not influence earlier outputs under the causal mask (and must without it)
    j = 9
    x2 = x.clone()
    x2[:, j:] += torch.randn_like(x2[:, j:])
    with torch.no_grad():
        out2, _ = mha(x2, causal=True)
        out3, _ = mha(x, causal=False)
        out4, _ = mha(x2, causal=False)
    c.add(f"causal: perturbing positions >= {j} leaves outputs < {j} unchanged",
          torch.allclose(out[:, :j], out2[:, :j], atol=1e-6), f"max diff={(out[:, :j]-out2[:, :j]).abs().max():.1e}")
    c.add("non-causal: the same perturbation does change earlier outputs",
          not torch.allclose(out3[:, :j], out4[:, :j], atol=1e-4))

    # key padding mask (incl. left padding, where early query rows are fully masked)
    pad = torch.ones(b, n, dtype=torch.bool, device=dev)
    pad[0, -4:] = False      # right padding, sample 0
    pad[1, :3] = False       # left padding, sample 1
    with torch.no_grad():
        outp, wp = mha(x, causal=True, attention_mask=pad)
    c.add("padding: masked key columns get zero weight",
          bool((wp[0, :, :, -4:] == 0).all()) and bool((wp[1, :, :, :3] == 0).all()))
    c.add("padding: fully masked query rows -> zero weights, finite output (no NaN)",
          bool((wp[1, :, :3] == 0).all()) and bool(torch.isfinite(outp).all()))
    real_rows = wp[1, :, 3:]
    c.add("padding: remaining rows still sum to 1", torch.allclose(real_rows.sum(-1), torch.ones_like(real_rows.sum(-1)), atol=1e-5))

    if not quiet:
        print("-- head splitting --")
    c.add("merge_heads(split_heads(x)) == x", torch.equal(mha.merge_heads(mha.split_heads(x)), x))
    with torch.no_grad():
        q, k, v = mha.project_qkv(x)
        ctx_vec, w_vec = scaled_dot_product_attention(q, k, v, mask=causal_mask(n, dev)[None, None], scale=mha.scale)
        per_head = []
        for i in range(h):          # each head is an independent attention over its own d_head slice
            ctx_i, _ = scaled_dot_product_attention(q[:, i], k[:, i], v[:, i], mask=causal_mask(n, dev), scale=mha.scale)
            per_head.append(ctx_i)
        manual = torch.cat(per_head, dim=-1)        # (b, n, h*d_head) == Concat(head_1..head_h)
    c.add("vectorised heads == explicit per-head loop + concat", torch.allclose(manual, mha.merge_heads(ctx_vec), atol=1e-6))
    c.add("head_dim * num_heads == d_model", mha.head_dim * mha.num_heads == mha.d_model, f"{mha.head_dim}x{mha.num_heads}")

    if not quiet:
        print("-- independent oracles (used only here, never in the implementation) --")
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    c.add("matches F.scaled_dot_product_attention(is_causal=True)", torch.allclose(ref, ctx_vec, atol=1e-5),
          f"max diff={(ref-ctx_vec).abs().max():.1e}")
    torch_mha = nn.MultiheadAttention(d, h, dropout=0.0, bias=False, batch_first=True).to(dev).eval()
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0))
        torch_mha.out_proj.weight.copy_(mha.w_o.weight)
        ref_out, ref_w = torch_mha(x, x, x, attn_mask=~causal_mask(n, dev), need_weights=True, average_attn_weights=False)
    c.add("matches nn.MultiheadAttention with copied weights (output)", torch.allclose(ref_out, out, atol=1e-5),
          f"max diff={(ref_out-out).abs().max():.1e}")
    c.add("matches nn.MultiheadAttention with copied weights (attention map)", torch.allclose(ref_w, w, atol=1e-5))

    if not quiet:
        print("-- training behaviour --")
    mha.train()
    xg = x.clone().requires_grad_(True)
    outg, wg = mha(xg, causal=True)
    outg.square().mean().backward()
    grads = [mha.w_q.weight.grad, mha.w_k.weight.grad, mha.w_v.weight.grad, mha.w_o.weight.grad]
    c.add("backward: finite, non-zero gradients for W_Q, W_K, W_V, W_O",
          all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
    c.add("backward: gradient reaches the input", xg.grad is not None and torch.isfinite(xg.grad).all())
    with torch.no_grad():
        wt1 = mha(x, causal=True)[1]
        wt2 = mha(x, causal=True)[1]
    c.add("dropout active in train mode (two passes differ)", not torch.equal(wt1, wt2))
    mha.eval()
    with torch.no_grad():
        we1 = mha(x, causal=True)[1]
        we2 = mha(x, causal=True)[1]
    c.add("deterministic in eval mode", torch.equal(we1, we2))
    n_params = sum(p.numel() for p in mha.parameters())
    c.add("parameter count = 4 * d_model^2 (bias-free)", n_params == 4 * d * d, f"{n_params:,}")

    if not quiet:
        print("-- variance preservation (1/sqrt(d_k)) --")
    rows = variance_study()
    c.add("Var(q.k) == d_k within 10% for d_k in 16..1024",
          all(abs(r["var_unscaled"] / r["d_k"] - 1) < 0.1 for r in rows),
          ", ".join(f"{r['d_k']}:{r['var_unscaled']:.1f}" for r in rows))
    c.add("Var(q.k / sqrt(d_k)) == 1 within 0.1 for every d_k",
          all(abs(r["var_scaled"] - 1) < 0.1 for r in rows),
          ", ".join(f"{r['d_k']}:{r['var_scaled']:.3f}" for r in rows))
    big = rows[-1]
    c.add("without the scale softmax saturates (entropy < 0.5 bit at d_k=1024; scaled stays > 4 bits)",
          big["entropy_bits_unscaled"] < 0.5 and big["entropy_bits_scaled"] > 4.0,
          f"H={big['entropy_bits_unscaled']:.3f} vs {big['entropy_bits_scaled']:.3f} of {big['max_entropy_bits']:.2f} bits")

    if not quiet:
        print("-- integration with the other stations --")
    rope = load_sibling(ROPE_FILE, "rope_engine")
    if rope is None or not all(hasattr(rope, a) for a in ("precompute_freqs_cis", "apply_rotary_emb")):
        c.add("Part 2 RoPE hook: attention weights invariant to a global position shift", False,
              "02_RoPE_Positional_Engine_starter.py not importable / not yet pushed", skipped=True)
    else:
        try:
            delta = 37
            freqs = rope.precompute_freqs_cis(mha.head_dim, n + delta + 1)
            hook_a = lambda q_, k_: rope.apply_rotary_emb(q_, k_, freqs[:n])
            hook_b = lambda q_, k_: rope.apply_rotary_emb(q_, k_, freqs[delta:delta + n])
            with torch.no_grad():
                _, wa = mha(x, causal=True, rope=hook_a)
                _, wb = mha(x, causal=True, rope=hook_b)
                _, w0 = mha(x, causal=True)
            c.add(f"Part 2 RoPE hook: attention weights invariant to a global shift of {delta} positions",
                  torch.allclose(wa, wb, atol=1e-4), f"max diff={(wa-wb).abs().max():.1e}")
            c.add("Part 2 RoPE hook: rotation does change the attention pattern vs. no positions",
                  not torch.allclose(wa, w0, atol=1e-4))
        except Exception as exc:
            c.add("Part 2 RoPE hook", False, f"{type(exc).__name__}: {exc}")
    gqa = load_sibling(GQA_FILE, "gqa_bench")
    if gqa is None or not hasattr(gqa, "GQAAttention"):
        c.add("Part 3 GQAAttention(H_KV == H_Q) equals MultiHeadSelfAttention", False,
              "03_GQA_and_KVCache_Benchmark_starter.py not importable / not yet pushed", skipped=True)
    else:
        try:
            g = gqa.GQAAttention(d_model=d, num_q_heads=h, num_kv_heads=h).to(dev).eval()
            with torch.no_grad():
                for src, dst in ((mha.w_q, g.w_q), (mha.w_k, g.w_k), (mha.w_v, g.w_v), (mha.w_o, g.w_o)):
                    dst.weight.copy_(src.weight)
                g_out = g(x)
                g_out = g_out[0] if isinstance(g_out, tuple) else g_out
                try:                       # the starter GQA has no causal mask; prefer a causal kwarg if present
                    g_c = g(x, causal=True)
                    g_c = g_c[0] if isinstance(g_c, tuple) else g_c
                    ok = torch.allclose(g_c, out, atol=1e-5)
                    mode = "causal"
                except TypeError:
                    ok = torch.allclose(g_out, out3, atol=1e-5)
                    mode = "non-causal"
            c.add(f"Part 3 GQAAttention(H_KV == H_Q) equals MultiHeadSelfAttention ({mode})", ok)
        except Exception as exc:
            c.add("Part 3 GQAAttention(H_KV == H_Q) equals MultiHeadSelfAttention", False, f"{type(exc).__name__}: {exc}")

    if not quiet:
        print(f"\n{c.passed} passed, {c.failed} failed, {sum(r['status']=='SKIP' for r in c.results)} skipped")
    return c


# --------------------------------------------------------------------------------------------------
# Naive-attention scaling benchmark (motivates Parts 3 and 4)
# --------------------------------------------------------------------------------------------------
def benchmark(seq_lengths=(128, 256, 512, 1024, 2048), d_model: int = 256, num_heads: int = 4,
              repeats: int = 3, device: str = "cpu", quiet: bool = False) -> List[Dict]:
    """Time one forward pass per N and report the size of the materialised (b, h, N, N) score matrix."""
    dev = torch.device(device)
    mha = MultiHeadSelfAttention(d_model, num_heads, dropout=0.0).to(dev).eval()
    rows = []
    if not quiet:
        print("=== Subsystem 03: naive attention scaling (CPU, batch 1) ===")
        print(f"{'N':>6} | {'forward ms':>10} | {'score matrix MB':>15} | {'x per doubling':>14}")
        print("-" * 58)
    prev = None
    for n in seq_lengths:
        x = torch.randn(1, n, d_model, device=dev)
        with torch.no_grad():
            mha(x)                                     # warm-up
            t0 = time.perf_counter()
            for _ in range(repeats):
                mha(x)
            ms = (time.perf_counter() - t0) / repeats * 1e3
        mb = num_heads * n * n * 4 / 2 ** 20           # float32 scores, per batch element
        ratio = ms / prev if prev else float("nan")
        rows.append({"seq_len": n, "forward_ms": ms, "score_matrix_mb": mb, "time_ratio_vs_prev": ratio})
        if not quiet:
            print(f"{n:>6} | {ms:>10.2f} | {mb:>15.2f} | {ratio:>14.2f}")
        prev = ms
    if not quiet:
        print("Score matrix memory grows exactly 4x per doubling of N (O(N^2)); wall time approaches 4x once the\n"
              "N x N product dominates the O(N d^2) projections - the regime Parts 3 (KV cache / GQA) and 4 (FlashAttention) attack.")
    return rows


# --------------------------------------------------------------------------------------------------
# Sign-off gate and logging
# --------------------------------------------------------------------------------------------------
def check_signoffs(readme: Path = README_FILE) -> Tuple[bool, List[str]]:
    """All four station boxes in README.md must be ticked before the repo URL is transmitted."""
    if not readme.exists():
        return False, ["README.md missing"]
    text = readme.read_text(encoding="utf-8")
    boxes = re.findall(r"^- \[( |x|X)\] \*\*([^*(`]+)", text, flags=re.M)
    missing = [role.strip() for state, role in boxes if state == " "]
    return len(boxes) >= 4 and not missing, missing


def write_log(mode: str, payload: Dict) -> Path:
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps03-{mode}.json"
    payload = {"mission": "03", "station": "Systems Architect", "mode": mode,
               "torch": torch.__version__, "python": sys.version.split()[0],
               "timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Mission 03 Part 1 - Multi-Head Self-Attention from first principles")
    p.add_argument("--selftest", action="store_true", help="run the full verification suite")
    p.add_argument("--bench", action="store_true", help="time naive attention for growing N")
    p.add_argument("--check-signoffs", action="store_true", help="verify all four README boxes are ticked")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-log", action="store_true", help="do not write logs/*.json")
    a = p.parse_args(argv)

    if a.check_signoffs:
        ok, missing = check_signoffs()
        print("Sign-off gate:", "PASS - all four stations signed" if ok else f"BLOCKED - missing: {', '.join(missing) or 'checklist not found'}")
        return 0 if ok else 1

    if a.selftest:
        c = self_test(seed=a.seed, quiet=a.quiet, device=a.device)
        if not a.no_log:
            path = write_log("selftest", {"seed": a.seed, "device": a.device, "passed": c.passed, "failed": c.failed,
                                          "checks": c.results, "variance_study": variance_study()})
            print(f"log: {path.relative_to(HERE)}")
        return 1 if c.failed else 0

    if a.bench:
        rows = benchmark(device=a.device, quiet=a.quiet)
        if not a.no_log:
            path = write_log("bench", {"device": a.device, "rows": rows})
            print(f"log: {path.relative_to(HERE)}")
        return 0

    # default: starter behaviour - variance test + demo forward pass
    torch.manual_seed(a.seed)
    rows = verify_variance_preservation(quiet=a.quiet)
    mha = MultiHeadSelfAttention(d_model=256, num_heads=4).to(a.device).eval()
    dummy_input = torch.randn(2, 16, 256, device=a.device)
    with torch.no_grad():
        out, attn = mha(dummy_input, causal=True)
    print(f"\n{mha}")
    print(f"Output shape: {tuple(out.shape)}, Attention shape: {tuple(attn.shape)}")
    print("Causal attention map, head 0, first 6 query rows (weights above the diagonal are exactly 0):")
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    print(attn[0, 0, :6, :6])
    if not a.no_log:
        path = write_log("demo", {"seed": a.seed, "variance_study": rows,
                                  "output_shape": list(out.shape), "attention_shape": list(attn.shape)})
        print(f"log: {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
