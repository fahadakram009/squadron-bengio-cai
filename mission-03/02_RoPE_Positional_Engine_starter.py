"""
Mission 03 - Part 2: Communications & Protocol Officer (Madhu)
Rotary Positional Embeddings (RoPE) Engine & Relative Invariance Validation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Subsystem 03: Distributed Multi-Head Attention Mesh & KV-Cache Transceiver

What is in this file:
  1. apply_rope(x, freqs_cis): frequencies theta_i = base^(-2(i-1)/d) and the 2D rotation of each pair
  2. relative invariance: derivation below + numerical checks (verify_relative_invariance, self_test)
  3. plot_decay(): RoPE decay vs ALiBi linear penalty -> figures/rope_vs_alibi_decay.png
  4. extrapolation_report(): what happens when the context gets longer than the training length

Note: Part 1 (01_MultiHead_Attention_Core_starter.py) imports this file and calls
precompute_freqs_cis() and apply_rotary_emb(), so I kept those names and arguments the same as the starter.

Derivation (why the score only depends on m - n)
-------------------------------------------------
Split the head dimension d into d/2 pairs and treat pair i as a complex number:
    x_c[i] = x[2i] + j*x[2i+1]
RoPE at position m multiplies pair i by e^(j*m*theta_i). For one pair this is the 2x2 rotation
    [cos(m*theta_i)  -sin(m*theta_i)]
    [sin(m*theta_i)   cos(m*theta_i)]
so the whole thing is a block diagonal rotation matrix R_m (orthogonal, R_0 = I, R_m R_n = R_(m+n)).

For real vectors written as complex pairs: <a, b> = Re( sum_i a_c[i] * conj(b_c[i]) ). So
    <R_m q, R_n k> = Re sum_i q_c[i] e^(j m theta_i) * conj(k_c[i] e^(j n theta_i))
                   = Re sum_i q_c[i] conj(k_c[i]) e^(j (m-n) theta_i)
                   = g(q, k, m - n)
m and n only appear as m - n. If both are shifted by delta, m - n stays the same, so the dot product
stays the same for any delta. Same thing in matrix form: R_m^T R_n = R_(n-m).
(Su et al. 2021, RoFormer, section 3.4)

How to run:
    python 02_RoPE_Positional_Engine_starter.py                  # everything
    python 02_RoPE_Positional_Engine_starter.py --selftest       # only the tests
    python 02_RoPE_Positional_Engine_starter.py --plot           # only the figure
    python 02_RoPE_Positional_Engine_starter.py --extrapolation  # only the extrapolation table
A json log is saved in logs/ (use --no-log to skip it).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_FIGURE = HERE / "figures" / "rope_vs_alibi_decay.png"


# ---------------------------------------------------------------------------
# 1. frequencies + rotation
# ---------------------------------------------------------------------------

def rope_frequencies(dim: int, base: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """theta_i = base^(-2(i-1)/d) for i = 1..d/2.
    First pair turns 1 rad per token, last pair turns very slowly (wavelength ~ 2*pi*base tokens)."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE needs a positive, even head dimension (got {dim}).")
    exponents = torch.arange(0, dim, 2, dtype=dtype) / dim   # 2(i-1)/d
    return base ** (-exponents)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    *,
    scale: float = 1.0,
    ntk_alpha: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Table of e^(j*m*theta_i) for positions m = 0..end-1, shape (end, dim // 2), complex.

    theta     : frequency base (10000 like RoFormer / LLaMA 2)
    scale     : position interpolation (Chen et al. 2023), position m is used as m / scale
    ntk_alpha : NTK-aware scaling, base becomes base * alpha^(d/(d-2))
    dtype     : float32 -> complex64 (default), float64 -> complex128 (used in the tests)
    """
    if dim % 2 != 0:
        raise ValueError("Dimension must be even for RoPE 2D pairing.")
    if scale <= 0 or ntk_alpha <= 0:
        raise ValueError("scale and ntk_alpha must be positive.")
    base = theta * ntk_alpha ** (dim / (dim - 2)) if dim > 2 else theta * ntk_alpha
    inv_freq = rope_frequencies(dim, base, dtype=dtype)          # (dim/2,)
    positions = torch.arange(0, end, 1, dtype=dtype) / scale     # (end,)
    angles = torch.outer(positions, inv_freq)                    # m * theta_i
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate each pair (x[2i], x[2i+1]) of the last dimension.

    x         : (..., seq_len, head_dim), e.g. (batch, heads, seq_len, head_dim)
    freqs_cis : (seq_len, head_dim // 2), already sliced to the right positions
    Returns a tensor with the same shape and dtype as x.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE needs an even last dimension (got {x.shape[-1]}).")
    seq_len, half = x.shape[-2], x.shape[-1] // 2
    if tuple(freqs_cis.shape) != (seq_len, half):
        raise ValueError(f"freqs_cis must have shape ({seq_len}, {half}) for x of shape {tuple(x.shape)}; "
                         f"got {tuple(freqs_cis.shape)}. Slice it with freqs_cis[start:start + seq_len].")
    # do the math in float32 (or float64 if the input is float64), fp16/bf16 lose too much precision
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    cplx = torch.complex128 if work == torch.float64 else torch.complex64
    x_c = torch.view_as_complex(x.to(work).reshape(*x.shape[:-1], half, 2).contiguous())
    f = freqs_cis.to(cplx).reshape(*([1] * (x.dim() - 2)), seq_len, half)   # broadcast over batch/heads
    return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE for queries and keys (V is not rotated). This is what Part 1 calls in its rope hook.
    xk can have fewer heads than xq (GQA from Part 3)."""
    return apply_rope(xq, freqs_cis), apply_rope(xk, freqs_cis)


# --- second implementations, only used to check apply_rope ---

def apply_rope_real(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Same rotation with cos/sin, no complex numbers."""
    d = x.shape[-1]
    inv = rope_frequencies(d, theta, dtype=torch.float64)
    ang = torch.outer(positions.to(torch.float64), inv)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].to(torch.float64), x[..., 1::2].to(torch.float64)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def rotation_matrix(dim: int, m: float, theta: float = 10000.0) -> torch.Tensor:
    """Full d x d block diagonal matrix R_m (float64)."""
    inv = rope_frequencies(dim, theta, dtype=torch.float64)
    r = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(dim // 2):
        c, s = math.cos(m * inv[i].item()), math.sin(m * inv[i].item())
        r[2 * i, 2 * i], r[2 * i, 2 * i + 1] = c, -s
        r[2 * i + 1, 2 * i], r[2 * i + 1, 2 * i + 1] = s, c
    return r


def relative_score(q: torch.Tensor, k: torch.Tensor, distance: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """g(q, k, m-n) from the derivation, computed only from the distance.
    q, k: (T, d), distance: (T,). Returns (T,)."""
    d = q.shape[-1]
    inv = rope_frequencies(d, theta, dtype=torch.float64)
    q_c = torch.view_as_complex(q.to(torch.float64).reshape(*q.shape[:-1], d // 2, 2))
    k_c = torch.view_as_complex(k.to(torch.float64).reshape(*k.shape[:-1], d // 2, 2))
    ang = torch.outer(distance.to(torch.float64), inv)
    rot = torch.polar(torch.ones_like(ang), ang)
    return (q_c * torch.conj(k_c) * rot).real.sum(dim=-1)


# ---------------------------------------------------------------------------
# 2. relative invariance checks
# ---------------------------------------------------------------------------

def verify_relative_invariance(quiet: bool = False) -> Dict:
    """Cadet Verification: <RoPE(q, m), RoPE(k, n)> == <RoPE(q, m + delta), RoPE(k, n + delta)>."""
    say = (lambda *a: None) if quiet else print
    say("=== Subsystem 03: RoPE Relative Shift Invariance Verification ===")
    torch.manual_seed(42)
    dim, max_seq_len = 64, 256
    freqs_cis = precompute_freqs_cis(dim=dim, end=max_seq_len)
    m, n, delta = 10, 25, 40

    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    q_m, _ = apply_rotary_emb(q, q, freqs_cis[m: m + 1])
    _, k_n = apply_rotary_emb(k, k, freqs_cis[n: n + 1])
    dot_orig = (q_m * k_n).sum().item()
    q_s, _ = apply_rotary_emb(q, q, freqs_cis[m + delta: m + delta + 1])
    _, k_s = apply_rotary_emb(k, k, freqs_cis[n + delta: n + delta + 1])
    dot_shifted = (q_s * k_s).sum().item()
    g = relative_score(q.reshape(1, dim), k.reshape(1, dim), torch.tensor([float(m - n)])).item()

    say(f"Dot product <RoPE(q, {m}), RoPE(k, {n})>:                         {dot_orig:.6f}")
    say(f"Dot product <RoPE(q, {m + delta}), RoPE(k, {n + delta})> (delta={delta}):        {dot_shifted:.6f}")
    say(f"Closed form g(q, k, m - n = {m - n}):                            {g:.6f}")
    diff = abs(dot_orig - dot_shifted)
    tol = 1e-5 * max(1.0, abs(dot_orig))   # float32, so compare relative to the size of the dot product
    say(f"Absolute Difference: {diff:.8e}   (fp32 tolerance {tol:.1e})")
    assert diff < tol, "RoPE relative invariance violation detected!"
    assert abs(dot_orig - g) < tol, "RoPE score does not match the closed form g(q, k, m - n)!"
    say("[PASS] RoPE relative distance invariance holds across spatial translations.\n")
    return {"m": m, "n": n, "delta": delta, "dot_orig": dot_orig, "dot_shifted": dot_shifted,
            "closed_form": g, "abs_diff": diff}


def invariance_sweep(trials: int = 2000, dim: int = 64, max_pos: int = 8192, seed: int = 0,
                     dtype: torch.dtype = torch.float64) -> Dict:
    """Same check as above but for many random q, k, m, n, delta at once."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(trials, dim, generator=gen, dtype=torch.float64).to(dtype)
    k = torch.randn(trials, dim, generator=gen, dtype=torch.float64).to(dtype)
    m = torch.randint(0, max_pos, (trials,), generator=gen)
    n = torch.randint(0, max_pos, (trials,), generator=gen)
    delta = torch.randint(0, max_pos, (trials,), generator=gen)
    freqs = precompute_freqs_cis(dim, 2 * max_pos, dtype=torch.float64 if dtype == torch.float64 else torch.float32)

    def score(pm, pn):
        return (apply_rope(q, freqs[pm]) * apply_rope(k, freqs[pn])).to(torch.float64).sum(dim=-1)

    s0, s1 = score(m, n), score(m + delta, n + delta)
    g = relative_score(q, k, (m - n).to(torch.float64))
    size = q.to(torch.float64).norm(dim=-1) * k.to(torch.float64).norm(dim=-1)   # |score| <= |q||k|
    return {
        "trials": trials, "dim": dim, "max_position": int(2 * max_pos), "dtype": str(dtype).replace("torch.", ""),
        "max_abs_shift_error": (s0 - s1).abs().max().item(),
        "max_rel_shift_error": ((s0 - s1).abs() / size).max().item(),
        "max_abs_closed_form_error": (s0 - g).abs().max().item(),
        "mean_abs_score": s0.abs().mean().item(),
    }


# ---------------------------------------------------------------------------
# 3. small causal attention (for the attention level test and the heatmaps)
# ---------------------------------------------------------------------------

def causal_attention_weights(q: torch.Tensor, k: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + bias) with future positions masked."""
    n, d = q.shape[-2], q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    if bias is not None:
        scores = scores + bias
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


def alibi_slopes(num_heads: int) -> List[float]:
    """ALiBi slopes from Press et al. 2022: 2^(-8/H), 2^(-16/H), ...
    (if H is not a power of 2 the paper takes extra slopes from the next power of 2)"""
    def power_of_two(h: int) -> List[float]:
        start = 2 ** (-8 / h)
        return [start ** (i + 1) for i in range(h)]
    if math.log2(num_heads).is_integer():
        return power_of_two(num_heads)
    closest = 2 ** math.floor(math.log2(num_heads))
    return power_of_two(closest) + power_of_two(2 * closest)[0::2][: num_heads - closest]


def alibi_bias(seq_len: int, slope: float) -> torch.Tensor:
    """bias[i, j] = -slope * (i - j) for j <= i."""
    pos = torch.arange(0, seq_len, 1, dtype=torch.float64)
    dist = pos.reshape(seq_len, 1) - pos.reshape(1, seq_len)
    return -slope * dist.clamp(min=0.0)


# ---------------------------------------------------------------------------
# 4. RoPE vs ALiBi decay
# ---------------------------------------------------------------------------

def rope_identical_content_score(dim: int, distances: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """<R_m c, R_(m+r) c> with c = all ones, which is 2 * sum_i cos(r * theta_i).
    So this is the position part of the score when query and key have the same content. Equals d at r = 0."""
    inv = rope_frequencies(dim, theta, dtype=torch.float64)
    return 2.0 * torch.cos(torch.outer(distances.to(torch.float64), inv)).sum(dim=-1)


def su_decay_bound(dim: int, distances: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Long-term decay bound from the RoFormer paper (section 3.4.3): mean over j of |S_j|,
    S_j = sum_{i<j} e^(j r theta_i). It gets smaller when the distance r grows."""
    inv = rope_frequencies(dim, theta, dtype=torch.float64)
    ang = torch.outer(distances.to(torch.float64), inv)
    partial = torch.cumsum(torch.polar(torch.ones_like(ang), ang), dim=-1)
    return partial.abs().mean(dim=-1)


def decay_curves(dim: int = 128, context: int = 1024, theta: float = 10000.0, num_heads: int = 8) -> Dict:
    """Logit change and attention weight vs distance, for the last query in a context of length `context`."""
    r = torch.arange(0, context, 1, dtype=torch.float64)
    rope_logit = rope_identical_content_score(dim, r, theta) / math.sqrt(dim)
    rope_penalty = rope_logit - rope_logit[0]
    rope_weight = torch.softmax(rope_logit, dim=-1)
    slopes = alibi_slopes(num_heads)
    content = dim / math.sqrt(dim)   # same content score as RoPE at r = 0, so the curves are comparable
    alibi = {f"{s:.4g}": {"penalty": -s * r, "weight": torch.softmax(content - s * r, dim=-1)} for s in slopes}
    return {"r": r, "rope_penalty": rope_penalty, "rope_weight": rope_weight,
            "rope_bound": su_decay_bound(dim, r, theta), "alibi": alibi, "slopes": slopes,
            "dim": dim, "context": context, "theta": theta}


def attention_maps(seq_len: int = 64, dim: int = 64, theta: float = 10000.0, alibi_slope: float = 1 / 16,
                   seed: int = 7, content_noise: float = 0.5) -> Dict:
    """Attention matrices with RoPE, with ALiBi and with no positions.
    All tokens share one content vector plus some noise, so the difference comes from the position method."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.ones(1, dim, dtype=torch.float64)
    x = base + content_noise * torch.randn(seq_len, dim, generator=gen, dtype=torch.float64)
    freqs = precompute_freqs_cis(dim, seq_len, theta, dtype=torch.float64)
    q_r, k_r = apply_rotary_emb(x, x, freqs)
    rope_map = causal_attention_weights(q_r, k_r)
    alibi_map = causal_attention_weights(x, x, alibi_bias(seq_len, alibi_slope))
    nopos_map = causal_attention_weights(x, x)
    return {"rope": rope_map, "alibi": alibi_map, "none": nopos_map, "slope": alibi_slope,
            "seq_len": seq_len, "dim": dim}


def mean_attention_distance(weights: torch.Tensor) -> float:
    """How far back a query looks on average."""
    n = weights.shape[-1]
    pos = torch.arange(0, n, 1, dtype=torch.float64)
    dist = (pos.reshape(n, 1) - pos.reshape(1, n)).abs()
    return (weights.to(torch.float64) * dist).sum(dim=-1).mean().item()


def plot_decay(path: Path = DEFAULT_FIGURE, dim: int = 128, context: int = 1024, theta: float = 10000.0) -> Optional[Path]:
    """4 plots: (a) logit penalty, (b) attention weight, (c) RoPE heatmap, (d) ALiBi heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed - `pip install matplotlib` to draw the decay figure.")
        return None

    curves = decay_curves(dim, context, theta)
    maps = attention_maps()
    r = curves["r"].numpy()
    shown = [curves["slopes"][i] for i in (0, 3, 7)]   # 1/2, 1/16, 1/256

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    a = ax[0, 0]
    a.plot(r[1:], curves["rope_penalty"].numpy()[1:], color="#1f6feb", lw=1.6,
           label=f"RoPE (d={dim}, base={theta:g})")
    for s, ls in zip(shown, ("-", "--", ":")):
        a.plot(r[1:], curves["alibi"][f"{s:.4g}"]["penalty"].numpy()[1:], color="#d1242f", ls=ls, lw=1.4,
               label=f"ALiBi slope {s:.4g}")
    a.set_xscale("log")
    a.set_ylim(-2.5 * math.sqrt(dim), 1)
    a.set_xlabel("relative distance  m - n  (tokens, log)")
    a.set_ylabel("logit change vs. distance 0")
    a.set_title("(a) Logit penalty vs distance: RoPE decays + oscillates; ALiBi is linear in r (log x-axis)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower left")
    b2 = a.twinx()
    b2.plot(r[1:], curves["rope_bound"].numpy()[1:], color="#8250df", lw=1.0, alpha=0.7)
    b2.set_ylabel("Su et al. bound  (1/(d/2)) sum |S_j|", color="#8250df")
    b2.tick_params(axis="y", colors="#8250df")

    b = ax[0, 1]
    b.plot(r, curves["rope_weight"].numpy(), color="#1f6feb", lw=1.6, label="RoPE")
    for s, ls in zip(shown, ("-", "--", ":")):
        b.plot(r, curves["alibi"][f"{s:.4g}"]["weight"].numpy(), color="#d1242f", ls=ls, lw=1.4,
               label=f"ALiBi slope {s:.4g}")
    b.plot(r, [1.0 / context] * len(r), color="grey", lw=0.8, ls="-.", label="uniform 1/N (no position)")
    b.set_xscale("symlog", linthresh=1)
    b.set_yscale("log")
    b.set_ylim(1e-12, 1.5)
    b.set_xlabel("distance from the query (tokens)")
    b.set_ylabel("attention weight  (query at position N-1)")
    b.set_title(f"(b) Attention weight vs distance, N = {context}, identical key content")
    b.grid(alpha=0.3)
    b.legend(fontsize=8, loc="lower left")

    n = maps["seq_len"]
    for axis, key, title in ((ax[1, 0], "rope", "(c) Causal attention with RoPE"),
                             (ax[1, 1], "alibi", f"(d) Causal attention with ALiBi (slope {maps['slope']:.4g})")):
        im = axis.imshow(maps[key].numpy(), cmap="magma", norm=matplotlib.colors.PowerNorm(0.4, vmin=0, vmax=1))
        axis.set_title(f"{title}\nmean attended distance {mean_attention_distance(maps[key]):.1f} tokens")
        axis.set_xlabel("key position j")
        axis.set_ylabel("query position i")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04, label="weight (power-scaled colour)")
    fig.suptitle(f"Mission 03 / Part 2 - RoPE long-term decay vs ALiBi linear distance penalty "
                 f"(heatmaps: N={n}, d={maps['dim']}, shared content + noise)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. context length extrapolation
# ---------------------------------------------------------------------------

def extrapolation_report(dim: int = 128, train_len: int = 4096, target_len: int = 16384,
                         theta: float = 10000.0) -> List[Dict]:
    """Count the pairs that see angles at target_len that they never saw during training.
    A pair that already did a full turn (2*pi) during training has seen every angle, so it is fine.
    The slow pairs did not, and with a longer context they go past what they saw."""
    s = target_len / train_len
    trained_inv = rope_frequencies(dim, theta, dtype=torch.float64)
    rows = []
    for name, kw in (("RoPE (plain)", {}), (f"Position Interpolation (s={s:g})", {"scale": s}),
                     (f"NTK-aware base (alpha={s:g})", {"ntk_alpha": s})):
        f = precompute_freqs_cis(dim, 2, theta, dtype=torch.float64, **kw)
        eff = torch.angle(f[1])                      # rotation per token for each pair with this method
        seen_max = trained_inv * train_len           # biggest angle seen in training
        new_max = eff * target_len                   # biggest angle at the new length
        unseen = ((seen_max < 2 * math.pi) & (new_max > seen_max * (1 + 1e-9)))
        rows.append({
            "method": name,
            "unseen_pairs": int(unseen.sum().item()),
            "pairs": dim // 2,
            "slowest_pair_angle_ratio": (new_max[-1] / seen_max[-1]).item(),
            "worst_overshoot": max([1.0] + [(new_max[i] / seen_max[i]).item() for i in range(dim // 2)
                                            if seen_max[i].item() < 2 * math.pi]),
            "fastest_pair_rad_per_token": eff[0].item(),
            "adjacent_token_resolution_vs_train": (eff[0] / trained_inv[0]).item(),
        })
    rows.append({"method": "ALiBi", "unseen_pairs": 0, "pairs": 0, "slowest_pair_angle_ratio": float("nan"),
                 "worst_overshoot": float("nan"),
                 "fastest_pair_rad_per_token": float("nan"), "adjacent_token_resolution_vs_train": 1.0})
    return rows


def print_extrapolation(rows: List[Dict], dim: int, train_len: int, target_len: int) -> None:
    print(f"=== Context-length extrapolation: d={dim}, trained on {train_len} tokens, run at {target_len} ===")
    print(f"{'method':<34}{'pairs at unseen angles':>24}{'worst overshoot':>17}{'slowest pair':>14}"
          f"{'local resolution':>18}")
    for r in rows:
        if r["method"] == "ALiBi":
            print(f"{r['method']:<34}{'n/a (no rotation)':>24}{'n/a':>17}{'n/a':>14}{'1.00x':>18}")
            continue
        print(f"{r['method']:<34}{str(r['unseen_pairs']) + ' / ' + str(r['pairs']):>24}"
              f"{r['worst_overshoot']:>16.2f}x{r['slowest_pair_angle_ratio']:>13.2f}x"
              f"{r['adjacent_token_resolution_vs_train']:>17.2f}x")
    print("(overshoot = biggest angle at the new length / biggest angle seen in training,\n"
          " local resolution = rotation per token of the fastest pair compared to training)")
    print("- plain RoPE: the slow pairs get angles they never saw in training -> perplexity goes up past L_train\n"
          "- Position Interpolation: all angles stay in range, but neighbouring tokens are harder to tell apart\n"
          "  (fast pairs turn s times slower), so it needs some fine-tuning\n"
          "- NTK scaling: fast pairs unchanged, slowest pair slowed by alpha, middle pairs still overshoot a bit\n"
          "- ALiBi: no rotation, just a linear penalty, so a longer context is no problem (but strong recency bias)\n")


# ---------------------------------------------------------------------------
# 6. self test
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self, quiet: bool = False):
        self.items: List[Dict] = []
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


def base_freq_row(dim: int, m: int) -> torch.Tensor:
    return precompute_freqs_cis(dim, m + 1, dtype=torch.float64)[m]


def self_test(seed: int = 42, quiet: bool = False) -> Tuple[Checks, Dict]:
    c = Checks(quiet)
    say = (lambda *a: None) if quiet else print
    say("=== Part 2 self-test: RoPE engine ===")
    gen = torch.Generator().manual_seed(seed)
    d, n, b, h = 64, 48, 2, 4
    f64 = precompute_freqs_cis(d, 4096, dtype=torch.float64)
    extra: Dict = {}

    # frequencies
    inv = rope_frequencies(d, 10000.0, dtype=torch.float64)
    expect = torch.tensor([10000.0 ** (-2 * (i - 1) / d) for i in range(1, d // 2 + 1)], dtype=torch.float64)
    c.add("theta_i = base^{-2(i-1)/d} for i = 1..d/2", (inv - expect).abs().max().item() < 1e-15,
          f"theta_1 = {inv[0].item():g}, theta_d/2 = {inv[-1].item():.3e}")
    f32 = precompute_freqs_cis(d, n)
    c.add("precompute_freqs_cis: shape (end, d/2), complex64, |cis| = 1",
          tuple(f32.shape) == (n, d // 2) and f32.dtype == torch.complex64
          and (f32.abs() - 1).abs().max().item() < 1e-6)
    try:
        precompute_freqs_cis(63, 8)
        c.add("odd head dimension is rejected", False)
    except (ValueError, AssertionError):
        c.add("odd head dimension is rejected", True)

    # the rotation, compared with the cos/sin version and the full matrix
    x = torch.randn(b, h, n, d, generator=gen, dtype=torch.float64)
    pos = torch.arange(0, n, 1, dtype=torch.float64)
    y = apply_rope(x, f64[:n])
    err_real = (y - apply_rope_real(x, pos)).abs().max().item()
    c.add("complex path == explicit cos/sin 2x2 rotation", err_real < 1e-12, f"max |diff| {err_real:.1e}")
    m_test = 37
    r37 = rotation_matrix(d, m_test)
    err_mat = (apply_rope(x[0, 0, :1], f64[m_test:m_test + 1])[0] - r37 @ x[0, 0, 0]).abs().max().item()
    c.add("complex path == dense block-diagonal matrix R_{Theta,m}", err_mat < 1e-12, f"max |diff| {err_mat:.1e}")
    ortho = (r37.transpose(0, 1) @ r37 - torch.eye(d, dtype=torch.float64)).abs().max().item()
    c.add("R_{Theta,m} is orthogonal (R^T R = I)", ortho < 1e-12)
    norm_err = (y.norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    c.add("rotation preserves vector norms", norm_err < 1e-12, f"max |diff| {norm_err:.1e}")
    c.add("position 0 is the identity", (apply_rope(x[..., :1, :], f64[:1]) - x[..., :1, :]).abs().max().item() < 1e-15)
    comp = (rotation_matrix(d, 5) @ rotation_matrix(d, 11) - rotation_matrix(d, 16)).abs().max().item()
    c.add("composition R_m R_n = R_{m+n}", comp < 1e-12)
    rel = (rotation_matrix(d, 9).transpose(0, 1) @ rotation_matrix(d, 30) - rotation_matrix(d, 21)).abs().max().item()
    c.add("R_m^T R_n = R_{n-m}  (matrix form of relative invariance)", rel < 1e-12)

    # shapes / dtypes
    q_a, k_a = apply_rotary_emb(x, x[:, :1], f64[:n])
    c.add("apply_rotary_emb keeps shapes, works with fewer K heads (GQA 4:1)",
          tuple(q_a.shape) == tuple(x.shape) and tuple(k_a.shape) == (b, 1, n, d))
    x16 = torch.randn(n, d, generator=gen, dtype=torch.float64).to(torch.float16)
    y16 = apply_rope(x16, precompute_freqs_cis(d, n))
    c.add("half-precision input is computed in fp32 and returned as fp16", y16.dtype == torch.float16)
    try:
        apply_rope(x, f64[: n + 1])
        c.add("mismatched freqs_cis length raises a clear error", False)
    except ValueError:
        c.add("mismatched freqs_cis length raises a clear error", True)

    # relative invariance
    sweep64 = invariance_sweep(dtype=torch.float64, seed=seed)
    c.add(f"shift invariance, {sweep64['trials']} random (q,k,m,n,delta), positions < {sweep64['max_position']}, fp64",
          sweep64["max_abs_shift_error"] < 1e-9, f"max |diff| {sweep64['max_abs_shift_error']:.1e}")
    c.add("score equals closed form g(q, k, m - n)", sweep64["max_abs_closed_form_error"] < 1e-9,
          f"max |diff| {sweep64['max_abs_closed_form_error']:.1e}")
    sweep32 = invariance_sweep(dtype=torch.float32, seed=seed)
    # in fp32 the angle m*theta gets rounded at big m, so only check it relative to |q||k|
    c.add("shift invariance in fp32 (angles m*theta lose precision at large m)",
          sweep32["max_rel_shift_error"] < 1e-2,
          f"max |diff| {sweep32['max_abs_shift_error']:.1e}, relative to |q||k| {sweep32['max_rel_shift_error']:.1e}")
    extra["invariance_fp64"], extra["invariance_fp32"] = sweep64, sweep32

    # whole attention map should not change if all positions move by delta
    qx = torch.randn(1, h, n, d, generator=gen, dtype=torch.float64)
    kx = torch.randn(1, h, n, d, generator=gen, dtype=torch.float64)
    delta = 1000
    wa = causal_attention_weights(*apply_rotary_emb(qx, kx, f64[:n]))
    wb = causal_attention_weights(*apply_rotary_emb(qx, kx, f64[delta:delta + n]))
    w0 = causal_attention_weights(qx, kx)
    att_err = (wa - wb).abs().max().item()
    c.add(f"causal attention map invariant to a global shift of {delta} positions", att_err < 1e-12,
          f"max |diff| {att_err:.1e}")
    c.add("RoPE does change the attention pattern vs. no positions", (wa - w0).abs().max().item() > 1e-3)
    qv = torch.randn(1, d, generator=gen, dtype=torch.float64)
    kv = torch.randn(1, d, generator=gen, dtype=torch.float64)
    s_near = relative_score(qv, kv, torch.tensor([3.0])).item()
    s_far = relative_score(qv, kv, torch.tensor([300.0])).item()
    c.add("score depends on the distance (|m-n| = 3 vs 300 differ)", abs(s_near - s_far) > 1e-3)

    # gradient: R is orthogonal so d|Rx|^2/dx = 2x
    try:
        xg = torch.randn(n, d, generator=gen, dtype=torch.float64).requires_grad_(True)
        apply_rope(xg, f64[:n]).pow(2).sum().backward()
        grad_ok = xg.grad is not None and torch.allclose(xg.grad, 2 * xg.detach(), atol=1e-10)
        c.add("gradients flow (d|Rx|^2/dx = 2x because R is orthogonal)", grad_ok)
    except Exception as exc:
        c.add("gradients flow", False, f"{type(exc).__name__}: {exc}")

    # decay / ALiBi / scaling
    curves = decay_curves()
    bound = curves["rope_bound"]
    early, late = bound[1:16].mean().item(), bound[512:].mean().item()
    c.add("Su et al. bound decays with distance (mean r<16 vs r>=512)", late < early,
          f"{early:.2f} -> {late:.2f}")
    s_list = alibi_slopes(8)
    c.add("ALiBi slopes for 8 heads = 1/2 ... 1/256", abs(s_list[0] - 0.5) < 1e-12 and abs(s_list[-1] - 1 / 256) < 1e-12)
    maps = attention_maps()
    rows_ok = all((maps[k].sum(dim=-1) - 1).abs().max().item() < 1e-12 for k in ("rope", "alibi", "none"))
    c.add("heatmap attention rows sum to 1 and are causal",
          rows_ok and maps["rope"][0, 1:].abs().max().item() == 0.0)
    ntk = precompute_freqs_cis(d, 2, ntk_alpha=4.0, dtype=torch.float64)
    base = precompute_freqs_cis(d, 2, dtype=torch.float64)
    ntk_fast = abs(torch.angle(ntk[1, 0]).item() - torch.angle(base[1, 0]).item())
    ntk_slow = torch.angle(base[1, -1]).item() / torch.angle(ntk[1, -1]).item()
    c.add("NTK scaling keeps the fastest pair and slows the slowest by alpha", ntk_fast < 1e-12 and abs(ntk_slow - 4) < 1e-9,
          f"slowest-pair slowdown {ntk_slow:.6f}")
    pi = precompute_freqs_cis(d, 4 * 100 + 1, scale=4.0, dtype=torch.float64)
    c.add("Position Interpolation maps position 4m onto trained position m", (pi[400] - base_freq_row(d, 100)).abs().max().item() < 1e-12)

    say(f"--- {c.passed} passed, {c.failed} failed ---\n")
    return c, extra


# ---------------------------------------------------------------------------
# 7. command line
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_log(mode: str, payload: Dict) -> Path:
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ps03-rope-{mode}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mission 03 Part 2 - Rotary Positional Embeddings engine")
    p.add_argument("--selftest", action="store_true", help="run the verification suite only")
    p.add_argument("--plot", action="store_true", help="draw the RoPE vs ALiBi decay figure only")
    p.add_argument("--extrapolation", action="store_true", help="print the context-extrapolation table only")
    p.add_argument("--out", type=Path, default=DEFAULT_FIGURE, help="figure path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-log", action="store_true", help="do not write logs/*.json")
    args = p.parse_args(argv)
    run_all = not (args.selftest or args.plot or args.extrapolation)
    payload: Dict = {"torch": torch.__version__}
    status = 0

    if run_all:
        payload["demo"] = verify_relative_invariance(quiet=args.quiet)
    if run_all or args.selftest:
        checks, extra = self_test(args.seed, quiet=args.quiet)
        payload["checks"], payload["selftest_extra"] = checks.items, extra
        payload["passed"], payload["failed"] = checks.passed, checks.failed
        status = 1 if checks.failed else 0
    if run_all or args.extrapolation:
        cfg = {"dim": 128, "train_len": 4096, "target_len": 16384}
        rows = extrapolation_report(**cfg)
        if not args.quiet:
            print_extrapolation(rows, **cfg)
        payload["extrapolation"] = {"config": cfg, "rows": rows}
    if run_all or args.plot:
        fig = plot_decay(args.out)
        if fig is not None:
            maps = attention_maps()
            payload["figure"] = str(fig)
            payload["mean_attended_distance"] = {k: mean_attention_distance(maps[k]) for k in ("rope", "alibi", "none")}
            if not args.quiet:
                print(f"Decay figure written to {fig}")
                print("Mean attended distance in the heatmaps: " +
                      ", ".join(f"{k} {v:.1f}" for k, v in payload["mean_attended_distance"].items()))

    if not args.no_log:
        mode = "all" if run_all else "-".join(m for m, on in (("selftest", args.selftest), ("plot", args.plot),
                                                              ("extrapolation", args.extrapolation)) if on)
        log = write_log(mode, payload)
        if not args.quiet:
            print(f"Log written to {log.relative_to(HERE)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
