#!/usr/bin/env python3
"""
03_GQA_and_KVCache_Benchmark_starter.py

Part 3 deliverable: Grouped-Query Attention, a stateful KV cache, and a
memory/latency benchmark for MHA, GQA (4:1), and MQA (32:1).

Run:
    python 03_GQA_and_KVCache_Benchmark_starter.py

The cache-size and bandwidth columns are analytical estimates. The latency
column is measured on the current device and is therefore hardware-dependent.
This is a small inference microbenchmark, not a production serving benchmark.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    """Causal self-attention with Hq query heads and Hkv key/value heads.

    Hq must be divisible by Hkv. The same implementation supports:
      * MHA: Hq == Hkv
      * GQA: Hq > Hkv, e.g. 32 query heads / 8 KV heads
      * MQA: Hkv == 1
    Input/output shape: (batch, sequence, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("Head counts must be positive")
        if d_model % num_query_heads:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_query_heads * self.head_dim, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        causal: bool = True,
    ):
        """Return output, or (output, (key, value)) when use_cache=True.

        Cache tensors use shape (batch, Hkv, cached_sequence, head_dim).
        If past_key_value is supplied, x may contain one or more new tokens.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x shaped (batch, sequence, {self.d_model})")

        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch, seq_len, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        k_new = self.k_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v_new = self.v_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.shape[:2] != (batch, self.num_kv_heads):
                raise ValueError("past key cache has incompatible batch/head dimensions")
            if past_v.shape[:2] != (batch, self.num_kv_heads):
                raise ValueError("past value cache has incompatible batch/head dimensions")
            if past_k.shape[-1] != self.head_dim or past_v.shape[-1] != self.head_dim:
                raise ValueError("past cache has incompatible head dimension")
            k = torch.cat((past_k, k_new), dim=2)
            v = torch.cat((past_v, v_new), dim=2)
            past_len = past_k.shape[2]
        else:
            k, v = k_new, v_new
            past_len = 0

        # Expand KV heads to query-head count. expand() itself is a view; the
        # following attention operations may materialize intermediates.
        k_grouped = k.repeat_interleave(self.group_size, dim=1)
        v_grouped = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k_grouped.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            # For cached decoding, query position i has absolute index past_len+i.
            key_positions = torch.arange(k.shape[2], device=x.device)
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        probs = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        attended = torch.matmul(probs, v_grouped)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.o_proj(attended)

        if use_cache:
            return output, (k, v)
        return output


class KVCache:
    """Append-only KV cache for autoregressive decoding.

    Stored shape is (batch, Hkv, sequence, head_dim). append() accepts new
    token slices with shape (batch, Hkv, new_tokens, head_dim).
    This simple implementation uses torch.cat, which is clear and correct but
    copies the existing cache on every append. For high-throughput serving,
    preallocated or paged storage is generally preferable.
    """

    def __init__(self) -> None:
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value must have shape (batch, heads, seq, head_dim)")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if self.key is None:
            self.key, self.value = key, value
        else:
            assert self.value is not None
            if key.shape[:2] != self.key.shape[:2] or key.shape[3] != self.key.shape[3]:
                raise ValueError("new key slice is incompatible with cache")
            if key.device != self.key.device or key.dtype != self.key.dtype:
                raise ValueError("new key slice must match cache device and dtype")
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        return self.key, self.value

    def clear(self) -> None:
        self.key = None
        self.value = None

    def memory_bytes(self) -> int:
        if self.key is None or self.value is None:
            return 0
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


@dataclass
class BenchmarkRow:
    attention: str
    kv_heads: int
    sequence_length: int
    cache_mb: float
    traffic_gbs_at_50_tps: float
    latency_ms: float


def cache_memory_bytes(
    sequence_length: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Bytes for K and V tensors, excluding allocator/metadata overhead."""
    element_bytes = torch.empty((), dtype=dtype).element_size()
    return 2 * sequence_length * batch_size * num_kv_heads * head_dim * element_bytes


@torch.inference_mode()
def measure_decode_latency_ms(
    *,
    sequence_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    """Measure attention-only one-token decode latency over an existing cache.

    Projection and cache-construction time are excluded. KV head expansion is
    included, so the result reflects this straightforward PyTorch reference
    implementation rather than an optimized fused GQA kernel.
    """
    batch = 1
    q = torch.randn(batch, num_query_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, sequence_length, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    group_size = num_query_heads // num_kv_heads

    def step():
        kg = k.repeat_interleave(group_size, dim=1)
        vg = v.repeat_interleave(group_size, dim=1)
        scores = torch.matmul(q, kg.transpose(-2, -1)) / math.sqrt(head_dim)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=dtype)
        return torch.matmul(weights, vg)

    for _ in range(warmup):
        out = step()
        del out
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        del out
    return sorted(samples)[len(samples) // 2]


def run_benchmark() -> list[BenchmarkRow]:
    # 32 query heads; head_dim=16 gives d_model=512 and makes the reference
    # benchmark usable on a wider range of machines.
    hq = 32
    head_dim = 16
    batch_size = 1
    sequence_lengths = (1024, 4096, 16384, 32768)
    variants = (("MHA", 32), ("GQA 4:1", 8), ("MQA 32:1", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device} | dtype: {dtype} | batch: {batch_size} | "
          f"Hq: {hq} | head_dim: {head_dim}")
    print("Cache and bandwidth are analytical estimates; latency is measured.")
    print("Latency measures attention-only one-token decode, median of 5 runs.")
    rows: list[BenchmarkRow] = []

    for n in sequence_lengths:
        for name, hkv in variants:
            cache_bytes = cache_memory_bytes(n, batch_size, hkv, head_dim, dtype)
            cache_mb = cache_bytes / (1024 ** 2)
            # One token decode reads approximately all cached K and V once.
            # This is estimated traffic rate at the stated generation rate,
            # not a measurement of the device's achieved bandwidth.
            traffic_gbs = cache_bytes * 50.0 / 1e9
            try:
                latency = measure_decode_latency_ms(
                    sequence_length=n,
                    num_query_heads=hq,
                    num_kv_heads=hkv,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else RuntimeError) as exc:
                latency = float("nan")
                print(f"Warning: latency benchmark skipped for {name}, N={n}: {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(BenchmarkRow(name, hkv, n, cache_mb, traffic_gbs, latency))

    print()
    header = (
        f"{'Attention':<12} {'N':>7} {'Hkv':>5} "
        f"{'KV cache (MB)':>15} {'Traffic @50 tok/s (GB/s)':>25} "
        f"{'Latency/token (ms)':>20}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        latency = f"{r.latency_ms:.3f}" if math.isfinite(r.latency_ms) else "N/A"
        print(
            f"{r.attention:<12} {r.sequence_length:>7,d} {r.kv_heads:>5d} "
            f"{r.cache_mb:>15.2f} {r.traffic_gbs_at_50_tps:>25.4f} {latency:>20}"
        )
    print("\nNotes:")
    print("  * Cache MB uses MiB (bytes / 1024^2), despite the compact MB label.")
    print("  * Traffic is estimated KV bytes read per token × 50 tokens/sec / 1e9.")
    print("  * Latency depends on hardware, PyTorch build, dtype, and memory pressure.")
    print("  * This reference implementation expands KV heads; fused kernels may be faster.")
    return rows


def _self_test() -> None:
    torch.manual_seed(7)
    # Check grouping, output shape, cache growth, and one-token cached decoding.
    layer = GQAAttention(d_model=64, num_query_heads=8, num_kv_heads=2).eval()
    x = torch.randn(2, 5, 64)
    y, (k, v) = layer(x, use_cache=True)
    assert y.shape == x.shape
    assert k.shape == (2, 2, 5, 8)
    assert v.shape == (2, 2, 5, 8)

    cache = KVCache()
    k_all, v_all = cache.append(k[:, :, :4], v[:, :, :4])
    k_all, v_all = cache.append(k[:, :, 4:5], v[:, :, 4:5])
    assert cache.length == 5
    assert k_all.shape == k.shape and v_all.shape == v.shape

    token_out, (k2, v2) = layer(x[:, 5-1:5], past_key_value=(k[:, :, :4], v[:, :, :4]), use_cache=True)
    assert token_out.shape == (2, 1, 64)
    assert k2.shape[2] == 5 and v2.shape[2] == 5
    print("Self-test: PASS (GQA shapes, causal forward, cache append, cached decode)")


if __name__ == "__main__":
    _self_test()
    run_benchmark()
