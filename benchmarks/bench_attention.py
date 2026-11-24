#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import numpy as np

import tinyinfer


def _numpy_decode_step_ref(
    q: np.ndarray,
    k_hist: np.ndarray,
    v_hist: np.ndarray,
    n_q_heads: int,
    n_kv_heads: int,
) -> np.ndarray:
    seq_len = k_hist.shape[0]
    head_dim = q.shape[0] // n_q_heads
    group = n_q_heads // n_kv_heads

    qh = q.reshape(n_q_heads, head_dim)
    kh = np.repeat(k_hist.reshape(seq_len, n_kv_heads, head_dim), group, axis=1)
    vh = np.repeat(v_hist.reshape(seq_len, n_kv_heads, head_dim), group, axis=1)

    scores = np.einsum("hd,thd->ht", qh, kh) / math.sqrt(head_dim)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("ht,thd->hd", probs, vh).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GQA decode attention with KV-Cache.")
    parser.add_argument("--n-q-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()

    max_seq = 2048
    n_layers = 1
    cache_gqa = tinyinfer.KVCache(n_layers, max_seq, args.n_kv_heads, args.head_dim)
    cache_mha = tinyinfer.KVCache(n_layers, max_seq, args.n_q_heads, args.head_dim)

    gqa_mb = cache_gqa.memory_bytes() / (1024.0 * 1024.0)
    mha_mb = cache_mha.memory_bytes() / (1024.0 * 1024.0)
    print(
        f"KV-Cache Footprint (1 layer, ctx={max_seq}, head_dim={args.head_dim}): "
        f"GQA ({args.n_kv_heads} kv heads) = {gqa_mb:.2f} MB vs "
        f"MHA ({args.n_q_heads} kv heads) = {mha_mb:.2f} MB ({mha_mb / gqa_mb:.1f}x reduction)\n"
    )

    rng = np.random.default_rng(2026)
    q_dim = args.n_q_heads * args.head_dim
    kv_dim = args.n_kv_heads * args.head_dim

    q_all = rng.standard_normal((max_seq, q_dim), dtype=np.float32)
    k_all = rng.standard_normal((max_seq, kv_dim), dtype=np.float32)
    v_all = rng.standard_normal((max_seq, kv_dim), dtype=np.float32)

    print(
        "| Context Len | NumPy GQA (ms) | tinyinfer GQA (ms) | Speedup | Max Abs Drift |\n"
        "| :---: | :---: | :---: | :---: | :---: |"
    )

    for ctx in (128, 512, 1024, 2048):
        cache_gqa.reset()
        tinyinfer.gqa_prefill(
            q_all[: ctx - 1],
            k_all[: ctx - 1],
            v_all[: ctx - 1],
            cache_gqa,
            layer=0,
            start_pos=0,
            n_q_heads=args.n_q_heads,
        )

        q_tok = q_all[ctx - 1]
        k_tok = k_all[ctx - 1]
        v_tok = v_all[ctx - 1]

        t0 = time.perf_counter()
        ref = _numpy_decode_step_ref(
            q_tok, k_all[:ctx], v_all[:ctx], args.n_q_heads, args.n_kv_heads
        )
        for _ in range(args.iters - 1):
            ref = _numpy_decode_step_ref(
                q_tok, k_all[:ctx], v_all[:ctx], args.n_q_heads, args.n_kv_heads
            )
        np_ms = ((time.perf_counter() - t0) * 1000.0) / args.iters

        time.sleep(0.15)

        t1 = time.perf_counter()
        out = tinyinfer.gqa_decode_step(
            q_tok, k_tok, v_tok, cache_gqa, layer=0, pos=ctx - 1, n_q_heads=args.n_q_heads
        )
        for _ in range(args.iters - 1):
            out = tinyinfer.gqa_decode_step(
                q_tok, k_tok, v_tok, cache_gqa, layer=0, pos=ctx - 1, n_q_heads=args.n_q_heads
            )
        ti_ms = ((time.perf_counter() - t1) * 1000.0) / args.iters

        drift = float(np.max(np.abs(out - ref)))
        print(f"| {ctx} | {np_ms:.3f} | {ti_ms:.3f} | {np_ms / ti_ms:.2f}x | {drift:.2e} |")


if __name__ == "__main__":
    main()
