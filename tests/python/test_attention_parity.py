from __future__ import annotations

import math

import numpy as np
import pytest

import tinyinfer


def _numpy_causal_gqa_ref(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_q_heads: int,
    n_kv_heads: int,
) -> np.ndarray:
    seq_len, q_dim = q.shape
    head_dim = q_dim // n_q_heads
    group_size = n_q_heads // n_kv_heads

    qh = q.reshape(seq_len, n_q_heads, head_dim).astype(np.float64)
    kh = np.repeat(k.reshape(seq_len, n_kv_heads, head_dim), group_size, axis=1).astype(np.float64)
    vh = np.repeat(v.reshape(seq_len, n_kv_heads, head_dim), group_size, axis=1).astype(np.float64)

    out = np.zeros_like(qh)
    scale = 1.0 / math.sqrt(head_dim)
    for h in range(n_q_heads):
        scores = (qh[:, h, :] @ kh[:, h, :].T) * scale
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask, -1e30, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        out[:, h, :] = probs @ vh[:, h, :]
    return out.reshape(seq_len, q_dim).astype(np.float32)


@pytest.mark.parametrize(
    ("n_q_heads", "n_kv_heads", "head_dim"),
    [(8, 8, 32), (8, 2, 64), (8, 1, 36)],
)
def test_gqa_prefill_and_decode_parity_against_numpy(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> None:
    rng = np.random.default_rng(2026)
    seq_len = 10
    prompt_len = 6

    q = rng.standard_normal((seq_len, n_q_heads * head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, n_kv_heads * head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, n_kv_heads * head_dim), dtype=np.float32)

    ref = _numpy_causal_gqa_ref(q, k, v, n_q_heads, n_kv_heads)

    # 1. Full prefill check
    cache_full = tinyinfer.KVCache(2, 32, n_kv_heads, head_dim)
    out_full = tinyinfer.gqa_prefill(
        q, k, v, cache_full, layer=0, start_pos=0, n_q_heads=n_q_heads
    )
    assert cache_full.seq_len(0) == seq_len
    assert np.allclose(out_full, ref, atol=1e-5, rtol=1e-4)

    # 2. Prompt prefill followed by autoregressive token-by-token decode check
    cache_inc = tinyinfer.KVCache(2, 32, n_kv_heads, head_dim)
    out_prompt = tinyinfer.gqa_prefill(
        q[:prompt_len],
        k[:prompt_len],
        v[:prompt_len],
        cache_inc,
        layer=0,
        start_pos=0,
        n_q_heads=n_q_heads,
    )
    decoded_steps = [
        tinyinfer.gqa_decode_step(
            q[pos],
            k[pos],
            v[pos],
            cache_inc,
            layer=0,
            pos=pos,
            n_q_heads=n_q_heads,
        )
        for pos in range(prompt_len, seq_len)
    ]
    out_combined = np.vstack([out_prompt, np.stack(decoded_steps, axis=0)])
    assert cache_inc.seq_len(0) == seq_len
    assert np.allclose(out_combined, ref, atol=1e-5, rtol=1e-4)


def test_kv_cache_memory_and_bounds_validation() -> None:
    cache_gqa = tinyinfer.KVCache(n_layers=4, max_seq_len=128, n_kv_heads=2, head_dim=64)
    cache_mha = tinyinfer.KVCache(n_layers=4, max_seq_len=128, n_kv_heads=8, head_dim=64)
    assert cache_mha.memory_bytes() == 4 * cache_gqa.memory_bytes()

    q = np.zeros(8 * 64, dtype=np.float32)
    kv = np.zeros(2 * 64, dtype=np.float32)

    with pytest.raises(IndexError):
        tinyinfer.gqa_decode_step(q, kv, kv, cache_gqa, layer=0, pos=128, n_q_heads=8)

    with pytest.raises(ValueError):
        tinyinfer.gqa_decode_step(
            np.zeros(5 * 64, dtype=np.float32), kv, kv, cache_gqa, layer=0, pos=0, n_q_heads=5
        )
