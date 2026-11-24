from __future__ import annotations

import math

import numpy as np
import pytest

import tinyinfer


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)))


@pytest.mark.parametrize(("m", "n", "k"), [(1, 128, 256), (4, 64, 77), (8, 96, 128)])
def test_matmul_fp32_and_q8_parity_against_numpy(m: int, n: int, k: int) -> None:
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((m, k), dtype=np.float32)
    w = rng.standard_normal((n, k), dtype=np.float32)

    ref = x @ w.T
    out_fp32 = tinyinfer.matmul_fp32_nt(x, w)
    assert np.allclose(out_fp32, ref, atol=1e-4, rtol=1e-4)

    w_q8 = tinyinfer.quantize_q8_0(w)
    out_q8 = tinyinfer.matmul_q8_0_nt(x, w_q8)
    assert _cosine_sim(out_q8, ref) > 0.999
    expected_ratio = (4.0 * k) / (36.0 * math.ceil(k / 32))
    assert w_q8.compression_ratio() == pytest.approx(expected_ratio, rel=1e-6)
    if k % 32 == 0:
        assert w_q8.compression_ratio() > 3.5


def test_dequantize_roundtrip_stays_within_q8_bound() -> None:
    rng = np.random.default_rng(7)
    w = rng.uniform(-2.0, 2.0, size=(32, 64)).astype(np.float32)
    q = tinyinfer.quantize_q8_0(w)
    recon = tinyinfer.dequantize_q8_0(q)
    max_err = float(np.max(np.abs(w - recon)))
    # Symmetric 7-bit step size <= 2.0 / 127 => half-step <= 0.008
    assert max_err < 0.01


def test_rmsnorm_softmax_silu_rope_parity() -> None:
    rng = np.random.default_rng(99)
    x = rng.standard_normal(64, dtype=np.float32)
    weight = rng.uniform(0.5, 1.5, size=64).astype(np.float32)

    rms_ref = (x / np.sqrt(np.mean(x**2) + 1e-5)) * weight
    assert np.allclose(tinyinfer.rmsnorm(x, weight, eps=1e-5), rms_ref, atol=1e-5)

    shifted = x - np.max(x)
    sm_ref = np.exp(shifted) / np.sum(np.exp(shifted))
    assert np.allclose(tinyinfer.softmax(x), sm_ref, atol=1e-5)

    silu_ref = x / (1.0 + np.exp(-x))
    assert np.allclose(tinyinfer.silu(x), silu_ref, atol=1e-5)

    rope_vec = x.copy()
    before_norm = float(np.linalg.norm(rope_vec))
    tinyinfer.rope_inplace(rope_vec, n_heads=4, head_dim=16, pos=11)
    assert float(np.linalg.norm(rope_vec)) == pytest.approx(before_norm, rel=1e-5)
    assert not np.allclose(rope_vec, x)
