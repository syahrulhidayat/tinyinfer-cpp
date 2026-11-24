from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tinyinfer import _core
from tinyinfer._core import BuildInfo, KVCache, Q8Matrix, dequantize_q8_0, get_build_info

__all__ = [
    "BuildInfo",
    "KVCache",
    "Q8Matrix",
    "dequantize_q8_0",
    "get_build_info",
    "gqa_decode_step",
    "gqa_prefill",
    "matmul_fp32_nt",
    "matmul_q8_0_nt",
    "quantize_q8_0",
    "rmsnorm",
    "rope_inplace",
    "silu",
    "softmax",
]


def _as_f32_2d(arr: NDArray[np.floating]) -> NDArray[np.float32]:
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {out.shape}.")
    return out


def _as_f32_1d(arr: NDArray[np.floating]) -> NDArray[np.float32]:
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {out.shape}.")
    return out


def quantize_q8_0(weights: NDArray[np.floating]) -> Q8Matrix:
    return _core.quantize_q8_0(_as_f32_2d(weights))


def matmul_fp32_nt(x: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.float32]:
    return _core.matmul_fp32_nt(_as_f32_2d(x), _as_f32_2d(w))


def matmul_q8_0_nt(x: NDArray[np.floating], w_q8: Q8Matrix) -> NDArray[np.float32]:
    return _core.matmul_q8_0_nt(_as_f32_2d(x), w_q8)


def rmsnorm(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    eps: float = 1e-5,
) -> NDArray[np.float32]:
    return _core.rmsnorm(_as_f32_1d(x), _as_f32_1d(weight), eps)


def softmax(x: NDArray[np.floating]) -> NDArray[np.float32]:
    return _core.softmax(_as_f32_1d(x))


def silu(x: NDArray[np.floating]) -> NDArray[np.float32]:
    return _core.silu(_as_f32_1d(x))


def rope_inplace(
    x: NDArray[np.float32],
    *,
    n_heads: int,
    head_dim: int,
    pos: int,
    theta_base: float = 10000.0,
) -> None:
    if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
        raise ValueError("rope_inplace requires a C-contiguous float32 1D array.")
    _core.rope_inplace(x, n_heads, head_dim, pos, theta_base)


def gqa_decode_step(
    q: NDArray[np.floating],
    k_new: NDArray[np.floating],
    v_new: NDArray[np.floating],
    cache: KVCache,
    *,
    layer: int,
    pos: int,
    n_q_heads: int,
) -> NDArray[np.float32]:
    return _core.gqa_decode_step(
        _as_f32_1d(q),
        _as_f32_1d(k_new),
        _as_f32_1d(v_new),
        cache,
        layer,
        pos,
        n_q_heads,
    )


def gqa_prefill(
    q: NDArray[np.floating],
    k: NDArray[np.floating],
    v: NDArray[np.floating],
    cache: KVCache,
    *,
    layer: int,
    start_pos: int = 0,
    n_q_heads: int,
) -> NDArray[np.float32]:
    return _core.gqa_prefill(
        _as_f32_2d(q),
        _as_f32_2d(k),
        _as_f32_2d(v),
        cache,
        layer,
        start_pos,
        n_q_heads,
    )
