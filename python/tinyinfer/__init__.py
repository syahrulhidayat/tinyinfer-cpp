from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tinyinfer._core import (
    BuildInfo,
    Q8Matrix,
    dequantize_q8_0,
    get_build_info,
)
from tinyinfer._core import (
    matmul_fp32_nt as _matmul_fp32_nt,
)
from tinyinfer._core import (
    matmul_q8_0_nt as _matmul_q8_0_nt,
)
from tinyinfer._core import (
    quantize_q8_0 as _quantize_q8_0,
)
from tinyinfer._core import (
    rmsnorm as _rmsnorm,
)
from tinyinfer._core import (
    rope_inplace as _rope_inplace,
)
from tinyinfer._core import (
    silu as _silu,
)
from tinyinfer._core import (
    softmax as _softmax,
)

__all__ = [
    "BuildInfo",
    "Q8Matrix",
    "dequantize_q8_0",
    "get_build_info",
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
    return _quantize_q8_0(_as_f32_2d(weights))


def matmul_fp32_nt(x: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.float32]:
    return _matmul_fp32_nt(_as_f32_2d(x), _as_f32_2d(w))


def matmul_q8_0_nt(x: NDArray[np.floating], w_q8: Q8Matrix) -> NDArray[np.float32]:
    return _matmul_q8_0_nt(_as_f32_2d(x), w_q8)


def rmsnorm(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    eps: float = 1e-5,
) -> NDArray[np.float32]:
    return _rmsnorm(_as_f32_1d(x), _as_f32_1d(weight), eps)


def softmax(x: NDArray[np.floating]) -> NDArray[np.float32]:
    return _softmax(_as_f32_1d(x))


def silu(x: NDArray[np.floating]) -> NDArray[np.float32]:
    return _silu(_as_f32_1d(x))


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
    _rope_inplace(x, n_heads, head_dim, pos, theta_base)
