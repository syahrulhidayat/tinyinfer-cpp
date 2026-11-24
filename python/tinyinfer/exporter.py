from __future__ import annotations

import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyinfer import _core
from tinyinfer._core import ModelConfig, WeightDType

TINF_HEADER_STRUCT = struct.Struct("<4s10I2f3I")


def _write_f32(f, arr: NDArray[np.floating], expected_shape: tuple[int, ...], name: str) -> None:
    buf = np.ascontiguousarray(arr, dtype=np.float32)
    if buf.shape != expected_shape:
        raise ValueError(f"Weight '{name}' has shape {buf.shape}, expected {expected_shape}.")
    f.write(buf.tobytes())


def _write_linear(
    f,
    arr: NDArray[np.floating],
    expected_shape: tuple[int, int],
    name: str,
    dtype: WeightDType,
) -> None:
    buf = np.ascontiguousarray(arr, dtype=np.float32)
    if buf.shape != expected_shape:
        raise ValueError(f"Weight '{name}' has shape {buf.shape}, expected {expected_shape}.")
    if dtype == WeightDType.FP32:
        f.write(buf.tobytes())
    else:
        scales, q_weights = _core.quantize_q8_0_arrays(buf)
        f.write(np.ascontiguousarray(scales, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(q_weights, dtype=np.int8).tobytes())


def export_tinf(
    path: str | Path,
    *,
    config: ModelConfig,
    weights: Mapping[str, NDArray[np.floating]],
    dtype: Literal["fp32", "q8_0"] = "fp32",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weight_dtype = WeightDType.FP32 if dtype == "fp32" else WeightDType.Q8_0
    dtype_code = 0 if weight_dtype == WeightDType.FP32 else 1

    dim = int(config.dim)
    hidden_dim = int(config.hidden_dim)
    vocab_size = int(config.vocab_size)
    n_layers = int(config.n_layers)
    n_q_heads = int(config.n_q_heads)
    n_kv_heads = int(config.n_kv_heads)
    head_dim = dim // n_q_heads
    kv_dim = n_kv_heads * head_dim

    header = TINF_HEADER_STRUCT.pack(
        b"TINF",
        1,
        dtype_code,
        vocab_size,
        dim,
        hidden_dim,
        n_layers,
        n_q_heads,
        n_kv_heads,
        int(config.max_seq_len),
        1 if config.tie_embeddings else 0,
        float(config.rms_norm_eps),
        float(config.rope_theta),
        0,
        0,
        0,
    )

    with out_path.open("wb") as f:
        f.write(header)
        _write_f32(f, weights["tok_embeddings"], (vocab_size, dim), "tok_embeddings")

        for layer in range(n_layers):
            prefix = f"layers.{layer}"
            _write_f32(f, weights[f"{prefix}.attn_norm"], (dim,), f"{prefix}.attn_norm")
            _write_f32(f, weights[f"{prefix}.ffn_norm"], (dim,), f"{prefix}.ffn_norm")
            _write_linear(f, weights[f"{prefix}.wq"], (dim, dim), f"{prefix}.wq", weight_dtype)
            _write_linear(f, weights[f"{prefix}.wk"], (kv_dim, dim), f"{prefix}.wk", weight_dtype)
            _write_linear(f, weights[f"{prefix}.wv"], (kv_dim, dim), f"{prefix}.wv", weight_dtype)
            _write_linear(f, weights[f"{prefix}.wo"], (dim, dim), f"{prefix}.wo", weight_dtype)
            _write_linear(
                f, weights[f"{prefix}.w_gate"], (hidden_dim, dim), f"{prefix}.w_gate", weight_dtype
            )
            _write_linear(
                f, weights[f"{prefix}.w_up"], (hidden_dim, dim), f"{prefix}.w_up", weight_dtype
            )
            _write_linear(
                f, weights[f"{prefix}.w_down"], (dim, hidden_dim), f"{prefix}.w_down", weight_dtype
            )

        _write_f32(f, weights["final_norm"], (dim,), "final_norm")
        if not config.tie_embeddings:
            _write_linear(f, weights["lm_head"], (vocab_size, dim), "lm_head", weight_dtype)

    return out_path
