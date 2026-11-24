#!/usr/bin/env python3
# Build and benchmark tinyinfer CUDA kernels on Google Colab / Kaggle GPUs.
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np


def _build_with_cuda() -> None:
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DTINYINFER_ENABLE_CUDA=ON"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
        env=env,
        check=True,
    )


def main() -> None:
    _build_with_cuda()
    import tinyinfer

    print("BuildInfo :", tinyinfer.get_build_info().cuda, tinyinfer.cuda_device_name())
    if not tinyinfer.cuda_available():
        raise SystemExit("No CUDA GPU detected in this runtime.")

    rng = np.random.default_rng(2026)
    x = rng.standard_normal((1, 4096), dtype=np.float32)
    w = rng.standard_normal((4096, 4096), dtype=np.float32)
    w_q8 = tinyinfer.quantize_q8_0(w)

    ref = x @ w.T
    out_fp32 = tinyinfer.cuda_matmul_fp32_nt(x, w)
    out_q8 = tinyinfer.cuda_matmul_q8_0_nt(x, w_q8)

    assert np.allclose(out_fp32, ref, atol=1e-3, rtol=1e-3)
    cos = float(
        np.dot(out_q8.ravel(), ref.ravel())
        / (np.linalg.norm(out_q8) * np.linalg.norm(ref))
    )

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        tinyinfer.cuda_matmul_fp32_nt(x, w)
    fp32_ms = (time.perf_counter() - t0) * 1000.0 / iters

    t1 = time.perf_counter()
    for _ in range(iters):
        tinyinfer.cuda_matmul_q8_0_nt(x, w_q8)
    q8_ms = (time.perf_counter() - t1) * 1000.0 / iters

    print(f"CUDA GEMV [1x4096 @ 4096x4096^T] FP32 : {fp32_ms:.3f} ms")
    print(f"CUDA GEMV [1x4096 @ 4096x4096^T] Q8_0 : {q8_ms:.3f} ms (cosine_sim={cos:.6f})")


if __name__ == "__main__":
    main()
