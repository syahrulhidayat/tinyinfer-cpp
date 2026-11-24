#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import tinyinfer

BENCH_START = "<!-- BENCHMARK_START -->"
BENCH_END = "<!-- BENCHMARK_END -->"


def _detect_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                if value.strip():
                    return value.strip()
    return platform.processor() or platform.machine()


def _hardware_summary() -> str:
    info = tinyinfer.get_build_info()
    cpu = _detect_cpu_model()
    threads = os.cpu_count() or 1
    stamp = datetime.now(UTC).strftime("%Y-%m-%d UTC")
    return (
        f"{cpu} ({threads} threads), AVX2+FMA={'ON' if info.avx2_fma else 'OFF'}, "
        f"OpenMP={'ON' if info.openmp else 'OFF'} ({info.max_threads} threads), "
        f"{platform.system()} {platform.release()}, measured on {stamp}"
    )


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


def _time_fn(
    fn: Callable[[], np.ndarray],
    *,
    warmup: int = 5,
    iters: int = 30,
) -> tuple[float, np.ndarray]:
    out = fn()
    for _ in range(warmup - 1):
        out = fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / iters
    return elapsed_ms, out


def _bench_workload(label: str, m: int, n: int, k: int, iters: int) -> list[str]:
    rng = np.random.default_rng(2026)
    x = rng.standard_normal((m, k), dtype=np.float32)
    w = rng.standard_normal((n, k), dtype=np.float32)
    w_q8 = tinyinfer.quantize_q8_0(w)

    flops = 2.0 * m * n * k
    fp32_mb = w.nbytes / (1024.0 * 1024.0)
    q8_mb = w_q8.memory_bytes() / (1024.0 * 1024.0)

    np_ms, ref = _time_fn(lambda: x @ w.T, iters=iters)
    # Let OpenBLAS worker threads exit spin-wait before OpenMP kernels run.
    time.sleep(0.25)
    ti_fp32_ms, out_fp32 = _time_fn(lambda: tinyinfer.matmul_fp32_nt(x, w), iters=iters)
    ti_q8_ms, out_q8 = _time_fn(lambda: tinyinfer.matmul_q8_0_nt(x, w_q8), iters=iters)

    entries = [
        ("NumPy FP32", np_ms, fp32_mb, 1.0, 0.0),
        (
            "tinyinfer FP32",
            ti_fp32_ms,
            fp32_mb,
            _cosine_sim(out_fp32, ref),
            float(np.max(np.abs(out_fp32 - ref))),
        ),
        (
            "tinyinfer Q8_0",
            ti_q8_ms,
            q8_mb,
            _cosine_sim(out_q8, ref),
            float(np.max(np.abs(out_q8 - ref))),
        ),
    ]

    rows: list[str] = []
    for impl, ms, mem_mb, cos, drift in entries:
        gflops = flops / (ms * 1e6)
        rows.append(
            f"| {label} | {impl} | {ms:.3f} | {gflops:.2f} | "
            f"{mem_mb:.2f} | {cos:.6f} | {drift:.2e} |"
        )
    return rows


def _update_readme(readme_path: Path, block: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(BENCH_START) + r".*?" + re.escape(BENCH_END),
        flags=re.DOTALL,
    )
    if not pattern.search(content):
        raise SystemExit(f"Markers '{BENCH_START}' / '{BENCH_END}' not found in {readme_path}.")
    readme_path.write_text(pattern.sub(block, content, count=1), encoding="utf-8")
    print(f"[UPDATED] Benchmark table written to {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tinyinfer FP32/Q8_0 kernels vs NumPy.")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--update-readme", type=Path, default=None)
    args = parser.parse_args()

    header = (
        "| Kernel Workload | Implementation | Mean Latency (ms) | Throughput (GFLOP/s) "
        "| Weight Memory (MB) | Cosine Sim vs FP32 | Max Abs Drift |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows = []
    rows.extend(_bench_workload("Decode GEMV (`1x2048 @ 2048x2048`)", 1, 2048, 2048, args.iters))
    rows.extend(_bench_workload("Prefill GEMM (`32x1024 @ 1024x1024`)", 32, 1024, 1024, args.iters))

    hw = _hardware_summary()
    table = header + "\n".join(rows) + f"\n\n*Hardware baseline: {hw}.*"
    print(table)

    if args.update_readme is not None:
        _update_readme(args.update_readme, f"{BENCH_START}\n{table}\n{BENCH_END}")


if __name__ == "__main__":
    main()
