from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tinyinfer


def test_cuda_build_info_and_graceful_fallback() -> None:
    info = tinyinfer.get_build_info()
    assert isinstance(info.cuda, bool)
    assert isinstance(tinyinfer.cuda_device_name(), str)

    x = np.ones((1, 32), dtype=np.float32)
    w = np.ones((16, 32), dtype=np.float32)
    w_q8 = tinyinfer.quantize_q8_0(w)

    if tinyinfer.cuda_available():
        out_fp32 = tinyinfer.cuda_matmul_fp32_nt(x, w)
        out_q8 = tinyinfer.cuda_matmul_q8_0_nt(x, w_q8)
        assert np.allclose(out_fp32, x @ w.T, atol=1e-3)
        assert np.allclose(out_q8, x @ w.T, atol=0.5)
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            tinyinfer.cuda_matmul_fp32_nt(x, w)
        with pytest.raises(RuntimeError, match="CUDA"):
            tinyinfer.cuda_matmul_q8_0_nt(x, w_q8)


def test_e2e_benchmark_script_outputs_valid_json(tmp_path: Path) -> None:
    out_json = tmp_path / "bench.json"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "bench_e2e_compare.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--layers",
            "2",
            "--dim",
            "64",
            "--hidden-dim",
            "128",
            "--n-q-heads",
            "4",
            "--n-kv-heads",
            "2",
            "--vocab-size",
            "96",
            "--prompt-len",
            "4",
            "--max-tokens",
            "6",
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "tinyinfer Q8_0 (mmap)" in proc.stdout

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "hardware" in data
    assert len(data["results"]) >= 3
    q8_entry = next(r for r in data["results"] if "Q8_0" in r["backend"])
    assert q8_entry["cosine_sim"] > 0.99
    assert q8_entry["decode_tok_s"] > 0.0
