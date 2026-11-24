from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dashboard.server import create_server


def test_dashboard_serves_html_and_results_json(tmp_path: Path) -> None:
    sample = {
        "timestamp_utc": "2026-11-25T00:00:00Z",
        "hardware": {
            "cpu": "Test CPU",
            "threads": 4,
            "avx2_fma": True,
            "openmp": True,
            "cuda_device": "none",
            "peak_rss_mb": 32.0,
        },
        "model_config": {
            "n_layers": 2,
            "dim": 64,
            "hidden_dim": 128,
            "n_q_heads": 4,
            "n_kv_heads": 2,
            "vocab_size": 96,
            "prompt_len": 4,
            "max_new_tokens": 8,
        },
        "results": [
            {
                "backend": "tinyinfer Q8_0 (mmap)",
                "weight_mb": 1.2,
                "prefill_ms": 0.5,
                "decode_ms": 1.0,
                "decode_tok_s": 8000.0,
                "cosine_sim": 0.9998,
                "max_abs_drift": 0.01,
            }
        ],
    }
    results_file = tmp_path / "latest.json"
    results_file.write_text(json.dumps(sample), encoding="utf-8")

    server = create_server("127.0.0.1", 0, results_file)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "tinyinfer-cpp Benchmark Telemetry" in html

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/results") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["results"][0]["backend"] == "tinyinfer Q8_0 (mmap)"

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/missing")
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
