#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

E2E_START = "<!-- E2E_BENCHMARK_START -->"
E2E_END = "<!-- E2E_BENCHMARK_END -->"


def format_e2e_block(report: dict[str, Any]) -> str:
    header = (
        "| Backend | Weights (MB) | Prefill (ms) | Decode (ms) "
        "| Decode (tok/s) | Cosine Sim | Max Abs Drift |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows: list[str] = []
    for r in report["results"]:
        rows.append(
            f"| {r['backend']} | {r['weight_mb']:.2f} | {r['prefill_ms']:.2f} "
            f"| {r['decode_ms']:.2f} | {r['decode_tok_s']:.2f} "
            f"| {r['cosine_sim']:.6f} | {r['max_abs_drift']:.2e} |"
        )

    hw = report["hardware"]
    cfg = report["model_config"]
    footer = (
        f"*Model: {cfg['n_layers']}L, dim={cfg['dim']}, hidden={cfg['hidden_dim']}, "
        f"heads={cfg['n_q_heads']}/{cfg['n_kv_heads']}, vocab={cfg['vocab_size']} "
        f"(prompt={cfg['prompt_len']}, decode={cfg['max_new_tokens']} tok). "
        f"Hardware: {hw['cpu']} ({hw['threads']} threads), "
        f"AVX2+FMA={'ON' if hw['avx2_fma'] else 'OFF'}, "
        f"OpenMP={'ON' if hw['openmp'] else 'OFF'}, "
        f"peak RSS={hw['peak_rss_mb']:.2f} MB, {report['timestamp_utc']}.*"
    )
    return f"{E2E_START}\n{header}" + "\n".join(rows) + f"\n\n{footer}\n{E2E_END}"


def update_readme(json_path: Path, readme_path: Path) -> None:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    block = format_e2e_block(report)
    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(E2E_START) + r".*?" + re.escape(E2E_END),
        flags=re.DOTALL,
    )
    if not pattern.search(content):
        raise SystemExit(f"Markers '{E2E_START}' / '{E2E_END}' not found in {readme_path}.")
    readme_path.write_text(pattern.sub(block, content, count=1), encoding="utf-8")
    print(f"[UPDATED] E2E benchmark table written to {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write E2E benchmark JSON table into README.md.")
    parser.add_argument("--json", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    update_readme(args.json, args.readme)


if __name__ == "__main__":
    main()
