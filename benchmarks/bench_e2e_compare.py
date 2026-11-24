#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import tinyinfer


def _detect_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                if value.strip():
                    return value.strip()
    return platform.processor() or platform.machine()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _make_weights(cfg: tinyinfer.ModelConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = cfg.dim
    hidden = cfg.hidden_dim
    kv_dim = cfg.n_kv_heads * (dim // cfg.n_q_heads)

    w: dict[str, np.ndarray] = {
        "tok_embeddings": rng.normal(0.0, 0.04, (cfg.vocab_size, dim)).astype(np.float32),
        "final_norm": rng.uniform(0.85, 1.15, (dim,)).astype(np.float32),
    }
    for layer in range(cfg.n_layers):
        p = f"layers.{layer}"
        w[f"{p}.attn_norm"] = rng.uniform(0.85, 1.15, (dim,)).astype(np.float32)
        w[f"{p}.ffn_norm"] = rng.uniform(0.85, 1.15, (dim,)).astype(np.float32)
        w[f"{p}.wq"] = rng.normal(0.0, 0.04, (dim, dim)).astype(np.float32)
        w[f"{p}.wk"] = rng.normal(0.0, 0.04, (kv_dim, dim)).astype(np.float32)
        w[f"{p}.wv"] = rng.normal(0.0, 0.04, (kv_dim, dim)).astype(np.float32)
        w[f"{p}.wo"] = rng.normal(0.0, 0.04, (dim, dim)).astype(np.float32)
        w[f"{p}.w_gate"] = rng.normal(0.0, 0.04, (hidden, dim)).astype(np.float32)
        w[f"{p}.w_up"] = rng.normal(0.0, 0.04, (hidden, dim)).astype(np.float32)
        w[f"{p}.w_down"] = rng.normal(0.0, 0.04, (dim, hidden)).astype(np.float32)
    if not cfg.tie_embeddings:
        w["lm_head"] = rng.normal(0.0, 0.04, (cfg.vocab_size, dim)).astype(np.float32)
    return w


def _rmsnorm_np(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return ((x / rms) * w).astype(np.float32)


def _rope_np(vec: np.ndarray, n_heads: int, head_dim: int, pos: int, theta: float) -> np.ndarray:
    out = vec.copy().reshape(n_heads, head_dim)
    half = head_dim // 2
    idx = np.arange(half, dtype=np.float64)
    angles = pos / (theta ** ((2.0 * idx) / head_dim))
    c = np.cos(angles).astype(np.float32)
    s = np.sin(angles).astype(np.float32)
    v0 = out[:, 0::2].copy()
    v1 = out[:, 1::2].copy()
    out[:, 0::2] = v0 * c - v1 * s
    out[:, 1::2] = v0 * s + v1 * c
    return out.reshape(-1)


def _run_numpy_baseline(
    cfg: tinyinfer.ModelConfig,
    weights: dict[str, np.ndarray],
    prompt: list[int],
    max_new_tokens: int,
) -> tuple[dict[str, Any], np.ndarray]:
    dim = cfg.dim
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    head_dim = dim // n_q
    group = n_q // n_kv
    max_ctx = len(prompt) + max_new_tokens
    k_cache = [np.zeros((max_ctx, n_kv, head_dim), dtype=np.float32) for _ in range(cfg.n_layers)]
    v_cache = [np.zeros((max_ctx, n_kv, head_dim), dtype=np.float32) for _ in range(cfg.n_layers)]

    def step(tok: int, pos: int) -> np.ndarray:
        x = weights["tok_embeddings"][tok].copy()
        for layer in range(cfg.n_layers):
            p = f"layers.{layer}"
            xb = _rmsnorm_np(x, weights[f"{p}.attn_norm"], cfg.rms_norm_eps)
            q = _rope_np(weights[f"{p}.wq"] @ xb, n_q, head_dim, pos, cfg.rope_theta)
            k = _rope_np(weights[f"{p}.wk"] @ xb, n_kv, head_dim, pos, cfg.rope_theta)
            v = weights[f"{p}.wv"] @ xb
            k_cache[layer][pos] = k.reshape(n_kv, head_dim)
            v_cache[layer][pos] = v.reshape(n_kv, head_dim)

            qh = q.reshape(n_q, head_dim)
            att = np.zeros_like(qh)
            for h in range(n_q):
                kvh = h // group
                kh = k_cache[layer][: pos + 1, kvh, :]
                vh = v_cache[layer][: pos + 1, kvh, :]
                scores = (kh @ qh[h]) / math.sqrt(head_dim)
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                att[h] = probs @ vh

            x = x + (weights[f"{p}.wo"] @ att.reshape(-1))
            xb = _rmsnorm_np(x, weights[f"{p}.ffn_norm"], cfg.rms_norm_eps)
            gate = weights[f"{p}.w_gate"] @ xb
            up = weights[f"{p}.w_up"] @ xb
            x = x + (weights[f"{p}.w_down"] @ ((gate / (1.0 + np.exp(-gate))) * up))

        xf = _rmsnorm_np(x, weights["final_norm"], cfg.rms_norm_eps)
        lm_w = weights["tok_embeddings"] if cfg.tie_embeddings else weights["lm_head"]
        return (lm_w @ xf).astype(np.float32)

    t0 = time.perf_counter()
    prompt_logits = np.zeros(cfg.vocab_size, dtype=np.float32)
    for pos, tok in enumerate(prompt):
        prompt_logits = step(tok, pos)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    next_tok = int(np.argmax(prompt_logits))
    for s in range(1, max_new_tokens):
        logits = step(next_tok, len(prompt) + s - 1)
        next_tok = int(np.argmax(logits))
    decode_ms = (time.perf_counter() - t1) * 1000.0

    weight_mb = sum(arr.nbytes for arr in weights.values()) / (1024.0 * 1024.0)
    row = {
        "backend": "NumPy FP32 (CPU)",
        "weight_mb": round(weight_mb, 2),
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tok_s": round(max_new_tokens * 1000.0 / max(decode_ms, 1e-6), 2),
        "cosine_sim": 1.0,
        "max_abs_drift": 0.0,
    }
    return row, prompt_logits


def _run_pytorch_baseline(
    cfg: tinyinfer.ModelConfig,
    weights: dict[str, np.ndarray],
    prompt: list[int],
    max_new_tokens: int,
    ref_logits: np.ndarray,
) -> dict[str, Any] | None:
    if importlib.util.find_spec("torch") is None:
        return None
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as f_nn  # type: ignore[import-not-found]

    dim = cfg.dim
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    head_dim = dim // n_q
    group = n_q // n_kv
    max_ctx = len(prompt) + max_new_tokens
    tw = {k: torch.from_numpy(v) for k, v in weights.items()}
    k_cache = [
        torch.zeros((max_ctx, n_kv, head_dim), dtype=torch.float32)
        for _ in range(cfg.n_layers)
    ]
    v_cache = [
        torch.zeros((max_ctx, n_kv, head_dim), dtype=torch.float32)
        for _ in range(cfg.n_layers)
    ]

    half = head_dim // 2
    idx = torch.arange(half, dtype=torch.float64)
    freqs = 1.0 / (cfg.rope_theta ** ((2.0 * idx) / head_dim))

    def rope_t(vec: Any, n_heads: int, pos: int) -> Any:
        out = vec.view(n_heads, head_dim).clone()
        ang = pos * freqs
        c = torch.cos(ang).to(torch.float32)
        s = torch.sin(ang).to(torch.float32)
        v0 = out[:, 0::2].clone()
        v1 = out[:, 1::2].clone()
        out[:, 0::2] = v0 * c - v1 * s
        out[:, 1::2] = v0 * s + v1 * c
        return out

    with torch.no_grad():
        def step(tok: int, pos: int) -> Any:
            x = tw["tok_embeddings"][tok].clone()
            for layer in range(cfg.n_layers):
                p = f"layers.{layer}"
                rms = torch.sqrt(torch.mean(x * x) + cfg.rms_norm_eps)
                xb = (x / rms) * tw[f"{p}.attn_norm"]
                q = rope_t(tw[f"{p}.wq"] @ xb, n_q, pos)
                k = rope_t(tw[f"{p}.wk"] @ xb, n_kv, pos)
                v = (tw[f"{p}.wv"] @ xb).view(n_kv, head_dim)
                k_cache[layer][pos] = k
                v_cache[layer][pos] = v

                kh = k_cache[layer][: pos + 1].repeat_interleave(group, dim=1).transpose(0, 1)
                vh = v_cache[layer][: pos + 1].repeat_interleave(group, dim=1).transpose(0, 1)
                att = f_nn.scaled_dot_product_attention(q.unsqueeze(1), kh, vh).squeeze(1)
                x = x + (tw[f"{p}.wo"] @ att.reshape(-1))

                rms2 = torch.sqrt(torch.mean(x * x) + cfg.rms_norm_eps)
                xb2 = (x / rms2) * tw[f"{p}.ffn_norm"]
                ffn = f_nn.silu(tw[f"{p}.w_gate"] @ xb2) * (tw[f"{p}.w_up"] @ xb2)
                x = x + (tw[f"{p}.w_down"] @ ffn)

            rmsf = torch.sqrt(torch.mean(x * x) + cfg.rms_norm_eps)
            xf = (x / rmsf) * tw["final_norm"]
            lm_w = tw["tok_embeddings"] if cfg.tie_embeddings else tw["lm_head"]
            return lm_w @ xf

        t0 = time.perf_counter()
        p_logits = None
        for pos, tok in enumerate(prompt):
            p_logits = step(tok, pos)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        next_tok = int(torch.argmax(p_logits).item())
        for s in range(1, max_new_tokens):
            logits = step(next_tok, len(prompt) + s - 1)
            next_tok = int(torch.argmax(logits).item())
        decode_ms = (time.perf_counter() - t1) * 1000.0

    p_np = p_logits.numpy()
    weight_mb = sum(arr.nbytes for arr in weights.values()) / (1024.0 * 1024.0)
    return {
        "backend": f"PyTorch {torch.__version__} FP32 (CPU)",
        "weight_mb": round(weight_mb, 2),
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tok_s": round(max_new_tokens * 1000.0 / max(decode_ms, 1e-6), 2),
        "cosine_sim": round(_cosine_sim(p_np, ref_logits), 6),
        "max_abs_drift": float(f"{float(np.max(np.abs(p_np - ref_logits))):.2e}"),
    }


def run_benchmark(
    *,
    n_layers: int = 6,
    dim: int = 512,
    hidden_dim: int = 1536,
    n_q_heads: int = 8,
    n_kv_heads: int = 2,
    vocab_size: int = 2048,
    prompt_len: int = 8,
    max_new_tokens: int = 32,
    out_json: Path | None = None,
) -> dict[str, Any]:
    cfg = tinyinfer.ModelConfig()
    cfg.vocab_size = vocab_size
    cfg.dim = dim
    cfg.hidden_dim = hidden_dim
    cfg.n_layers = n_layers
    cfg.n_q_heads = n_q_heads
    cfg.n_kv_heads = n_kv_heads
    cfg.max_seq_len = prompt_len + max_new_tokens + 16
    cfg.tie_embeddings = True

    weights = _make_weights(cfg, seed=2026)
    prompt = [int((i * 17 + 1) % vocab_size) for i in range(prompt_len)]

    rows: list[dict[str, Any]] = []
    np_row, ref_logits = _run_numpy_baseline(cfg, weights, prompt, max_new_tokens)
    rows.append(np_row)

    pt_row = _run_pytorch_baseline(cfg, weights, prompt, max_new_tokens, ref_logits)
    if pt_row is not None:
        rows.append(pt_row)

    # Cooldown OpenBLAS/PyTorch threads before measuring OpenMP C++ runtime.
    time.sleep(0.25)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        exported: list[tuple[str, Path]] = []
        for label, dtype in (
            ("tinyinfer FP32 (mmap)", "fp32"),
            ("tinyinfer Q8_0 (mmap)", "q8_0"),
        ):
            tinf_path = tinyinfer.export_tinf(
                tmp / f"model_{dtype}.tinf",
                config=cfg,
                weights=weights,
                dtype=dtype,  # type: ignore[arg-type]
            )
            exported.append((label, tinf_path))

        # Let OpenBLAS threads exit spin-wait once after all NumPy/export work is done.
        time.sleep(0.30)

        measured: list[tuple[str, float, Any, np.ndarray]] = []
        for label, tinf_path in exported:
            model = tinyinfer.TransformerModel(str(tinf_path))
            # Warmup OpenMP pool and mmap pages before timed generation.
            _ = model.generate(prompt, max_new_tokens=4, temperature=0.0)
            gen = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

            model.reset_cache()
            logits = np.zeros(vocab_size, dtype=np.float32)
            for pos, tok in enumerate(prompt):
                logits = model.forward_token(tok, pos)
            measured.append((label, model.mapped_bytes() / (1024.0 * 1024.0), gen, logits))

        for label, weight_mb, gen, logits in measured:
            rows.append(
                {
                    "backend": label,
                    "weight_mb": round(weight_mb, 2),
                    "prefill_ms": round(gen.prefill_ms, 2),
                    "decode_ms": round(gen.decode_ms, 2),
                    "decode_tok_s": round(gen.decode_tok_per_sec, 2),
                    "cosine_sim": round(_cosine_sim(logits, ref_logits), 6),
                    "max_abs_drift": float(f"{float(np.max(np.abs(logits - ref_logits))):.2e}"),
                }
            )

    info = tinyinfer.get_build_info()
    report = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "cpu": _detect_cpu_model(),
            "threads": os.cpu_count() or 1,
            "os": f"{platform.system()} {platform.release()}",
            "avx2_fma": info.avx2_fma,
            "openmp": info.openmp,
            "cuda_compiled": info.cuda,
            "cuda_device": tinyinfer.cuda_device_name(),
            "peak_rss_mb": round(tinyinfer.process_peak_rss_mb(), 2),
        },
        "model_config": {
            "n_layers": n_layers,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "vocab_size": vocab_size,
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
        },
        "results": rows,
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Transformer inference benchmark (tinyinfer vs NumPy/PyTorch)."
    )
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--n-q-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(
        n_layers=args.layers,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_tokens,
        out_json=args.out_json,
    )

    print(
        "| Backend | Weights (MB) | Prefill (ms) | Decode (ms) "
        "| Decode (tok/s) | Cosine Sim | Max Abs Drift |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in report["results"]:
        print(
            f"| {r['backend']} | {r['weight_mb']:.2f} | {r['prefill_ms']:.2f} "
            f"| {r['decode_ms']:.2f} | {r['decode_tok_s']:.2f} "
            f"| {r['cosine_sim']:.6f} | {r['max_abs_drift']:.2e} |"
        )
    if args.out_json is not None:
        print(f"\n[SAVED] JSON telemetry -> {args.out_json}")


if __name__ == "__main__":
    main()
