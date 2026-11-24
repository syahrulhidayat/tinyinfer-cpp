from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import tinyinfer


def _rmsnorm_np(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return ((x / rms) * w).astype(np.float32)


def _rope_np(vec: np.ndarray, n_heads: int, head_dim: int, pos: int, theta: float) -> np.ndarray:
    out = vec.copy().reshape(n_heads, head_dim)
    half = head_dim // 2
    for i in range(half):
        freq = 1.0 / (theta ** ((2.0 * i) / head_dim))
        angle = pos * freq
        c, s = math.cos(angle), math.sin(angle)
        v0 = out[:, 2 * i].copy()
        v1 = out[:, 2 * i + 1].copy()
        out[:, 2 * i] = v0 * c - v1 * s
        out[:, 2 * i + 1] = v0 * s + v1 * c
    return out.reshape(-1).astype(np.float32)


def _numpy_transformer_forward(
    prompt: list[int],
    cfg: tinyinfer.ModelConfig,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    dim = cfg.dim
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    head_dim = dim // n_q
    group = n_q // n_kv

    k_cache = [
        np.zeros((len(prompt), n_kv, head_dim), dtype=np.float32)
        for _ in range(cfg.n_layers)
    ]
    v_cache = [
        np.zeros((len(prompt), n_kv, head_dim), dtype=np.float32)
        for _ in range(cfg.n_layers)
    ]
    logits = np.zeros(cfg.vocab_size, dtype=np.float32)

    for pos, tok in enumerate(prompt):
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
            silu_gate = gate / (1.0 + np.exp(-gate))
            x = x + (weights[f"{p}.w_down"] @ (silu_gate * up))

        xf = _rmsnorm_np(x, weights["final_norm"], cfg.rms_norm_eps)
        lm_w = weights["tok_embeddings"] if cfg.tie_embeddings else weights["lm_head"]
        logits = (lm_w @ xf).astype(np.float32)

    return logits


def _make_random_weights(cfg: tinyinfer.ModelConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = cfg.dim
    hidden = cfg.hidden_dim
    kv_dim = cfg.n_kv_heads * (dim // cfg.n_q_heads)

    w: dict[str, np.ndarray] = {
        "tok_embeddings": rng.normal(0.0, 0.05, (cfg.vocab_size, dim)).astype(np.float32),
        "final_norm": rng.uniform(0.85, 1.15, (dim,)).astype(np.float32),
    }
    for layer in range(cfg.n_layers):
        p = f"layers.{layer}"
        w[f"{p}.attn_norm"] = rng.uniform(0.85, 1.15, (dim,)).astype(np.float32)
        w[f"{p}.ffn_norm"] = rng.uniform(0.85, 1.15, (dim,)).astype(np.float32)
        w[f"{p}.wq"] = rng.normal(0.0, 0.05, (dim, dim)).astype(np.float32)
        w[f"{p}.wk"] = rng.normal(0.0, 0.05, (kv_dim, dim)).astype(np.float32)
        w[f"{p}.wv"] = rng.normal(0.0, 0.05, (kv_dim, dim)).astype(np.float32)
        w[f"{p}.wo"] = rng.normal(0.0, 0.05, (dim, dim)).astype(np.float32)
        w[f"{p}.w_gate"] = rng.normal(0.0, 0.05, (hidden, dim)).astype(np.float32)
        w[f"{p}.w_up"] = rng.normal(0.0, 0.05, (hidden, dim)).astype(np.float32)
        w[f"{p}.w_down"] = rng.normal(0.0, 0.05, (dim, hidden)).astype(np.float32)
    if not cfg.tie_embeddings:
        w["lm_head"] = rng.normal(0.0, 0.05, (cfg.vocab_size, dim)).astype(np.float32)
    return w


def test_exported_mmap_transformer_matches_numpy_reference(tmp_path: Path) -> None:
    cfg = tinyinfer.ModelConfig()
    cfg.vocab_size = 96
    cfg.dim = 64
    cfg.hidden_dim = 160
    cfg.n_layers = 2
    cfg.n_q_heads = 4
    cfg.n_kv_heads = 2
    cfg.max_seq_len = 32
    cfg.tie_embeddings = False

    weights = _make_random_weights(cfg, seed=2026)
    fp32_file = tinyinfer.export_tinf(
        tmp_path / "m_fp32.tinf", config=cfg, weights=weights, dtype="fp32"
    )
    q8_file = tinyinfer.export_tinf(
        tmp_path / "m_q8.tinf", config=cfg, weights=weights, dtype="q8_0"
    )

    assert q8_file.stat().st_size < fp32_file.stat().st_size * 0.4

    prompt = [3, 19, 44, 7, 12]
    ref_logits = _numpy_transformer_forward(prompt, cfg, weights)

    m_fp32 = tinyinfer.TransformerModel(str(fp32_file))
    for pos, tok in enumerate(prompt):
        out_fp32 = m_fp32.forward_token(tok, pos)
    assert np.allclose(out_fp32, ref_logits, atol=1e-4, rtol=1e-4)

    m_q8 = tinyinfer.TransformerModel(str(q8_file))
    for pos, tok in enumerate(prompt):
        out_q8 = m_q8.forward_token(tok, pos)
    cos_sim = float(
        np.dot(out_q8.astype(np.float64), ref_logits.astype(np.float64))
        / (np.linalg.norm(out_q8) * np.linalg.norm(ref_logits))
    )
    assert cos_sim > 0.995

    res1 = m_q8.generate(prompt, max_new_tokens=6, temperature=0.0)
    res2 = m_q8.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert len(res1.generated_tokens) == 6
    assert res1.generated_tokens == res2.generated_tokens
    assert res1.decode_tok_per_sec > 0.0
