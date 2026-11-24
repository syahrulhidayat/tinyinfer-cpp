#pragma once

#include <cstddef>

#include "tinyinfer/kv_cache.hpp"

namespace tinyinfer {

// Single-token Grouped-Query Attention decode step at 0-based position `pos`.
// Appends (k_new, v_new) into cache[layer, pos] and attends over t in [0, pos].
//   q:     [n_q_heads * head_dim]
//   k_new: [n_kv_heads * head_dim]
//   v_new: [n_kv_heads * head_dim]
//   out:   [n_q_heads * head_dim]
void gqa_decode_step(
    const float* q,
    const float* k_new,
    const float* v_new,
    KVCache& cache,
    std::size_t layer,
    std::size_t pos,
    std::size_t n_q_heads,
    float* out
);

// Causal Grouped-Query Attention prefill for `seq_len` tokens starting at `start_pos`.
// Populates cache[layer, start_pos .. start_pos + seq_len - 1].
//   q:   [seq_len, n_q_heads * head_dim]
//   k:   [seq_len, n_kv_heads * head_dim]
//   v:   [seq_len, n_kv_heads * head_dim]
//   out: [seq_len, n_q_heads * head_dim]
void gqa_prefill(
    const float* q,
    const float* k,
    const float* v,
    KVCache& cache,
    std::size_t layer,
    std::size_t start_pos,
    std::size_t seq_len,
    std::size_t n_q_heads,
    float* out
);

}  // namespace tinyinfer
