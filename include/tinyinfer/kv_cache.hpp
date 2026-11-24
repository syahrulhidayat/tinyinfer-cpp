#pragma once

#include <cstddef>
#include <vector>

namespace tinyinfer {

// Pre-allocated contiguous KV cache with layout [layer, pos, kv_head, head_dim].
// Single contiguous write per decode step and contiguous head_dim reads during attention.
class KVCache {
public:
    KVCache(
        std::size_t n_layers,
        std::size_t max_seq_len,
        std::size_t n_kv_heads,
        std::size_t head_dim
    );

    // Copies one token's K and V vectors (each length n_kv_heads * head_dim) at (layer, pos).
    void update(std::size_t layer, std::size_t pos, const float* k_new, const float* v_new);

    void reset() noexcept;

    [[nodiscard]] const float* key_ptr(
        std::size_t layer,
        std::size_t pos,
        std::size_t kv_head
    ) const noexcept;

    [[nodiscard]] const float* value_ptr(
        std::size_t layer,
        std::size_t pos,
        std::size_t kv_head
    ) const noexcept;

    [[nodiscard]] std::size_t n_layers() const noexcept { return n_layers_; }
    [[nodiscard]] std::size_t max_seq_len() const noexcept { return max_seq_len_; }
    [[nodiscard]] std::size_t n_kv_heads() const noexcept { return n_kv_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] std::size_t kv_dim() const noexcept { return kv_dim_; }
    [[nodiscard]] std::size_t seq_len(std::size_t layer) const;
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    std::size_t n_layers_;
    std::size_t max_seq_len_;
    std::size_t n_kv_heads_;
    std::size_t head_dim_;
    std::size_t kv_dim_;
    std::size_t layer_stride_;

    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    std::vector<std::size_t> layer_seq_len_;
};

}  // namespace tinyinfer
