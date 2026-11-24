#include "tinyinfer/kv_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tinyinfer {

KVCache::KVCache(
    std::size_t n_layers,
    std::size_t max_seq_len,
    std::size_t n_kv_heads,
    std::size_t head_dim
)
    : n_layers_(n_layers),
      max_seq_len_(max_seq_len),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      kv_dim_(n_kv_heads * head_dim),
      layer_stride_(max_seq_len * n_kv_heads * head_dim) {
    if (n_layers == 0 || max_seq_len == 0 || n_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("KVCache dimensions must all be non-zero.");
    }
    const std::size_t total_elems = n_layers_ * layer_stride_;
    k_buf_.assign(total_elems, 0.0f);
    v_buf_.assign(total_elems, 0.0f);
    layer_seq_len_.assign(n_layers_, 0);
}

void KVCache::update(
    std::size_t layer,
    std::size_t pos,
    const float* k_new,
    const float* v_new
) {
    if (layer >= n_layers_) {
        throw std::out_of_range("KVCache::update: layer index out of range.");
    }
    if (pos >= max_seq_len_) {
        throw std::out_of_range("KVCache::update: position exceeds max_seq_len.");
    }
    if (k_new == nullptr || v_new == nullptr) {
        throw std::invalid_argument("KVCache::update: null K/V pointer.");
    }

    const std::size_t offset = layer * layer_stride_ + pos * kv_dim_;
    std::memcpy(k_buf_.data() + offset, k_new, kv_dim_ * sizeof(float));
    std::memcpy(v_buf_.data() + offset, v_new, kv_dim_ * sizeof(float));

    if (pos + 1 > layer_seq_len_[layer]) {
        layer_seq_len_[layer] = pos + 1;
    }
}

void KVCache::reset() noexcept {
    std::fill(layer_seq_len_.begin(), layer_seq_len_.end(), 0);
}

const float* KVCache::key_ptr(
    std::size_t layer,
    std::size_t pos,
    std::size_t kv_head
) const noexcept {
    return k_buf_.data() + layer * layer_stride_ + pos * kv_dim_ + kv_head * head_dim_;
}

const float* KVCache::value_ptr(
    std::size_t layer,
    std::size_t pos,
    std::size_t kv_head
) const noexcept {
    return v_buf_.data() + layer * layer_stride_ + pos * kv_dim_ + kv_head * head_dim_;
}

std::size_t KVCache::seq_len(std::size_t layer) const {
    if (layer >= n_layers_) {
        throw std::out_of_range("KVCache::seq_len: layer index out of range.");
    }
    return layer_seq_len_[layer];
}

std::size_t KVCache::memory_bytes() const noexcept {
    return (k_buf_.size() + v_buf_.size()) * sizeof(float);
}

}  // namespace tinyinfer
