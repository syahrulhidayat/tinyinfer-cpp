#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "tinyinfer/kv_cache.hpp"
#include "tinyinfer/mmap_loader.hpp"

namespace tinyinfer {

enum class WeightDType : std::uint32_t {
    FP32 = 0,
    Q8_0 = 1,
};

struct TinfHeader {
    char magic[4];
    std::uint32_t version;
    std::uint32_t dtype;
    std::uint32_t vocab_size;
    std::uint32_t dim;
    std::uint32_t hidden_dim;
    std::uint32_t n_layers;
    std::uint32_t n_q_heads;
    std::uint32_t n_kv_heads;
    std::uint32_t max_seq_len;
    std::uint32_t tie_embeddings;
    float rms_norm_eps;
    float rope_theta;
    std::uint32_t reserved[3];
};

static_assert(sizeof(TinfHeader) == 64, "TinfHeader must be exactly 64 bytes.");

struct ModelConfig {
    std::size_t vocab_size = 0;
    std::size_t dim = 0;
    std::size_t hidden_dim = 0;
    std::size_t n_layers = 0;
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t max_seq_len = 0;
    bool tie_embeddings = false;
    WeightDType dtype = WeightDType::FP32;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;

    [[nodiscard]] std::size_t head_dim() const noexcept { return dim / n_q_heads; }
    [[nodiscard]] std::size_t kv_dim() const noexcept { return n_kv_heads * head_dim(); }
};

struct LinearWeightView {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t blocks_per_row = 0;
    WeightDType dtype = WeightDType::FP32;
    const float* fp32_data = nullptr;
    const float* q8_scales = nullptr;
    const int8_t* q8_weights = nullptr;
};

struct LayerWeightsView {
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    LinearWeightView wq;
    LinearWeightView wk;
    LinearWeightView wv;
    LinearWeightView wo;
    LinearWeightView w_gate;
    LinearWeightView w_up;
    LinearWeightView w_down;
};

struct GenerationResult {
    std::vector<std::uint32_t> generated_tokens;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double decode_tok_per_sec = 0.0;
    double peak_rss_mb = 0.0;
};

// Samples one token ID from unnormalized logits.
// temperature <= 0 uses deterministic greedy argmax.
[[nodiscard]] std::uint32_t sample_token(
    const float* logits,
    std::size_t vocab_size,
    float temperature,
    float top_p,
    std::mt19937& rng
);

// Returns current process peak resident set size in MiB (POSIX getrusage).
[[nodiscard]] double process_peak_rss_mb() noexcept;

// End-to-end zero-copy mmap Decoder-Only Transformer (Llama/Qwen architecture).
class TransformerModel {
public:
    explicit TransformerModel(const std::string& tinf_path);

    // Runs one forward step at `pos` and returns reference to logits [vocab_size].
    [[nodiscard]] const std::vector<float>& forward_token(std::uint32_t token_id, std::size_t pos);

    // Resets KV-Cache state to start a fresh sequence.
    void reset_cache() noexcept;

    // Runs prompt prefill followed by autoregressive token generation.
    [[nodiscard]] GenerationResult generate(
        const std::vector<std::uint32_t>& prompt_tokens,
        std::size_t max_new_tokens,
        float temperature = 0.0f,
        float top_p = 0.9f,
        std::uint32_t seed = 42
    );

    [[nodiscard]] const ModelConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return mapped_.size(); }
    [[nodiscard]] std::size_t kv_cache_bytes() const noexcept { return cache_.memory_bytes(); }

private:
    void parse_and_bind_weights();
    void linear_matvec(const float* x, const LinearWeightView& w, float* out) const;

    MappedFile mapped_;
    ModelConfig cfg_;
    const float* tok_embeddings_ = nullptr;
    std::vector<LayerWeightsView> layers_;
    const float* final_norm_ = nullptr;
    LinearWeightView lm_head_;

    KVCache cache_;

    // Pre-allocated scratch buffers for zero-allocation decode steps.
    std::vector<float> x_;
    std::vector<float> xb_;
    std::vector<float> xb2_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> att_out_;
    std::vector<float> hb_;
    std::vector<float> hb2_;
    std::vector<float> logits_;
};

// Utility to serialize a deterministic synthetic .tinf model for C++ tests and CLI benchmarking.
void write_synthetic_tinf(const std::string& path, const ModelConfig& cfg, std::uint32_t seed);

}  // namespace tinyinfer
