#include "tinyinfer/transformer.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "tinyinfer/attention.hpp"
#include "tinyinfer/matmul.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/quant.hpp"
#include "tinyinfer/types.hpp"

namespace tinyinfer {

namespace {

class ByteCursor {
public:
    ByteCursor(const std::byte* base, std::size_t total_size)
        : base_(base), total_size_(total_size), offset_(0) {}

    template <typename T>
    const T* take_array(std::size_t count) {
        const std::size_t bytes = count * sizeof(T);
        if (offset_ + bytes > total_size_) {
            throw std::runtime_error("TINF weight file is truncated.");
        }
        const auto* ptr = reinterpret_cast<const T*>(base_ + offset_);
        offset_ += bytes;
        return ptr;
    }

    LinearWeightView take_linear(std::size_t rows, std::size_t cols, WeightDType dtype) {
        LinearWeightView view;
        view.rows = rows;
        view.cols = cols;
        view.dtype = dtype;
        if (dtype == WeightDType::FP32) {
            view.fp32_data = take_array<float>(rows * cols);
        } else {
            view.blocks_per_row = (cols + kQ8BlockSize - 1) / kQ8BlockSize;
            view.q8_scales = take_array<float>(rows * view.blocks_per_row);
            view.q8_weights = take_array<int8_t>(rows * view.blocks_per_row * kQ8BlockSize);
        }
        return view;
    }

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
    const std::byte* base_;
    std::size_t total_size_;
    std::size_t offset_;
};

void validate_config(const ModelConfig& cfg) {
    if (cfg.vocab_size == 0 || cfg.dim == 0 || cfg.hidden_dim == 0 || cfg.n_layers == 0 ||
        cfg.n_q_heads == 0 || cfg.n_kv_heads == 0 || cfg.max_seq_len == 0) {
        throw std::invalid_argument("ModelConfig: all model dimensions must be non-zero.");
    }
    if (cfg.dim % cfg.n_q_heads != 0) {
        throw std::invalid_argument("ModelConfig: dim must be divisible by n_q_heads.");
    }
    if (cfg.n_q_heads % cfg.n_kv_heads != 0) {
        throw std::invalid_argument("ModelConfig: n_q_heads must be divisible by n_kv_heads.");
    }
    if ((cfg.dim / cfg.n_q_heads) % 2 != 0) {
        throw std::invalid_argument("ModelConfig: head_dim must be even for RoPE.");
    }
}

ModelConfig config_from_header(const TinfHeader& hdr) {
    if (std::memcmp(hdr.magic, "TINF", 4) != 0) {
        throw std::runtime_error("Invalid TINF file: bad magic header.");
    }
    if (hdr.version != 1) {
        throw std::runtime_error("Unsupported TINF format version.");
    }
    if (hdr.dtype != 0 && hdr.dtype != 1) {
        throw std::runtime_error("Unsupported TINF weight dtype.");
    }

    ModelConfig cfg;
    cfg.vocab_size = hdr.vocab_size;
    cfg.dim = hdr.dim;
    cfg.hidden_dim = hdr.hidden_dim;
    cfg.n_layers = hdr.n_layers;
    cfg.n_q_heads = hdr.n_q_heads;
    cfg.n_kv_heads = hdr.n_kv_heads;
    cfg.max_seq_len = hdr.max_seq_len;
    cfg.tie_embeddings = (hdr.tie_embeddings != 0);
    cfg.dtype = (hdr.dtype == 0) ? WeightDType::FP32 : WeightDType::Q8_0;
    cfg.rms_norm_eps = hdr.rms_norm_eps;
    cfg.rope_theta = hdr.rope_theta;
    validate_config(cfg);
    return cfg;
}

ModelConfig read_header_config(const MappedFile& mapped) {
    if (mapped.size() < sizeof(TinfHeader)) {
        throw std::runtime_error("Invalid TINF file: smaller than 64-byte header.");
    }
    TinfHeader hdr {};
    std::memcpy(&hdr, mapped.data(), sizeof(TinfHeader));
    return config_from_header(hdr);
}

void write_linear_raw(
    std::ofstream& ofs,
    const std::vector<float>& mat,
    std::size_t rows,
    std::size_t cols,
    WeightDType dtype
) {
    if (dtype == WeightDType::FP32) {
        ofs.write(
            reinterpret_cast<const char*>(mat.data()),
            static_cast<std::streamsize>(mat.size() * sizeof(float))
        );
    } else {
        const auto q = quantize_q8_0(mat.data(), rows, cols);
        ofs.write(
            reinterpret_cast<const char*>(q.scales.data()),
            static_cast<std::streamsize>(q.scales.size() * sizeof(float))
        );
        ofs.write(
            reinterpret_cast<const char*>(q.weights.data()),
            static_cast<std::streamsize>(q.weights.size() * sizeof(int8_t))
        );
    }
}

}  // namespace

double process_peak_rss_mb() noexcept {
    struct rusage ru {};
    if (::getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0.0;
    }
#ifdef __APPLE__
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

std::uint32_t sample_token(
    const float* logits,
    std::size_t vocab_size,
    float temperature,
    float top_p,
    std::mt19937& rng
) {
    if (logits == nullptr || vocab_size == 0) {
        throw std::invalid_argument("sample_token: empty logits.");
    }

    if (temperature <= 0.0f) {
        std::size_t best_idx = 0;
        float best_val = logits[0];
        for (std::size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return static_cast<std::uint32_t>(best_idx);
    }

    std::vector<float> scaled(vocab_size);
    const float inv_temp = 1.0f / temperature;
    for (std::size_t i = 0; i < vocab_size; ++i) {
        scaled[i] = logits[i] * inv_temp;
    }

    std::vector<float> probs(vocab_size);
    softmax(probs.data(), scaled.data(), vocab_size);

    std::vector<std::pair<float, std::uint32_t>> ranked;
    ranked.reserve(vocab_size);
    for (std::size_t i = 0; i < vocab_size; ++i) {
        ranked.emplace_back(probs[i], static_cast<std::uint32_t>(i));
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    const float clamped_top_p = std::clamp(top_p, 1e-4f, 1.0f);
    double cum_prob = 0.0;
    std::size_t cutoff = 0;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        cum_prob += static_cast<double>(ranked[i].first);
        cutoff = i + 1;
        if (cum_prob >= static_cast<double>(clamped_top_p)) {
            break;
        }
    }

    std::uniform_real_distribution<double> uni(0.0, cum_prob);
    const double target = uni(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < cutoff; ++i) {
        acc += static_cast<double>(ranked[i].first);
        if (acc >= target) {
            return ranked[i].second;
        }
    }
    return ranked[cutoff - 1].second;
}

TransformerModel::TransformerModel(const std::string& tinf_path)
    : mapped_(tinf_path),
      cfg_(read_header_config(mapped_)),
      cache_(cfg_.n_layers, cfg_.max_seq_len, cfg_.n_kv_heads, cfg_.head_dim()),
      x_(cfg_.dim, 0.0f),
      xb_(cfg_.dim, 0.0f),
      xb2_(cfg_.dim, 0.0f),
      q_(cfg_.dim, 0.0f),
      k_(cfg_.kv_dim(), 0.0f),
      v_(cfg_.kv_dim(), 0.0f),
      att_out_(cfg_.dim, 0.0f),
      hb_(cfg_.hidden_dim, 0.0f),
      hb2_(cfg_.hidden_dim, 0.0f),
      logits_(cfg_.vocab_size, 0.0f) {
    parse_and_bind_weights();
}

void TransformerModel::parse_and_bind_weights() {
    ByteCursor cursor(mapped_.data(), mapped_.size());
    (void)cursor.take_array<std::byte>(sizeof(TinfHeader));

    tok_embeddings_ = cursor.take_array<float>(cfg_.vocab_size * cfg_.dim);

    layers_.resize(cfg_.n_layers);
    const std::size_t kv_dim = cfg_.kv_dim();
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        auto& lw = layers_[l];
        lw.attn_norm = cursor.take_array<float>(cfg_.dim);
        lw.ffn_norm = cursor.take_array<float>(cfg_.dim);
        lw.wq = cursor.take_linear(cfg_.dim, cfg_.dim, cfg_.dtype);
        lw.wk = cursor.take_linear(kv_dim, cfg_.dim, cfg_.dtype);
        lw.wv = cursor.take_linear(kv_dim, cfg_.dim, cfg_.dtype);
        lw.wo = cursor.take_linear(cfg_.dim, cfg_.dim, cfg_.dtype);
        lw.w_gate = cursor.take_linear(cfg_.hidden_dim, cfg_.dim, cfg_.dtype);
        lw.w_up = cursor.take_linear(cfg_.hidden_dim, cfg_.dim, cfg_.dtype);
        lw.w_down = cursor.take_linear(cfg_.dim, cfg_.hidden_dim, cfg_.dtype);
    }

    final_norm_ = cursor.take_array<float>(cfg_.dim);
    if (cfg_.tie_embeddings) {
        lm_head_.rows = cfg_.vocab_size;
        lm_head_.cols = cfg_.dim;
        lm_head_.dtype = WeightDType::FP32;
        lm_head_.fp32_data = tok_embeddings_;
    } else {
        lm_head_ = cursor.take_linear(cfg_.vocab_size, cfg_.dim, cfg_.dtype);
    }

    if (cursor.offset() != mapped_.size()) {
        throw std::runtime_error("TINF file size mismatch with header specification.");
    }
}

void TransformerModel::linear_matvec(
    const float* x,
    const LinearWeightView& w,
    float* out
) const {
    if (w.dtype == WeightDType::FP32) {
        matmul_fp32_nt(x, w.fp32_data, out, 1, w.rows, w.cols);
    } else {
        matmul_q8_0_nt_raw(x, w.q8_weights, w.q8_scales, out, 1, w.rows, w.cols, w.blocks_per_row);
    }
}

void TransformerModel::reset_cache() noexcept {
    cache_.reset();
}

const std::vector<float>& TransformerModel::forward_token(
    std::uint32_t token_id,
    std::size_t pos
) {
    if (token_id >= cfg_.vocab_size) {
        throw std::out_of_range("forward_token: token_id out of vocab_size range.");
    }
    if (pos >= cfg_.max_seq_len) {
        throw std::out_of_range("forward_token: pos exceeds max_seq_len.");
    }

    const std::size_t dim = cfg_.dim;
    const std::size_t hidden_dim = cfg_.hidden_dim;
    const std::size_t head_dim = cfg_.head_dim();

    std::memcpy(x_.data(), tok_embeddings_ + static_cast<std::size_t>(token_id) * dim,
                dim * sizeof(float));

    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        const auto& lw = layers_[l];

        // Attention block
        rmsnorm(xb_.data(), x_.data(), lw.attn_norm, dim, cfg_.rms_norm_eps);
        linear_matvec(xb_.data(), lw.wq, q_.data());
        linear_matvec(xb_.data(), lw.wk, k_.data());
        linear_matvec(xb_.data(), lw.wv, v_.data());

        rope_inplace(q_.data(), cfg_.n_q_heads, head_dim, pos, cfg_.rope_theta);
        rope_inplace(k_.data(), cfg_.n_kv_heads, head_dim, pos, cfg_.rope_theta);

        gqa_decode_step(
            q_.data(), k_.data(), v_.data(), cache_, l, pos, cfg_.n_q_heads, att_out_.data()
        );
        linear_matvec(att_out_.data(), lw.wo, xb2_.data());
        for (std::size_t i = 0; i < dim; ++i) {
            x_[i] += xb2_[i];
        }

        // SwiGLU FFN block
        rmsnorm(xb_.data(), x_.data(), lw.ffn_norm, dim, cfg_.rms_norm_eps);
        linear_matvec(xb_.data(), lw.w_gate, hb_.data());
        linear_matvec(xb_.data(), lw.w_up, hb2_.data());
        silu(hb_.data(), hb_.data(), hidden_dim);
        for (std::size_t i = 0; i < hidden_dim; ++i) {
            hb_[i] *= hb2_[i];
        }
        linear_matvec(hb_.data(), lw.w_down, xb2_.data());
        for (std::size_t i = 0; i < dim; ++i) {
            x_[i] += xb2_[i];
        }
    }

    rmsnorm(xb_.data(), x_.data(), final_norm_, dim, cfg_.rms_norm_eps);
    linear_matvec(xb_.data(), lm_head_, logits_.data());
    return logits_;
}

GenerationResult TransformerModel::generate(
    const std::vector<std::uint32_t>& prompt_tokens,
    std::size_t max_new_tokens,
    float temperature,
    float top_p,
    std::uint32_t seed
) {
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("generate: prompt_tokens cannot be empty.");
    }
    if (prompt_tokens.size() + max_new_tokens > cfg_.max_seq_len) {
        throw std::out_of_range("generate: total sequence length exceeds max_seq_len.");
    }

    reset_cache();
    std::mt19937 rng(seed);
    GenerationResult res;
    res.generated_tokens.reserve(max_new_tokens);

    const auto t_prefill_start = std::chrono::steady_clock::now();
    const std::vector<float>* last_logits = nullptr;
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
        last_logits = &forward_token(prompt_tokens[i], i);
    }
    const auto t_prefill_end = std::chrono::steady_clock::now();
    res.prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    if (max_new_tokens == 0) {
        res.peak_rss_mb = process_peak_rss_mb();
        return res;
    }

    const auto t_decode_start = std::chrono::steady_clock::now();
    std::uint32_t next_tok =
        sample_token(last_logits->data(), cfg_.vocab_size, temperature, top_p, rng);
    res.generated_tokens.push_back(next_tok);

    for (std::size_t step = 1; step < max_new_tokens; ++step) {
        const std::size_t pos = prompt_tokens.size() + step - 1;
        const auto& logits = forward_token(next_tok, pos);
        next_tok = sample_token(logits.data(), cfg_.vocab_size, temperature, top_p, rng);
        res.generated_tokens.push_back(next_tok);
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    res.decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    res.decode_tok_per_sec =
        (res.decode_ms > 0.0) ? (static_cast<double>(max_new_tokens) * 1000.0 / res.decode_ms) : 0.0;
    res.peak_rss_mb = process_peak_rss_mb();
    return res;
}

void write_synthetic_tinf(const std::string& path, const ModelConfig& cfg, std::uint32_t seed) {
    validate_config(cfg);
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("write_synthetic_tinf: cannot open output file: " + path);
    }

    TinfHeader hdr {};
    std::memcpy(hdr.magic, "TINF", 4);
    hdr.version = 1;
    hdr.dtype = static_cast<std::uint32_t>(cfg.dtype);
    hdr.vocab_size = static_cast<std::uint32_t>(cfg.vocab_size);
    hdr.dim = static_cast<std::uint32_t>(cfg.dim);
    hdr.hidden_dim = static_cast<std::uint32_t>(cfg.hidden_dim);
    hdr.n_layers = static_cast<std::uint32_t>(cfg.n_layers);
    hdr.n_q_heads = static_cast<std::uint32_t>(cfg.n_q_heads);
    hdr.n_kv_heads = static_cast<std::uint32_t>(cfg.n_kv_heads);
    hdr.max_seq_len = static_cast<std::uint32_t>(cfg.max_seq_len);
    hdr.tie_embeddings = cfg.tie_embeddings ? 1U : 0U;
    hdr.rms_norm_eps = cfg.rms_norm_eps;
    hdr.rope_theta = cfg.rope_theta;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(TinfHeader));

    std::mt19937 rng(seed);
    std::normal_distribution<float> weight_dist(0.0f, 0.04f);
    std::uniform_real_distribution<float> norm_dist(0.85f, 1.15f);

    auto rand_vec = [&](std::size_t n, bool is_norm) {
        std::vector<float> v(n);
        for (float& x : v) {
            x = is_norm ? norm_dist(rng) : weight_dist(rng);
        }
        return v;
    };

    const auto emb = rand_vec(cfg.vocab_size * cfg.dim, false);
    ofs.write(
        reinterpret_cast<const char*>(emb.data()),
        static_cast<std::streamsize>(emb.size() * sizeof(float))
    );

    const std::size_t kv_dim = cfg.kv_dim();
    for (std::size_t l = 0; l < cfg.n_layers; ++l) {
        const auto an = rand_vec(cfg.dim, true);
        const auto fn = rand_vec(cfg.dim, true);
        ofs.write(reinterpret_cast<const char*>(an.data()),
                  static_cast<std::streamsize>(an.size() * sizeof(float)));
        ofs.write(reinterpret_cast<const char*>(fn.data()),
                  static_cast<std::streamsize>(fn.size() * sizeof(float)));

        write_linear_raw(ofs, rand_vec(cfg.dim * cfg.dim, false), cfg.dim, cfg.dim, cfg.dtype);
        write_linear_raw(ofs, rand_vec(kv_dim * cfg.dim, false), kv_dim, cfg.dim, cfg.dtype);
        write_linear_raw(ofs, rand_vec(kv_dim * cfg.dim, false), kv_dim, cfg.dim, cfg.dtype);
        write_linear_raw(ofs, rand_vec(cfg.dim * cfg.dim, false), cfg.dim, cfg.dim, cfg.dtype);
        write_linear_raw(ofs, rand_vec(cfg.hidden_dim * cfg.dim, false), cfg.hidden_dim, cfg.dim,
                         cfg.dtype);
        write_linear_raw(ofs, rand_vec(cfg.hidden_dim * cfg.dim, false), cfg.hidden_dim, cfg.dim,
                         cfg.dtype);
        write_linear_raw(ofs, rand_vec(cfg.dim * cfg.hidden_dim, false), cfg.dim, cfg.hidden_dim,
                         cfg.dtype);
    }

    const auto fnorm = rand_vec(cfg.dim, true);
    ofs.write(
        reinterpret_cast<const char*>(fnorm.data()),
        static_cast<std::streamsize>(fnorm.size() * sizeof(float))
    );

    if (!cfg.tie_embeddings) {
        write_linear_raw(
            ofs, rand_vec(cfg.vocab_size * cfg.dim, false), cfg.vocab_size, cfg.dim, cfg.dtype
        );
    }
    ofs.flush();
    if (!ofs.good()) {
        throw std::runtime_error("write_synthetic_tinf: write error on " + path);
    }
}

}  // namespace tinyinfer
