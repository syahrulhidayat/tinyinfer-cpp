#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

#include "tinyinfer/transformer.hpp"

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

void test_mmap_fp32_vs_q8_end_to_end() {
    const auto tmp = std::filesystem::temp_directory_path();
    const auto fp32_path = (tmp / "tinyinfer_test_fp32.tinf").string();
    const auto q8_path = (tmp / "tinyinfer_test_q8.tinf").string();

    tinyinfer::ModelConfig cfg;
    cfg.vocab_size = 128;
    cfg.dim = 64;
    cfg.hidden_dim = 192;
    cfg.n_layers = 2;
    cfg.n_q_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.max_seq_len = 32;
    cfg.tie_embeddings = false;
    cfg.dtype = tinyinfer::WeightDType::FP32;

    tinyinfer::write_synthetic_tinf(fp32_path, cfg, 2026);
    cfg.dtype = tinyinfer::WeightDType::Q8_0;
    tinyinfer::write_synthetic_tinf(q8_path, cfg, 2026);

    tinyinfer::TransformerModel m_fp32(fp32_path);
    tinyinfer::TransformerModel m_q8(q8_path);

    require(m_q8.mapped_bytes() < m_fp32.mapped_bytes() / 2,
            "Q8_0 .tinf file should be significantly smaller than FP32");

    const std::vector<std::uint32_t> prompt = {1, 5, 17, 9};
    std::vector<float> logits_fp32;
    std::vector<float> logits_q8;
    for (std::size_t pos = 0; pos < prompt.size(); ++pos) {
        logits_fp32 = m_fp32.forward_token(prompt[pos], pos);
        logits_q8 = m_q8.forward_token(prompt[pos], pos);
    }

    const double sim = cosine_sim(logits_fp32, logits_q8);
    require(sim > 0.995, "Q8_0 end-to-end transformer logits drifted from FP32");

    // Verify deterministic generation across repeated calls.
    const auto g1 = m_q8.generate(prompt, 8, 0.7f, 0.9f, 99);
    const auto g2 = m_q8.generate(prompt, 8, 0.7f, 0.9f, 99);
    require(g1.generated_tokens.size() == 8, "Unexpected generated token count");
    require(g1.generated_tokens == g2.generated_tokens, "Seeded generation is not deterministic");

    std::filesystem::remove(fp32_path);
    std::filesystem::remove(q8_path);
}

}  // namespace

int main() {
    test_mmap_fp32_vs_q8_end_to_end();
    std::cout << "[PASS] All C++20 mmap Transformer runtime tests passed.\n";
    return 0;
}
