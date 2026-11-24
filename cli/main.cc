#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tinyinfer/transformer.hpp"
#include "tinyinfer/types.hpp"

namespace {

std::vector<std::uint32_t> parse_csv_tokens(const std::string& raw) {
    std::vector<std::uint32_t> tokens;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            tokens.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        }
    }
    return tokens;
}

void print_usage() {
    std::cout
        << "Usage: tinyinfer_cli [--model <path.tinf> | --synthetic] [options]\n"
        << "Options:\n"
        << "  --dtype <fp32|q8_0>   Weight dtype for --synthetic (default: q8_0)\n"
        << "  --prompt <id,id,...>  Comma-separated prompt token IDs (default: 1,42,7,13)\n"
        << "  --max-tokens <N>      Tokens to generate (default: 32)\n"
        << "  --temp <T>            Sampling temperature, 0 = greedy (default: 0.0)\n"
        << "  --top-p <P>           Nucleus sampling top-p (default: 0.9)\n"
        << "  --seed <S>            RNG seed (default: 42)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    bool synthetic = false;
    std::string dtype_str = "q8_0";
    std::string prompt_str = "1,42,7,13";
    std::size_t max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 0.9f;
    std::uint32_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--synthetic") {
            synthetic = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--dtype" && i + 1 < argc) {
            dtype_str = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt_str = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoull(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (!synthetic && model_path.empty()) {
        print_usage();
        return 1;
    }

    bool cleanup_temp = false;
    if (synthetic) {
        model_path = (std::filesystem::temp_directory_path() / "tinyinfer_synth.tinf").string();
        cleanup_temp = true;
        tinyinfer::ModelConfig cfg;
        cfg.vocab_size = 2048;
        cfg.dim = 512;
        cfg.hidden_dim = 1536;
        cfg.n_layers = 6;
        cfg.n_q_heads = 8;
        cfg.n_kv_heads = 2;
        cfg.max_seq_len = 256;
        cfg.tie_embeddings = true;
        cfg.dtype = (dtype_str == "fp32") ? tinyinfer::WeightDType::FP32
                                          : tinyinfer::WeightDType::Q8_0;
        tinyinfer::write_synthetic_tinf(model_path, cfg, seed);
    }

    try {
        const auto info = tinyinfer::get_build_info();
        tinyinfer::TransformerModel model(model_path);
        const auto prompt = parse_csv_tokens(prompt_str);
        const auto res = model.generate(prompt, max_tokens, temperature, top_p, seed);

        const auto& cfg = model.config();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Build      : " << info.compiler
                  << " | AVX2+FMA=" << (info.avx2_fma ? "ON" : "OFF")
                  << " | OpenMP=" << (info.openmp ? "ON" : "OFF")
                  << " (" << info.max_threads << " threads)\n";
        std::cout << "Model      : layers=" << cfg.n_layers << " dim=" << cfg.dim
                  << " hidden=" << cfg.hidden_dim << " heads=" << cfg.n_q_heads << "/"
                  << cfg.n_kv_heads << " vocab=" << cfg.vocab_size << " dtype="
                  << (cfg.dtype == tinyinfer::WeightDType::Q8_0 ? "Q8_0" : "FP32") << "\n";
        std::cout << "Footprint  : mmap=" << (model.mapped_bytes() / (1024.0 * 1024.0))
                  << " MB | kv_cache=" << (model.kv_cache_bytes() / (1024.0 * 1024.0))
                  << " MB | peak_rss=" << res.peak_rss_mb << " MB\n";
        std::cout << "Latency    : prefill(" << prompt.size() << " tok)=" << res.prefill_ms
                  << " ms | decode(" << res.generated_tokens.size() << " tok)=" << res.decode_ms
                  << " ms (" << res.decode_tok_per_sec << " tok/s)\n";
        std::cout << "Tokens     :";
        for (std::uint32_t t : res.generated_tokens) {
            std::cout << " " << t;
        }
        std::cout << "\n";
    } catch (const std::exception& exc) {
        if (cleanup_temp) {
            std::filesystem::remove(model_path);
        }
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }

    if (cleanup_temp) {
        std::filesystem::remove(model_path);
    }
    return 0;
}
