#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>

#include "tinyinfer/attention.hpp"
#include "tinyinfer/cuda_backend.hpp"
#include "tinyinfer/kv_cache.hpp"
#include "tinyinfer/matmul.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/quant.hpp"
#include "tinyinfer/transformer.hpp"
#include "tinyinfer/types.hpp"

namespace nb = nanobind;

namespace {

using ConstF32Mat = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ConstF32Vec = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using MutF32Vec = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, float, nb::ndim<2>> make_mat2d(std::size_t rows, std::size_t cols) {
    auto* buf = new float[rows * cols];
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::ndim<2>>(buf, {rows, cols}, owner);
}

nb::ndarray<nb::numpy, float, nb::ndim<1>> make_vec1d(std::size_t n) {
    auto* buf = new float[n];
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(buf, {n}, owner);
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "tinyinfer C++20 quantized tensor kernels, GQA attention, mmap transformer, and CUDA";

    nb::enum_<tinyinfer::WeightDType>(m, "WeightDType")
        .value("FP32", tinyinfer::WeightDType::FP32)
        .value("Q8_0", tinyinfer::WeightDType::Q8_0);

    nb::class_<tinyinfer::BuildInfo>(m, "BuildInfo")
        .def_ro("avx2_fma", &tinyinfer::BuildInfo::avx2_fma)
        .def_ro("openmp", &tinyinfer::BuildInfo::openmp)
        .def_ro("cuda", &tinyinfer::BuildInfo::cuda)
        .def_ro("max_threads", &tinyinfer::BuildInfo::max_threads)
        .def_ro("compiler", &tinyinfer::BuildInfo::compiler);

    nb::class_<tinyinfer::Q8Matrix>(m, "Q8Matrix")
        .def_ro("rows", &tinyinfer::Q8Matrix::rows)
        .def_ro("cols", &tinyinfer::Q8Matrix::cols)
        .def_ro("blocks_per_row", &tinyinfer::Q8Matrix::blocks_per_row)
        .def("memory_bytes", &tinyinfer::Q8Matrix::memory_bytes)
        .def("compression_ratio", &tinyinfer::Q8Matrix::compression_ratio);

    nb::class_<tinyinfer::KVCache>(m, "KVCache")
        .def(
            nb::init<std::size_t, std::size_t, std::size_t, std::size_t>(),
            nb::arg("n_layers"),
            nb::arg("max_seq_len"),
            nb::arg("n_kv_heads"),
            nb::arg("head_dim")
        )
        .def_prop_ro("n_layers", &tinyinfer::KVCache::n_layers)
        .def_prop_ro("max_seq_len", &tinyinfer::KVCache::max_seq_len)
        .def_prop_ro("n_kv_heads", &tinyinfer::KVCache::n_kv_heads)
        .def_prop_ro("head_dim", &tinyinfer::KVCache::head_dim)
        .def_prop_ro("kv_dim", &tinyinfer::KVCache::kv_dim)
        .def("seq_len", &tinyinfer::KVCache::seq_len, nb::arg("layer") = 0)
        .def("memory_bytes", &tinyinfer::KVCache::memory_bytes)
        .def("reset", &tinyinfer::KVCache::reset);

    nb::class_<tinyinfer::ModelConfig>(m, "ModelConfig")
        .def(nb::init<>())
        .def_rw("vocab_size", &tinyinfer::ModelConfig::vocab_size)
        .def_rw("dim", &tinyinfer::ModelConfig::dim)
        .def_rw("hidden_dim", &tinyinfer::ModelConfig::hidden_dim)
        .def_rw("n_layers", &tinyinfer::ModelConfig::n_layers)
        .def_rw("n_q_heads", &tinyinfer::ModelConfig::n_q_heads)
        .def_rw("n_kv_heads", &tinyinfer::ModelConfig::n_kv_heads)
        .def_rw("max_seq_len", &tinyinfer::ModelConfig::max_seq_len)
        .def_rw("tie_embeddings", &tinyinfer::ModelConfig::tie_embeddings)
        .def_rw("dtype", &tinyinfer::ModelConfig::dtype)
        .def_rw("rms_norm_eps", &tinyinfer::ModelConfig::rms_norm_eps)
        .def_rw("rope_theta", &tinyinfer::ModelConfig::rope_theta);

    nb::class_<tinyinfer::GenerationResult>(m, "GenerationResult")
        .def_ro("generated_tokens", &tinyinfer::GenerationResult::generated_tokens)
        .def_ro("prefill_ms", &tinyinfer::GenerationResult::prefill_ms)
        .def_ro("decode_ms", &tinyinfer::GenerationResult::decode_ms)
        .def_ro("decode_tok_per_sec", &tinyinfer::GenerationResult::decode_tok_per_sec)
        .def_ro("peak_rss_mb", &tinyinfer::GenerationResult::peak_rss_mb);

    nb::class_<tinyinfer::TransformerModel>(m, "TransformerModel")
        .def(nb::init<const std::string&>(), nb::arg("tinf_path"))
        .def("config", &tinyinfer::TransformerModel::config)
        .def("mapped_bytes", &tinyinfer::TransformerModel::mapped_bytes)
        .def("kv_cache_bytes", &tinyinfer::TransformerModel::kv_cache_bytes)
        .def("reset_cache", &tinyinfer::TransformerModel::reset_cache)
        .def(
            "forward_token",
            [](tinyinfer::TransformerModel& self, std::uint32_t token_id, std::size_t pos) {
                const auto& logits = self.forward_token(token_id, pos);
                auto out = make_vec1d(logits.size());
                std::copy(logits.begin(), logits.end(), out.data());
                return out;
            },
            nb::arg("token_id"),
            nb::arg("pos")
        )
        .def(
            "generate",
            &tinyinfer::TransformerModel::generate,
            nb::arg("prompt_tokens"),
            nb::arg("max_new_tokens"),
            nb::arg("temperature") = 0.0f,
            nb::arg("top_p") = 0.9f,
            nb::arg("seed") = 42
        );

    m.def("get_build_info", &tinyinfer::get_build_info);
    m.def("cuda_available", &tinyinfer::cuda_available);
    m.def("cuda_device_name", &tinyinfer::cuda_device_name);
    m.def("process_peak_rss_mb", &tinyinfer::process_peak_rss_mb);
    m.def(
        "write_synthetic_tinf",
        &tinyinfer::write_synthetic_tinf,
        nb::arg("path"),
        nb::arg("config"),
        nb::arg("seed") = 42
    );

    m.def("quantize_q8_0", [](const ConstF32Mat& w) {
        return tinyinfer::quantize_q8_0(w.data(), w.shape(0), w.shape(1));
    });

    m.def("quantize_q8_0_arrays", [](const ConstF32Mat& w) {
        auto qmat = tinyinfer::quantize_q8_0(w.data(), w.shape(0), w.shape(1));
        auto* s_buf = new float[qmat.scales.size()];
        std::copy(qmat.scales.begin(), qmat.scales.end(), s_buf);
        nb::capsule s_owner(s_buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        auto s_arr = nb::ndarray<nb::numpy, float, nb::ndim<2>>(
            s_buf, {qmat.rows, qmat.blocks_per_row}, s_owner
        );

        auto* w_buf = new int8_t[qmat.weights.size()];
        std::copy(qmat.weights.begin(), qmat.weights.end(), w_buf);
        nb::capsule w_owner(w_buf, [](void* p) noexcept { delete[] static_cast<int8_t*>(p); });
        auto w_arr = nb::ndarray<nb::numpy, int8_t, nb::ndim<2>>(
            w_buf, {qmat.rows, qmat.blocks_per_row * tinyinfer::kQ8BlockSize}, w_owner
        );
        return nb::make_tuple(s_arr, w_arr);
    });

    m.def("dequantize_q8_0", [](const tinyinfer::Q8Matrix& qmat) {
        auto vec = tinyinfer::dequantize_q8_0(qmat);
        auto out = make_mat2d(qmat.rows, qmat.cols);
        std::copy(vec.begin(), vec.end(), out.data());
        return out;
    });

    m.def("matmul_fp32_nt", [](const ConstF32Mat& x, const ConstF32Mat& w) {
        if (x.shape(1) != w.shape(1)) {
            throw std::invalid_argument("matmul_fp32_nt: inner dimension K mismatch.");
        }
        auto out = make_mat2d(x.shape(0), w.shape(0));
        tinyinfer::matmul_fp32_nt(x.data(), w.data(), out.data(), x.shape(0), w.shape(0), x.shape(1));
        return out;
    });

    m.def("matmul_q8_0_nt", [](const ConstF32Mat& x, const tinyinfer::Q8Matrix& w_q8) {
        if (x.shape(1) != w_q8.cols) {
            throw std::invalid_argument("matmul_q8_0_nt: inner dimension K mismatch.");
        }
        auto out = make_mat2d(x.shape(0), w_q8.rows);
        tinyinfer::matmul_q8_0_nt(x.data(), w_q8, out.data(), x.shape(0));
        return out;
    });

    m.def("cuda_matmul_fp32_nt", [](const ConstF32Mat& x, const ConstF32Mat& w) {
        if (x.shape(1) != w.shape(1)) {
            throw std::invalid_argument("cuda_matmul_fp32_nt: inner dimension K mismatch.");
        }
        auto out = make_mat2d(x.shape(0), w.shape(0));
        tinyinfer::cuda_matmul_fp32_nt(
            x.data(), w.data(), out.data(), x.shape(0), w.shape(0), x.shape(1)
        );
        return out;
    });

    m.def("cuda_matmul_q8_0_nt", [](const ConstF32Mat& x, const tinyinfer::Q8Matrix& w_q8) {
        if (x.shape(1) != w_q8.cols) {
            throw std::invalid_argument("cuda_matmul_q8_0_nt: inner dimension K mismatch.");
        }
        auto out = make_mat2d(x.shape(0), w_q8.rows);
        tinyinfer::cuda_matmul_q8_0_nt(x.data(), w_q8, out.data(), x.shape(0));
        return out;
    });

    m.def(
        "rmsnorm",
        [](const ConstF32Vec& x, const ConstF32Vec& weight, float eps) {
            if (x.shape(0) != weight.shape(0)) {
                throw std::invalid_argument("rmsnorm: x and weight length mismatch.");
            }
            auto out = make_vec1d(x.shape(0));
            tinyinfer::rmsnorm(out.data(), x.data(), weight.data(), x.shape(0), eps);
            return out;
        },
        nb::arg("x"),
        nb::arg("weight"),
        nb::arg("eps") = 1e-5f
    );

    m.def("softmax", [](const ConstF32Vec& x) {
        auto out = make_vec1d(x.shape(0));
        tinyinfer::softmax(out.data(), x.data(), x.shape(0));
        return out;
    });

    m.def("silu", [](const ConstF32Vec& x) {
        auto out = make_vec1d(x.shape(0));
        tinyinfer::silu(out.data(), x.data(), x.shape(0));
        return out;
    });

    m.def(
        "rope_inplace",
        [](MutF32Vec x, std::size_t n_heads, std::size_t head_dim, std::size_t pos, float theta) {
            if (x.shape(0) != n_heads * head_dim) {
                throw std::invalid_argument("rope_inplace: vector length != n_heads * head_dim.");
            }
            tinyinfer::rope_inplace(x.data(), n_heads, head_dim, pos, theta);
        },
        nb::arg("x"),
        nb::arg("n_heads"),
        nb::arg("head_dim"),
        nb::arg("pos"),
        nb::arg("theta_base") = 10000.0f
    );

    m.def(
        "gqa_decode_step",
        [](
            const ConstF32Vec& q,
            const ConstF32Vec& k_new,
            const ConstF32Vec& v_new,
            tinyinfer::KVCache& cache,
            std::size_t layer,
            std::size_t pos,
            std::size_t n_q_heads
        ) {
            const std::size_t expected_q = n_q_heads * cache.head_dim();
            if (q.shape(0) != expected_q) {
                throw std::invalid_argument("gqa_decode_step: q length != n_q_heads * head_dim.");
            }
            if (k_new.shape(0) != cache.kv_dim() || v_new.shape(0) != cache.kv_dim()) {
                throw std::invalid_argument("gqa_decode_step: k_new/v_new length != kv_dim.");
            }
            auto out = make_vec1d(expected_q);
            tinyinfer::gqa_decode_step(
                q.data(), k_new.data(), v_new.data(), cache, layer, pos, n_q_heads, out.data()
            );
            return out;
        },
        nb::arg("q"),
        nb::arg("k_new"),
        nb::arg("v_new"),
        nb::arg("cache"),
        nb::arg("layer"),
        nb::arg("pos"),
        nb::arg("n_q_heads")
    );

    m.def(
        "gqa_prefill",
        [](
            const ConstF32Mat& q,
            const ConstF32Mat& k,
            const ConstF32Mat& v,
            tinyinfer::KVCache& cache,
            std::size_t layer,
            std::size_t start_pos,
            std::size_t n_q_heads
        ) {
            const std::size_t seq_len = q.shape(0);
            const std::size_t expected_q = n_q_heads * cache.head_dim();
            if (k.shape(0) != seq_len || v.shape(0) != seq_len) {
                throw std::invalid_argument("gqa_prefill: q, k, v must share seq_len.");
            }
            if (q.shape(1) != expected_q) {
                throw std::invalid_argument("gqa_prefill: q width != n_q_heads * head_dim.");
            }
            if (k.shape(1) != cache.kv_dim() || v.shape(1) != cache.kv_dim()) {
                throw std::invalid_argument("gqa_prefill: k/v width != kv_dim.");
            }
            auto out = make_mat2d(seq_len, expected_q);
            tinyinfer::gqa_prefill(
                q.data(), k.data(), v.data(), cache, layer, start_pos, seq_len, n_q_heads, out.data()
            );
            return out;
        },
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("cache"),
        nb::arg("layer"),
        nb::arg("start_pos"),
        nb::arg("n_q_heads")
    );
}
