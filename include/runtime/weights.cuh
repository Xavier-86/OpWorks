#pragma once

#include <cuda_bf16.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/typed_device_buffer.cuh"
#include "model_config.cuh"
#include "pack_reader.cuh"

namespace opworks {

// Per-layer decoder weights, device-resident. WT is the storage type of the
// projection matrices (float or __nv_bfloat16); norm weights are always fp32.
template <typename WT> struct LayerWeights {
    TypedDeviceBuffer<float> input_norm;    // [H]
    TypedDeviceBuffer<WT> q_proj;           // [qh*D, H]
    TypedDeviceBuffer<WT> k_proj;           // [kvh*D, H]
    TypedDeviceBuffer<WT> v_proj;           // [kvh*D, H]
    TypedDeviceBuffer<WT> o_proj;           // [H, qh*D]
    TypedDeviceBuffer<float> post_norm;     // [H]
    TypedDeviceBuffer<WT> gate_proj;        // [I, H]
    TypedDeviceBuffer<WT> up_proj;          // [I, H]
    TypedDeviceBuffer<WT> down_proj;        // [H, I]
};

// All model weights on device, shared across sessions.
template <typename WT> struct ModelWeights {
    TypedDeviceBuffer<WT> embed_tokens;     // [V, H]; also the LM head (tied)
    TypedDeviceBuffer<float> final_norm;    // [H]
    std::vector<LayerWeights<WT>> layers;
    TypedDeviceBuffer<float> rope_inv_freq; // [head_dim/2]

    static ModelWeights load(const PackReader &pack, const ModelConfig &cfg, cudaStream_t stream = nullptr) {
        if constexpr (std::is_same_v<WT, float>) {
            if (pack.dtype() != "float32")
                throw std::runtime_error("fp32 model requires a float32 pack");
        } else {
            if (pack.dtype() != "bfloat16")
                throw std::runtime_error("bf16 model requires a bfloat16 pack");
        }

        ModelWeights w;
        auto upload = [&](const std::string &name, int64_t expected_numel) {
            const PackReader::TensorMeta &m = pack.meta(name);
            if (m.numel != expected_numel)
                throw std::runtime_error(name + ": element count mismatch");
            if constexpr (std::is_same_v<WT, float>) {
                return TypedDeviceBuffer<WT>::from_host(pack.read(name), stream);
            } else {
                std::vector<uint8_t> raw = pack.read_raw(name);
                return TypedDeviceBuffer<WT>::from_host(reinterpret_cast<const WT *>(raw.data()),
                                                        expected_numel, stream);
            }
        };
        auto upload_norm = [&](const std::string &name, int64_t expected_numel) {
            const PackReader::TensorMeta &m = pack.meta(name);
            if (m.numel != expected_numel)
                throw std::runtime_error(name + ": element count mismatch");
            if constexpr (std::is_same_v<WT, float>) {
                return TypedDeviceBuffer<float>::from_host(pack.read(name), stream);
            } else {
                return TypedDeviceBuffer<float>::from_host(pack.read_bf16_as_float(name), stream);
            }
        };

        const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
        const int64_t qw = cfg.q_width(), kw = cfg.kv_width();
        w.embed_tokens = upload("model.embed_tokens.weight", static_cast<int64_t>(cfg.vocab_size) * H);
        w.final_norm = upload_norm("model.norm.weight", H);
        w.layers.reserve(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            LayerWeights<WT> lw;
            lw.input_norm = upload_norm(p + "input_layernorm.weight", H);
            lw.q_proj = upload(p + "self_attn.q_proj.weight", qw * H);
            lw.k_proj = upload(p + "self_attn.k_proj.weight", kw * H);
            lw.v_proj = upload(p + "self_attn.v_proj.weight", kw * H);
            lw.o_proj = upload(p + "self_attn.o_proj.weight", H * qw);
            lw.post_norm = upload_norm(p + "post_attention_layernorm.weight", H);
            lw.gate_proj = upload(p + "mlp.gate_proj.weight", I * H);
            lw.up_proj = upload(p + "mlp.up_proj.weight", I * H);
            lw.down_proj = upload(p + "mlp.down_proj.weight", H * I);
            w.layers.push_back(std::move(lw));
        }
        // Verify the tying alias exists and matches the manifest.
        if (!pack.has("lm_head.weight"))
            throw std::runtime_error("pack has neither lm_head.weight nor a tying alias");
        w.rope_inv_freq = TypedDeviceBuffer<float>::from_host(cfg.rope_inv_freq(), stream);
        return w;
    }
};

} // namespace opworks
