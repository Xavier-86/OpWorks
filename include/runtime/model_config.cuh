#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "json.cuh"

namespace opworks {

// Model hyperparameters, sourced from the pack manifest's "config" object
// (copied from the HF config.json at export time).
struct ModelConfig {
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    float rms_norm_eps = 0.f;
    float rope_theta = 0.f;
    bool rope_llama3 = false;
    float rope_factor = 0.f;
    float rope_low_freq = 0.f;
    float rope_high_freq = 0.f;
    int rope_original_max_pos = 0;
    std::vector<int32_t> eos_token_ids;

    int kv_width() const { return num_kv_heads * head_dim; }
    int q_width() const { return num_q_heads * head_dim; }
    int kv_rep() const { return num_q_heads / num_kv_heads; }

    static ModelConfig from_manifest(const json::Value &manifest) {
        const json::Value &c = manifest.at("config");
        ModelConfig m;
        m.num_layers = static_cast<int>(c.at("num_hidden_layers").as_int());
        m.hidden_size = static_cast<int>(c.at("hidden_size").as_int());
        m.intermediate_size = static_cast<int>(c.at("intermediate_size").as_int());
        m.num_q_heads = static_cast<int>(c.at("num_attention_heads").as_int());
        m.num_kv_heads = static_cast<int>(c.at("num_key_value_heads").as_int());
        m.head_dim = static_cast<int>(c.at("head_dim").as_int());
        m.vocab_size = static_cast<int>(c.at("vocab_size").as_int());
        m.rms_norm_eps = static_cast<float>(c.at("rms_norm_eps").as_number());
        m.rope_theta = static_cast<float>(c.at("rope_theta").as_number());
        if (const json::Value *rs = c.find("rope_scaling"); rs && !rs->is_null()) {
            if (rs->at("rope_type").as_string() != "llama3")
                throw std::runtime_error("unsupported rope_scaling type");
            m.rope_llama3 = true;
            m.rope_factor = static_cast<float>(rs->at("factor").as_number());
            m.rope_low_freq = static_cast<float>(rs->at("low_freq_factor").as_number());
            m.rope_high_freq = static_cast<float>(rs->at("high_freq_factor").as_number());
            m.rope_original_max_pos = static_cast<int>(rs->at("original_max_position_embeddings").as_int());
        }
        for (const auto &v : manifest.at("eos_token_ids").as_array())
            m.eos_token_ids.push_back(static_cast<int32_t>(v.as_int()));
        if (m.num_layers <= 0 || m.hidden_size <= 0 || m.head_dim <= 0 || m.num_q_heads <= 0 ||
            m.num_kv_heads <= 0 || m.num_q_heads % m.num_kv_heads != 0)
            throw std::runtime_error("invalid model config in pack manifest");
        if (m.num_q_heads * m.head_dim != m.hidden_size)
            throw std::runtime_error("num_attention_heads * head_dim != hidden_size");
        return m;
    }

    // inv_freq[i] for head_dim/2 pairs, llama3 wavelength scaling applied.
    // Computed in double, returned as float.
    std::vector<float> rope_inv_freq() const {
        const int half = head_dim / 2;
        std::vector<float> out(half);
        const double pi = 3.14159265358979323846;
        for (int i = 0; i < half; ++i) {
            double f = 1.0 / std::pow(static_cast<double>(rope_theta), (2.0 * i) / head_dim);
            if (rope_llama3) {
                double wavelen = 2.0 * pi / f;
                double low_wl = rope_original_max_pos / rope_low_freq;
                double high_wl = rope_original_max_pos / rope_high_freq;
                if (wavelen > low_wl) {
                    f /= rope_factor;
                } else if (wavelen >= high_wl) {
                    double smooth = (rope_original_max_pos / wavelen - rope_low_freq) /
                                    (rope_high_freq - rope_low_freq);
                    f = (1.0 - smooth) * f / rope_factor + smooth * f;
                }
            }
            out[i] = static_cast<float>(f);
        }
        return out;
    }
};

} // namespace opworks
