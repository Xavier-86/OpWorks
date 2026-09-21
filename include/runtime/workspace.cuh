#pragma once

#include "../core/typed_device_buffer.cuh"
#include "model_config.cuh"

namespace opworks {

// Per-session scratch buffers, allocated once at session creation and reused
// by every forward pass. Sizes cover the largest shape each buffer holds.
struct Workspace {
    TypedDeviceBuffer<float> hidden;   // [T, H] residual stream
    TypedDeviceBuffer<float> normed;   // [T, H]
    TypedDeviceBuffer<float> q;        // [T, qh*D] packed, post-RoPE
    TypedDeviceBuffer<float> q_heads;  // [qh, T, D] split for prefill
    TypedDeviceBuffer<float> k_proj;   // [T, kvh*D]
    TypedDeviceBuffer<float> v_proj;   // [T, kvh*D]
    TypedDeviceBuffer<float> attn_out; // [T, qh*D] packed attention result
    TypedDeviceBuffer<float> scores;   // [T, S] per-head score matrix (reused per head)
    TypedDeviceBuffer<float> attn_heads; // [qh, T, D] per-head attention output
    TypedDeviceBuffer<float> gate;     // [T, I]
    TypedDeviceBuffer<float> up;       // [T, I]
    TypedDeviceBuffer<float> logits;   // [V] last-position logits
    TypedDeviceBuffer<int32_t> argmax_out; // [1]
    TypedDeviceBuffer<int32_t> tokens;     // [T] current step input ids

    Workspace() = default;
    Workspace(const ModelConfig &cfg, int max_seq_len) {
        const int64_t T = max_seq_len;
        const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
        hidden = TypedDeviceBuffer<float>(T * H);
        normed = TypedDeviceBuffer<float>(T * H);
        q = TypedDeviceBuffer<float>(T * cfg.q_width());
        q_heads = TypedDeviceBuffer<float>(T * cfg.q_width());
        k_proj = TypedDeviceBuffer<float>(T * cfg.kv_width());
        v_proj = TypedDeviceBuffer<float>(T * cfg.kv_width());
        attn_out = TypedDeviceBuffer<float>(T * cfg.q_width());
        // The explicit QK/softmax/PV prefill path needs a [rows, rows] score
        // matrix and is only used up to this many rows; longer prefills take
        // the chunked online-softmax path (ops::attention_prefill_chunked).
        constexpr int64_t kExplicitAttnLimit = 2048;
        const int64_t score_rows = T < kExplicitAttnLimit ? T : kExplicitAttnLimit;
        scores = TypedDeviceBuffer<float>(score_rows * score_rows);
        attn_heads = TypedDeviceBuffer<float>(T * cfg.q_width());
        gate = TypedDeviceBuffer<float>(T * I);
        up = TypedDeviceBuffer<float>(T * I);
        logits = TypedDeviceBuffer<float>(cfg.vocab_size);
        argmax_out = TypedDeviceBuffer<int32_t>(1);
        tokens = TypedDeviceBuffer<int32_t>(T);
    }

    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    Workspace(Workspace &&) = default;
    Workspace &operator=(Workspace &&) = default;
};

} // namespace opworks
