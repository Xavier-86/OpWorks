#pragma once

#include "../core/typed_device_buffer.cuh"
#include "model_config.cuh"

namespace opworks {

// Per-session KV cache: one [num_kv_heads, max_seq_len, head_dim] slab per
// layer for K (post-RoPE) and V (projection output). `length` tracks valid
// entries; capacity never changes after construction.
struct KvCache {
    std::vector<TypedDeviceBuffer<float>> k; // per layer
    std::vector<TypedDeviceBuffer<float>> v;
    int max_seq_len = 0;
    int length = 0; // valid entries, same for every layer

    KvCache() = default;
    KvCache(const ModelConfig &cfg, int max_seq_len) : max_seq_len(max_seq_len) {
        const int64_t slab = static_cast<int64_t>(cfg.num_kv_heads) * max_seq_len * cfg.head_dim;
        k.reserve(cfg.num_layers);
        v.reserve(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            k.emplace_back(slab);
            v.emplace_back(slab);
        }
    }

    KvCache(const KvCache &) = delete;
    KvCache &operator=(const KvCache &) = delete;
    KvCache(KvCache &&) = default;
    KvCache &operator=(KvCache &&) = default;

    // Reset only invalidates the length; stale bytes are never read.
    void reset() { length = 0; }
};

} // namespace opworks
