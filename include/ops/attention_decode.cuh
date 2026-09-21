#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// One decode query against the KV cache. One block per query head.
// q[qh, D]; k/v cache [kvh, max_seq, D] holding `len` valid entries;
// out[qh, D]. GQA: kv head = q_head / (qh / kvh).
// Scores for up to kAttnDecodeMaxLen keys live in shared memory.
constexpr int kAttnDecodeMaxLen = 8192;

static __global__ void attention_decode_kernel(const float *q, const float *kcache, const float *vcache,
                                               float *out, int len, int max_seq, int head_dim, int kv_rep,
                                               float scale) {
    const int h = blockIdx.x;
    const float *k = kcache + static_cast<int64_t>(h / kv_rep) * max_seq * head_dim;
    const float *v = vcache + static_cast<int64_t>(h / kv_rep) * max_seq * head_dim;

    __shared__ float scores[kAttnDecodeMaxLen];
    __shared__ float q_shared[64]; // head_dim <= 64 for the supported models

    OPWORKS_BLOCK_LOOP(d, head_dim) {
        q_shared[d] = q[h * head_dim + d];
    }
    __syncthreads();

    OPWORKS_BLOCK_LOOP(j, len) {
        const float *k_row = k + static_cast<int64_t>(j) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += q_shared[d] * k_row[d];
        scores[j] = dot * scale;
    }
    __syncthreads();

    // softmax over scores[0..len) with the standard max-subtracted form
    float local_max = ReduceMax::identity();
    OPWORKS_BLOCK_LOOP(j, len) {
        local_max = ReduceMax::combine(local_max, scores[j]);
    }
    float max_val = block_reduce_all<ReduceMax>(local_max);
    float local_sum = ReduceSum::identity();
    OPWORKS_BLOCK_LOOP(j, len) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    __syncthreads();
    float sum = block_reduce_all<ReduceSum>(local_sum);
    float inv_sum = 1.f / sum;

    // V accumulation parallel over keys: every thread strides over keys with
    // a full head_dim register accumulator, then warps reduce per dim.
    float acc[64]; // head_dim <= 64
#pragma unroll
    for (int d = 0; d < 64; ++d)
        acc[d] = 0.f;
    OPWORKS_BLOCK_LOOP(j, len) {
        float p = scores[j] * inv_sum;
        const float *v_row = v + static_cast<int64_t>(j) * head_dim;
#pragma unroll
        for (int d = 0; d < 64; ++d)
            if (d < head_dim)
                acc[d] += p * v_row[d];
    }
    __syncthreads();

    // reduce acc across the block: warp shuffle first, then cross-warp shared
    __shared__ float red[kMaxThreadsPerBlock / kWarpSize][64];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
#pragma unroll
    for (int d = 0; d < 64; ++d) {
        float val = acc[d];
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(kFullWarpMask, val, offset);
        if (lane == 0)
            red[warp][d] = val;
    }
    __syncthreads();
    OPWORKS_BLOCK_LOOP(d, head_dim) {
        float val = 0.f;
        for (int wq = 0; wq < nwarps; ++wq)
            val += red[wq][d];
        out[h * head_dim + d] = val;
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Single-query GQA attention over the first `len` entries of the KV cache.
// q and out are [num_q_heads, head_dim]; caches are [num_kv_heads, max_seq,
// head_dim]. kv_rep = num_q_heads / num_kv_heads.
inline void attention_decode(const float *q, const float *kcache, const float *vcache, float *out, int num_q_heads,
                             int kv_rep, int head_dim, int len, int max_seq, cudaStream_t stream = nullptr) {
    detail::require(num_q_heads > 0 && kv_rep > 0 && head_dim > 0, "attention_decode requires positive head shape");
    detail::require(head_dim <= 64, "attention_decode supports head_dim <= 64");
    detail::require(len > 0 && len <= max_seq, "attention_decode length outside the cache capacity");
    detail::require(len <= detail::kAttnDecodeMaxLen, "attention_decode length exceeds the shared-memory kernel limit");
    detail::require(q && kcache && vcache && out, "attention_decode requires non-null pointers");
    float scale = rsqrtf(static_cast<float>(head_dim));
    launch_async(stream, detail::attention_decode_kernel, num_q_heads, kThreads, q, kcache, vcache, out, len,
                 max_seq, head_dim, kv_rep, scale);
}

} // namespace opworks::ops
