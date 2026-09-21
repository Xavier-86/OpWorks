#pragma once

// Chunked (online-softmax) causal GQA prefill attention. Avoids the full
// [heads, T, S] score matrix, so prefill length is not bounded by a quadratic
// workspace. FP32 in/out.
//
// Block tile: 32 query rows x 32 keys, head_dim <= 64, 128 threads.
// Thread t owns query row r = t/4 throughout; the 4 threads of a row split
// the 32 keys (scores) and the 64 head dims (accumulator) between them.

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

constexpr int kAttnBr = 32; // query rows per tile
constexpr int kAttnBc = 32; // keys per tile
constexpr int kAttnMaxD = 64;
constexpr int kAttnThreads = 128;

// q_heads:   [qh, T, D]   queries for rows [0, T), absolute positions
//                         [q_offset, q_offset + T)
// k/v cache: [kvh, cap, D] holding S valid entries
// out:       [qh, T, D]
static __global__ void attention_prefill_chunked_kernel(const float *__restrict__ q_heads,
                                                        const float *__restrict__ kcache,
                                                        const float *__restrict__ vcache,
                                                        float *__restrict__ out, int T, int S, int q_offset,
                                                        int head_dim, int kv_rep, int cap, float scale) {
    constexpr int Br = kAttnBr, Bc = kAttnBc;
    __shared__ float q_tile[Br][kAttnMaxD + 1];
    __shared__ float k_tile[Bc][kAttnMaxD + 1];
    __shared__ float v_tile[Bc][kAttnMaxD + 1];
    __shared__ float probs[Br][Bc];
    __shared__ float run_max[Br];
    __shared__ float run_sum[Br];

    const int h = blockIdx.y;
    const int qt = blockIdx.x * Br;
    const float *q_head = q_heads + static_cast<int64_t>(h) * T * head_dim;
    const float *k_head = kcache + static_cast<int64_t>(h / kv_rep) * cap * head_dim;
    const float *v_head = vcache + static_cast<int64_t>(h / kv_rep) * cap * head_dim;
    float *o_head = out + static_cast<int64_t>(h) * T * head_dim;

    const int r = threadIdx.x / 4;   // query row within tile [0, Br)
    const int part = threadIdx.x % 4;

    // load Q tile (zero-padded past T)
    for (int i = threadIdx.x; i < Br * head_dim; i += kAttnThreads) {
        int row = i / head_dim, d = i % head_dim;
        int gr = qt + row;
        q_tile[row][d] = gr < T ? q_head[static_cast<int64_t>(gr) * head_dim + d] : 0.f;
    }
    if (part == 0) {
        run_max[r] = kNegInf;
        run_sum[r] = 0.f;
    }
    __syncthreads();

    // accumulator: this thread owns dims [part*16, part*16+16) of row r
    float acc[16];

    const int qpos = q_offset + qt + r;           // absolute query position
    const int diag = q_offset + qt + Br - 1;      // furthest key this tile may see
    const int last_kc = diag / Bc < (S + Bc - 1) / Bc - 1 ? diag / Bc : (S + Bc - 1) / Bc - 1;

    for (int kc = 0; kc <= last_kc; ++kc) {
        const int k0 = kc * Bc;
        const int kn = S - k0 < Bc ? S - k0 : Bc; // valid keys in this tile

        for (int i = threadIdx.x; i < Bc * head_dim; i += kAttnThreads) {
            int row = i / head_dim, d = i % head_dim;
            k_tile[row][d] = row < kn ? k_head[static_cast<int64_t>(k0 + row) * head_dim + d] : 0.f;
        }
        __syncthreads();

        // scores for this row, keys [part*8, part*8+8)
        float local_max = kNegInf;
#pragma unroll
        for (int jj = 0; jj < Bc / 4; ++jj) {
            int j = part * (Bc / 4) + jj;
            float s = kNegInf;
            if (j < kn && k0 + j <= qpos) {
                s = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    s += q_tile[r][d] * k_tile[j][d];
                s *= scale;
            }
            probs[r][j] = s;
            local_max = fmaxf(local_max, s);
        }
        // row max over the 4 owner threads (consecutive lanes)
#pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(kFullWarpMask, local_max, off, 4));

        float m_old = run_max[r];
        float m_new = fmaxf(m_old, local_max);
        float corr = m_old == kNegInf ? 0.f : expf(m_old - m_new);

        float p_sum = 0.f;
#pragma unroll
        for (int jj = 0; jj < Bc / 4; ++jj) {
            int j = part * (Bc / 4) + jj;
            float p = probs[r][j] == kNegInf ? 0.f : expf(probs[r][j] - m_new);
            probs[r][j] = p;
            p_sum += p;
        }
#pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            p_sum += __shfl_xor_sync(kFullWarpMask, p_sum, off, 4);

        if (part == 0) {
            run_max[r] = m_new;
            run_sum[r] = run_sum[r] * corr + p_sum;
        }
        __syncthreads(); // probs finalized for the row; run_max/sum updated

        // load V tile while rescaling/accumulating
        for (int i = threadIdx.x; i < Bc * head_dim; i += kAttnThreads) {
            int row = i / head_dim, d = i % head_dim;
            v_tile[row][d] = row < kn ? v_head[static_cast<int64_t>(k0 + row) * head_dim + d] : 0.f;
        }
        __syncthreads();

        if (kc == 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                acc[i] = 0.f;
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                acc[i] *= corr;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int d = part * 16 + i;
            float a = 0.f;
            for (int j = 0; j < Bc; ++j)
                a += probs[r][j] * v_tile[j][d];
            acc[i] += a;
        }
        __syncthreads(); // probs reused next iteration
    }

    const int gr = qt + r;
    if (gr < T) {
        float inv = run_sum[r] > 0.f ? 1.f / run_sum[r] : 0.f; // all-masked row -> zeros
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int d = part * 16 + i;
            if (d < head_dim)
                o_head[static_cast<int64_t>(gr) * head_dim + d] = acc[i] * inv;
        }
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Chunked causal attention over the KV cache for rows>1 (prefill).
// Same tensor semantics as the explicit QK/softmax/PV path.
inline void attention_prefill_chunked(const float *q_heads, const float *kcache, const float *vcache,
                                      float *out, int T, int S, int q_offset, int num_q_heads, int kv_rep,
                                      int head_dim, int cap, cudaStream_t stream = nullptr) {
    detail::require(T > 0 && S >= T && q_offset == S - T, "attention_prefill_chunked: bad shape/offset");
    detail::require(num_q_heads > 0 && kv_rep > 0, "attention_prefill_chunked: bad head shape");
    detail::require(head_dim > 0 && head_dim <= detail::kAttnMaxD && head_dim % 16 == 0,
                    "attention_prefill_chunked supports head_dim in {16,32,48,64}");
    detail::require(cap >= S, "attention_prefill_chunked: length exceeds cache capacity");
    detail::require(q_heads && kcache && vcache && out, "attention_prefill_chunked requires non-null pointers");
    float scale = 1.f / sqrtf(static_cast<float>(head_dim));
    launch_async(stream, detail::attention_prefill_chunked_kernel,
                 dim3(detail::blocks_for(T, detail::kAttnBr), num_q_heads), dim3(detail::kAttnThreads), q_heads,
                 kcache, vcache, out, T, S, q_offset, head_dim, kv_rep, cap, scale);
}

} // namespace opworks::ops
