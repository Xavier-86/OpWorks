#pragma once

#include <cuda_bf16.h>

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }

// Y[rows, out] = X[rows, in] @ W[out, in]^T, W read transposed (no materialized
// transpose). WT is float or __nv_bfloat16; accumulation is always fp32.
// Same tiling as mat_mul_kernel; the B tile is loaded from W with the flat
// index arranged so consecutive threads read consecutive `in` — coalesced
// global loads, padded transposed shared stores.
template <typename WT>
static __global__ void linear_kernel(const float *X, const WT *W, float *Y, int M, int N, int K) {
    constexpr int kTile = kMatmulTile;
    constexpr int kBlock = kMatmulBlock;
    constexpr int kSub = kMatmulSub;
    __shared__ float tileA[kTile][kBlock + 1];
    __shared__ float tileB[kBlock][kTile + 1]; // [in-slice][out-slice]

    const int block_row = blockIdx.y * kTile;
    const int block_col = blockIdx.x * kTile;

    float acc[kSub][kSub] = {};

    for (int t = 0; t < N / kBlock + (N % kBlock != 0); ++t) {
#pragma unroll
        OPWORKS_BLOCK_LOOP_FLAT(i, kTile * kBlock) {
            int r = i / kBlock, c = i % kBlock;
            int gr = block_row + r, gc = t * kBlock + c;
            tileA[r][c] = (gr < M && gc < N) ? X[static_cast<int64_t>(gr) * N + gc] : 0.f;
        }
#pragma unroll
        OPWORKS_BLOCK_LOOP_FLAT(i, kBlock * kTile) {
            int c = i / kBlock, r = i % kBlock; // consecutive threads sweep `in`
            int gr = t * kBlock + r, gc = block_col + c;
            tileB[r][c] = (gr < N && gc < K) ? to_f32(W[static_cast<int64_t>(gc) * N + gr]) : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBlock; ++k) {
            float a[kSub], b[kSub];
#pragma unroll
            for (int i = 0; i < kSub; ++i) {
                a[i] = tileA[threadIdx.y * kSub + i][k];
                b[i] = tileB[k][threadIdx.x * kSub + i];
            }
#pragma unroll
            for (int i = 0; i < kSub; ++i)
#pragma unroll
                for (int j = 0; j < kSub; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kSub; ++i) {
        int r = block_row + threadIdx.y * kSub + i;
#pragma unroll
        for (int j = 0; j < kSub; ++j) {
            int c = block_col + threadIdx.x * kSub + j;
            if (r < M && c < K)
                Y[static_cast<int64_t>(r) * K + c] = acc[i][j];
        }
    }
}

// Decode path: y[out] = x[in] @ W[out, in]^T for a single row, fp32 weights.
// One block per output feature, block-reduced dot product, float4 loads.
static __global__ void gemv_kernel(const float *x, const float *W, float *y, int in, int out) {
    const float *w_row = W + static_cast<int64_t>(blockIdx.x) * in;
    float local = 0.f;
    if (in % 4 == 0) {
        const float4 *w4 = reinterpret_cast<const float4 *>(w_row);
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        int n4 = in / 4;
        OPWORKS_BLOCK_LOOP(i, n4) {
            float4 w = w4[i], xv = x4[i];
            local += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
        }
    } else {
        OPWORKS_BLOCK_LOOP(i, in) {
            local += x[i] * w_row[i];
        }
    }
    float dot = block_reduce<ReduceSum>(local);
    if (threadIdx.x == 0)
        y[blockIdx.x] = dot;
}

// bf16-weight GEMV: 8 weights per uint4 load, fp32 accumulate.
static __global__ void gemv_bf16_kernel(const float *x, const __nv_bfloat16 *W, float *y, int in, int out) {
    const __nv_bfloat16 *w_row = W + static_cast<int64_t>(blockIdx.x) * in;
    float local = 0.f;
    if (in % 8 == 0) {
        const uint4 *w8 = reinterpret_cast<const uint4 *>(w_row);
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        int n8 = in / 8;
        OPWORKS_BLOCK_LOOP(i, n8) {
            uint4 packed = w8[i];
            const __nv_bfloat16 *wp = reinterpret_cast<const __nv_bfloat16 *>(&packed);
            float4 xa = x4[2 * i], xb = x4[2 * i + 1];
            local += to_f32(wp[0]) * xa.x + to_f32(wp[1]) * xa.y + to_f32(wp[2]) * xa.z + to_f32(wp[3]) * xa.w;
            local += to_f32(wp[4]) * xb.x + to_f32(wp[5]) * xb.y + to_f32(wp[6]) * xb.z + to_f32(wp[7]) * xb.w;
        }
    } else {
        OPWORKS_BLOCK_LOOP(i, in) {
            local += x[i] * to_f32(w_row[i]);
        }
    }
    float dot = block_reduce<ReduceSum>(local);
    if (threadIdx.x == 0)
        y[blockIdx.x] = dot;
}

} // namespace opworks::detail

namespace opworks::ops {

// Y[rows, out_features] = X[rows, in_features] @ W[out_features, in_features]^T.
// Weights stay in the HF [out, in] layout; nothing is transposed on device.
// WT is float or __nv_bfloat16; activations and accumulation stay fp32.
template <typename WT>
inline void linear(const float *X, const WT *W, float *Y, int rows, int in_features, int out_features,
                   cudaStream_t stream = nullptr) {
    detail::validate_matrix(rows, in_features);
    detail::validate_matrix(out_features, in_features);
    detail::validate_matrix(rows, out_features);
    if (rows == 0 || out_features == 0)
        return;
    detail::require(X && W && Y, "linear requires non-null pointers");
    detail::require(in_features > 0, "linear requires a positive reduction dim");
    if (rows == 1) {
        if constexpr (std::is_same_v<WT, __nv_bfloat16>) {
            launch_async(stream, detail::gemv_bf16_kernel, out_features, kThreads, X, W, Y, in_features,
                         out_features);
        } else {
            launch_async(stream, detail::gemv_kernel, out_features, kThreads, X, W, Y, in_features, out_features);
        }
        return;
    }
    detail::require(detail::blocks_for(rows, kMatmulTile) <= 65535, "linear row grid exceeds CUDA grid.y limit");
    launch_async(stream, detail::linear_kernel<WT>,
                 dim3(detail::blocks_for(out_features, kMatmulTile), detail::blocks_for(rows, kMatmulTile)),
                 dim3(kMatmulBlock, kMatmulBlock), X, W, Y, rows, in_features, out_features);
}

} // namespace opworks::ops
