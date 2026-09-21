#pragma once

// Tensor Core (WMMA) linear for bf16 weights: Y = X @ W^T with X fp32,
// W bf16 [out, in], fp32 accumulation. X is rounded to bf16 during the shared
// tile load. Used for multi-row (prefill) GEMMs in the bf16 model; the
// single-row decode GEMV stays on the SIMT path.
//
// Block tile 64x64, K-slice 32, 4 warps each computing a 32x32 sub-tile as
// 2x2 wmma fragments of 16x16x16 bf16.

#include <cuda_bf16.h>
#include <mma.h>

#include "../core/cuda_utils.cuh"
#include "linear.cuh"

namespace opworks::detail {

constexpr int kMmaBM = 64;
constexpr int kMmaBN = 64;
constexpr int kMmaBK = 32;
constexpr int kMmaWarps = 4; // 2x2 warp grid, 32x32 each

static __global__ void linear_mma_bf16_kernel(const float *__restrict__ X, const __nv_bfloat16 *__restrict__ W,
                                              float *__restrict__ Y, int M, int N, int K) {
    using namespace nvcuda;
    __shared__ __align__(16) __nv_bfloat16 As[kMmaBM][kMmaBK + 8];
    __shared__ __align__(16) __nv_bfloat16 Bs[kMmaBN][kMmaBK + 8];
    __shared__ __align__(16) float Cs[kMmaBM][kMmaBN + 4];

    const int block_row = blockIdx.y * kMmaBM;
    const int block_col = blockIdx.x * kMmaBN;
    const int warp = threadIdx.x / kWarpSize;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < N; k0 += kMmaBK) {
        // load tiles: X fp32 -> bf16, W bf16 direct; zero-pad edges
        for (int i = threadIdx.x; i < kMmaBM * kMmaBK; i += kMmaWarps * kWarpSize) {
            int r = i / kMmaBK, c = i % kMmaBK;
            int gr = block_row + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < N) ? __float2bfloat16(X[static_cast<int64_t>(gr) * N + gc])
                                          : __nv_bfloat16(0.f);
        }
        for (int i = threadIdx.x; i < kMmaBN * kMmaBK; i += kMmaWarps * kWarpSize) {
            int r = i / kMmaBK, c = i % kMmaBK;
            int gr = block_col + r, gc = k0 + c;
            Bs[r][c] = (gr < K && gc < N) ? W[static_cast<int64_t>(gr) * N + gc] : __nv_bfloat16(0.f);
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < kMmaBK; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], &As[warp_m + 16 * i][ks], kMmaBK + 8);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                // B fragment reads W's [n][k] rows as col-major [k][n]
                wmma::load_matrix_sync(b[j], &Bs[warp_n + 16 * j][ks], kMmaBK + 8);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // stage through shared so partial edge tiles stay in bounds
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[warp_m + 16 * i][warp_n + 16 * j], acc[i][j], kMmaBN + 4,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kMmaBM * kMmaBN; i += kMmaWarps * kWarpSize) {
        int r = i / kMmaBN, c = i % kMmaBN;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < K)
            Y[static_cast<int64_t>(gr) * K + gc] = Cs[r][c];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Y[rows, out_features] = X[rows, in_features] @ W[out, in]^T on Tensor Cores.
// Requires SM80+; falls back to the SIMT tiled kernel for tiny row counts.
inline void linear_mma_bf16(const float *X, const __nv_bfloat16 *W, float *Y, int rows, int in_features,
                            int out_features, cudaStream_t stream = nullptr) {
    detail::validate_matrix(rows, in_features);
    detail::validate_matrix(out_features, in_features);
    detail::validate_matrix(rows, out_features);
    if (rows == 0 || out_features == 0)
        return;
    detail::require(X && W && Y, "linear_mma_bf16 requires non-null pointers");
    detail::require(in_features > 0, "linear_mma_bf16 requires a positive reduction dim");
    detail::require(detail::blocks_for(rows, detail::kMmaBM) <= 65535, "linear row grid exceeds CUDA grid.y limit");
    launch_async(stream, detail::linear_mma_bf16_kernel,
                 dim3(detail::blocks_for(out_features, detail::kMmaBN),
                      detail::blocks_for(rows, detail::kMmaBM)),
                 dim3(detail::kMmaWarps * kWarpSize), X, W, Y, rows, in_features, out_features);
}

} // namespace opworks::ops
