#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// Tiled GEMM kernel: kMatmulTile x kMatmulTile block tile, kMatmulBlock x
// kMatmulBlock threads, each thread computes a kMatmulSub x kMatmulSub
// sub-tile (shapes come from core/config.cuh). Epilogue is a unary elementwise
// functor applied to every output element.
template <typename Epilogue>
__global__ void mat_mul_kernel(const float *A, const float *B, float *C, int M, int N, int K, Epilogue epilogue) {
    constexpr int kTile = kMatmulTile;
    constexpr int kBlock = kMatmulBlock;
    constexpr int kSub = kMatmulSub;
    __shared__ float tileA[kTile][kBlock + 1]; // +1 avoids bank conflicts
    __shared__ float tileB[kBlock][kTile + 1];

    const int block_row = blockIdx.y * kTile;
    const int block_col = blockIdx.x * kTile;

    float acc[kSub][kSub] = {};

    for (int t = 0; t < N / kBlock + (N % kBlock != 0); ++t) {
#pragma unroll
        OPWORKS_BLOCK_LOOP_FLAT(i, kTile * kBlock) {
            int r = i / kBlock, c = i % kBlock;
            int gr = block_row + r, gc = t * kBlock + c;
            tileA[r][c] = (gr < M && gc < N) ? A[gr * N + gc] : 0.f;
        }
#pragma unroll
        OPWORKS_BLOCK_LOOP_FLAT(i, kBlock * kTile) {
            int r = i / kTile, c = i % kTile;
            int gr = t * kBlock + r, gc = block_col + c;
            tileB[r][c] = (gr < N && gc < K) ? B[gr * K + gc] : 0.f;
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
                C[r * K + c] = epilogue(acc[i][j]);
        }
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Default mat_mul epilogue: identity (plain GEMM, no fused op).
struct PassThrough {
    __device__ float operator()(float x) const { return x; }
};

// C(MxK) = epilogue(A(MxN) @ B(NxK)); the default epilogue is pass-through.
template <typename Epilogue = PassThrough>
inline void mat_mul(const float *A, const float *B, float *C, int M, int N, int K, Epilogue epilogue = {},
                    cudaStream_t stream = nullptr) {
    detail::validate_matrix(M, N);
    detail::validate_matrix(N, K);
    detail::validate_matrix(M, K);
    if (M == 0 || K == 0)
        return;
    detail::require(C && (N == 0 || (A && B)), "matmul requires non-null pointers for nonempty tensors");
    detail::require(detail::blocks_for(M, kMatmulTile) <= 65535, "matmul row grid exceeds CUDA grid.y limit");
    launch_async(stream, detail::mat_mul_kernel<Epilogue>,
                 dim3(detail::blocks_for(K, kMatmulTile), detail::blocks_for(M, kMatmulTile)),
                 dim3(kMatmulBlock, kMatmulBlock), A, B, C, M, N, K, epilogue);
}

} // namespace opworks::ops
