#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::ops {

namespace detail {

struct PassThrough {
  __device__ float operator()(float x) const { return x; }
};

}  // namespace detail

// Tiled GEMM skeleton: C(MxK) = epilogue(A(MxN) @ B(NxK)).
// 64x64 block tile, 16x16 threads, each thread computes a 4x4 sub-tile.
// Epilogue is a unary elementwise functor applied to every output element.
template <typename Epilogue>
__global__ void mat_mul_kernel(const float* A, const float* B, float* C, int M,
                               int N, int K, Epilogue epilogue) {
  constexpr int kTile = 64;
  constexpr int kBlock = 16;
  __shared__ float tileA[kTile][kBlock + 1];  // +1 avoids bank conflicts
  __shared__ float tileB[kBlock][kTile + 1];

  const int tid = threadIdx.x + threadIdx.y * kBlock;
  const int block_row = blockIdx.y * kTile;
  const int block_col = blockIdx.x * kTile;

  float acc[4][4] = {};

  for (int t = 0; t < (N + kBlock - 1) / kBlock; ++t) {
#pragma unroll
    for (int i = tid; i < kTile * kBlock; i += kBlock * kBlock) {
      int r = i / kBlock, c = i % kBlock;
      int gr = block_row + r, gc = t * kBlock + c;
      tileA[r][c] = (gr < M && gc < N) ? A[gr * N + gc] : 0.f;
    }
#pragma unroll
    for (int i = tid; i < kBlock * kTile; i += kBlock * kBlock) {
      int r = i / kTile, c = i % kTile;
      int gr = t * kBlock + r, gc = block_col + c;
      tileB[r][c] = (gr < N && gc < K) ? B[gr * K + gc] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kBlock; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = tileA[threadIdx.y * 4 + i][k];
        b[i] = tileB[k][threadIdx.x * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r = block_row + threadIdx.y * 4 + i;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = block_col + threadIdx.x * 4 + j;
      if (r < M && c < K) C[r * K + c] = epilogue(acc[i][j]);
    }
  }
}

// C(MxK) = epilogue(A(MxN) @ B(NxK)); the default epilogue is pass-through.
template <typename Epilogue = detail::PassThrough>
inline void mat_mul(const float* A, const float* B, float* C, int M, int N,
                    int K, Epilogue epilogue = {}) {
  launch(mat_mul_kernel<Epilogue>, dim3((K + 63) / 64, (M + 63) / 64),
         dim3(16, 16), A, B, C, M, N, K, epilogue);
}

}  // namespace opworks::ops
