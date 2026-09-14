// LeetGPU 113_layer_normalization — custom operator built on
// opworks::block_reduce_all, one block per row
#include <opworks>

using namespace opworks;

struct Sum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};

__global__ void layernorm_kernel(const float* in, float* out,
                                 const float* gamma, const float* beta,
                                 int cols, float eps) {
  const float* row_in = in + blockIdx.x * cols;
  float* row_out = out + blockIdx.x * cols;

  float local_sum = Sum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) { local_sum += row_in[j]; }
  float mean = block_reduce_all<Sum>(local_sum) / cols;

  float local_var = Sum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) {
    float d = row_in[j] - mean;
    local_var += d * d;
  }
  float rstd = rsqrtf(block_reduce_all<Sum>(local_var) / cols + eps);

  OPWORKS_BLOCK_LOOP(j, cols) {
    row_out[j] = (row_in[j] - mean) * rstd * gamma[j] + beta[j];
  }
}

extern "C" void solve(const float* input, const float* weight,
                      const float* bias, float* output, int N, int C,
                      float eps) {
  launch(layernorm_kernel, N, kThreads, input, output, weight, bias, C, eps);
}
