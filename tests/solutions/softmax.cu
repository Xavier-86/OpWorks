// LeetGPU 5_softmax — custom operator built on opworks::block_reduce_all
#include <opworks>

using namespace opworks;

struct Sum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};
struct Max {
  static __device__ float identity() { return -INFINITY; }
  static __device__ float combine(float a, float b) { return fmaxf(a, b); }
};

__global__ void softmax_kernel(const float* in, float* out, int n) {
  // whole vector is a single row -> one block
  float local_max = Max::identity();
  OPWORKS_BLOCK_LOOP(j, n) { local_max = Max::combine(local_max, in[j]); }
  float max_val = block_reduce_all<Max>(local_max);

  float local_sum = Sum::identity();
  OPWORKS_BLOCK_LOOP(j, n) { local_sum += expf(in[j] - max_val); }
  float sum = block_reduce_all<Sum>(local_sum);

  OPWORKS_BLOCK_LOOP(j, n) { out[j] = expf(in[j] - max_val) / sum; }
}

extern "C" void solve(const float* input, float* output, int N) {
  launch(softmax_kernel, 1, kThreads, input, output, N);
}
