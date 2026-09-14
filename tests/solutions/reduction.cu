// LeetGPU 4_reduction — exercises ReductionBuilder
#include <opworks>

using namespace opworks;

struct Sum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};

extern "C" void solve(const float* input, float* output, int N) {
  auto in = DeviceBuffer::wrap(input, N);
  auto out = ReductionBuilder(in).apply<Sum>();
  OPWORKS_CUDA_CHECK(
      cudaMemcpy(output, out.data(), sizeof(float), cudaMemcpyDeviceToDevice));
}
