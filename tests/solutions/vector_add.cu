// LeetGPU 1_vector_add — exercises ElementwiseBuilder binary path
#include <opworks>

using namespace opworks;

struct Add {
  static constexpr int kArity = 2;
  __device__ float operator()(float a, float b) const { return a + b; }
};

extern "C" void solve(const float* A, const float* B, float* C, size_t N) {
  auto a = DeviceBuffer::wrap(A, (int)N);
  auto b = DeviceBuffer::wrap(B, (int)N);
  auto out = ElementwiseBuilder(a, b).apply<Add>();
  OPWORKS_CUDA_CHECK(
      cudaMemcpy(C, out.data(), N * sizeof(float), cudaMemcpyDeviceToDevice));
}
