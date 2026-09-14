// LeetGPU 21_relu — exercises ElementwiseBuilder unary path
#include <opworks>

using namespace opworks;

struct Relu {
  static constexpr int kArity = 1;
  __device__ float operator()(float x) const { return fmaxf(x, 0.f); }
};

extern "C" void solve(const float* input, float* output, int N) {
  auto in = DeviceBuffer::wrap(input, N);
  auto out = ElementwiseBuilder(in).apply<Relu>();
  OPWORKS_CUDA_CHECK(cudaMemcpy(output, out.data(), N * sizeof(float),
                                cudaMemcpyDeviceToDevice));
}
