#include <opworks>

struct Relu {
    __device__ float operator()(float x) const { return fmaxf(x, 0.f); }
};

extern "C" void solve(const float *in, float *out, int n) {
    opworks::ops::map(in, out, n, Relu{});
    opworks::synchronize();
}
