#include <opworks>

struct Add {
    __device__ float operator()(float a, float b) const { return a + b; }
};

extern "C" void solve(const float *a, const float *b, float *out, size_t n) {
    opworks::detail::require(n <= static_cast<size_t>(std::numeric_limits<int>::max()), "input too large");
    opworks::ops::map(a, b, out, static_cast<int>(n), Add{});
    opworks::synchronize();
}
