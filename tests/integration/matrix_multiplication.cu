#include <opworks>

extern "C" void solve(const float *a, const float *b, float *out, int m, int n, int k) {
    opworks::ops::mat_mul(a, b, out, m, n, k);
    opworks::synchronize();
}
