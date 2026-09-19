#include <opworks>

extern "C" void solve(const float *in, float *out, int n) {
    opworks::ops::softmax(in, out, 1, n);
    opworks::synchronize();
}
