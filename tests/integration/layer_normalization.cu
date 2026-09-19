#include <opworks>

extern "C" void solve(const float *in, const float *gamma, const float *beta, float *out, int rows, int cols,
                      float eps) {
    opworks::ops::layer_norm(in, out, gamma, beta, rows, cols, eps);
    opworks::synchronize();
}
