#include <opworks>

#include <unordered_map>

struct Sum {
    static __device__ float identity() { return 0.f; }
    static __device__ float combine(float a, float b) { return a + b; }
};

extern "C" void solve(const float *in, float *out, int n) {
    // This benchmark owns reusable scratch; do not time allocation on every call.
    // Calls complete before returning, so reuse on this host thread is safe.
    static thread_local std::unordered_map<int, opworks::DeviceBuffer> workspaces;
    int device = 0;
    OPWORKS_CUDA_CHECK(cudaGetDevice(&device));
    auto &workspace = workspaces[device];
    const int required = opworks::ops::reduction_workspace_size(n);
    if (workspace.size() < required)
        workspace = opworks::DeviceBuffer(required);
    opworks::ops::reduce<Sum>(in, out, n, workspace.view());
    opworks::synchronize();
}
