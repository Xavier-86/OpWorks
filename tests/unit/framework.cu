#include <opworks>

#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace opworks;

void second_translation_unit();

struct Add {
    static constexpr int kArity = 2;
    __device__ float operator()(float a, float b) const { return a + b; }
};
struct Scale {
    static constexpr int kArity = 1;
    float factor = 2;
    __device__ float operator()(float x) const { return factor * x; }
};
struct Sum {
    static __device__ float identity() { return 0; }
    static __device__ float combine(float a, float b) { return a + b; }
};
struct Product {
    static __device__ float identity() { return 1; }
    static __device__ float combine(float a, float b) { return a * b; }
};

static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
static_assert(std::is_convertible_v<DeviceSpan<float>, DeviceSpan<const float>>);
static_assert(!std::is_convertible_v<DeviceSpan<const float>, DeviceSpan<float>>);
static_assert(std::is_same_v<decltype(std::declval<const DeviceBuffer &>().view().data()), const float *>);
static_assert(!std::is_constructible_v<ElementwiseBuilder, DeviceBuffer &&>);
static_assert(!std::is_constructible_v<ElementwiseBuilder, DeviceBuffer &, DeviceBuffer &&>);
static_assert(!std::is_constructible_v<ElementwiseBuilder, DeviceBuffer &&, DeviceBuffer &>);
static_assert(!std::is_constructible_v<ReductionBuilder, DeviceBuffer &&>);

void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}

void close(float actual, float expected, float tolerance = 1e-5f) {
    check(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance * (1 + std::fabs(expected)),
          "numerical mismatch");
}

template <typename F> void invalid(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("expected invalid_argument");
}

void buffers_and_contracts() {
    DeviceBuffer empty(0);
    check(empty.data() == nullptr && empty.to_host().empty(), "empty buffer");
    check(DeviceBuffer::from_host({}).size() == 0, "empty host copy");
    invalid([] { DeviceBuffer bad(-1); });
    invalid([] { DeviceSpan<float> bad(nullptr, 1); });
    invalid([&] { empty.to_host_scalar(); });
    auto a = DeviceBuffer::from_host({1, 2, 3});
    auto b = DeviceBuffer::from_host({4, 5, 6});
    auto short_input = DeviceBuffer::from_host({1});
    invalid([&] { ElementwiseBuilder bad(a, short_input); });
    invalid([&] { ElementwiseBuilder(a).apply<Add>(); });
    invalid([&] { ElementwiseBuilder(a, b).apply<Scale>(); });
    invalid([&] { a.to_host_scalar(); });
    auto sum = ElementwiseBuilder(a, b).apply<Add>().to_host();
    for (int i = 0; i < 3; ++i)
        close(sum[i], 5.f + 2 * i);
    auto scaled = ElementwiseBuilder(a.view()).apply(Scale{3}).to_host();
    for (int i = 0; i < 3; ++i)
        close(scaled[i], 3.f * (i + 1));
    close(ReductionBuilder(a).apply<Sum>().to_host_scalar(), 6);
    close(ReductionBuilder(empty).apply<Product>().to_host_scalar(), 1);
    check(ElementwiseBuilder(empty).apply<Scale>().size() == 0, "empty builder");

    const float *pointer = a.data();
    auto view = a.view();
    DeviceBuffer moved(std::move(a));
    check(a.data() == nullptr && a.size() == 0 && moved.data() == pointer, "move constructor");
    b = std::move(moved);
    check(moved.data() == nullptr && b.data() == view.data(), "move assignment preserves borrowed address");
    close(b.to_host()[2], 3);

    ops::map(nullptr, nullptr, 0, Scale{});
    ops::softmax(nullptr, nullptr, 0, 3);
    ops::layer_norm(nullptr, nullptr, nullptr, nullptr, 0, 3);
    ops::mat_mul(nullptr, nullptr, nullptr, 0, 4, 5);
    invalid([] { ops::map(nullptr, nullptr, -1, Scale{}); });
    invalid([] { ops::map(nullptr, nullptr, 1, Scale{}); });
    invalid([] { ops::reduce<Sum>(nullptr, nullptr, 0); });
    invalid([] { ops::softmax(nullptr, nullptr, 1, 0); });
    invalid([] { ops::softmax(nullptr, nullptr, -1, 2); });
    invalid([] { ops::layer_norm(nullptr, nullptr, nullptr, nullptr, 1, 0); });
    invalid([] { ops::layer_norm(nullptr, nullptr, nullptr, nullptr, 0, 3, 0); });
    invalid([] { ops::layer_norm(nullptr, nullptr, nullptr, nullptr, 0, 3, INFINITY); });
    invalid([] { ops::mat_mul(nullptr, nullptr, nullptr, 2, 2, 2); });
    invalid([] { detail::validate_matrix(65536, 65536); });
    check(detail::blocks_for(std::numeric_limits<int>::max(), 256) == 8388608, "overflow-safe ceil divide");
    check(num_blocks_for(0) == 0, "empty grid calculation");
}

void elementwise_boundaries() {
    for (int n : {1, 2, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1025}) {
        for (int offset : {0, 1}) {
            std::vector<float> host(n + offset);
            for (int i = 0; i < n + offset; ++i)
                host[i] = static_cast<float>(i - 10);
            auto a = DeviceBuffer::from_host(host);
            auto b = DeviceBuffer::from_host(host);
            DeviceBuffer out(n + offset);
            ops::map(a.data() + offset, b.data() + offset, out.data() + offset, n, Add{});
            auto actual = out.to_host();
            for (int i = 0; i < n; ++i)
                close(actual[i + offset], 2 * host[i + offset]);
            // Exact in-place use is supported on both aligned and scalar paths.
            ops::map(a.data() + offset, a.data() + offset, n, Scale{3});
            actual = a.to_host();
            for (int i = 0; i < n; ++i)
                close(actual[i + offset], 3 * host[i + offset]);
        }
    }
}

__global__ void set_value(float *out, float value) {
    out[0] = value;
}

void streams_and_reduction() {
    cudaStream_t stream;
    OPWORKS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto input = DeviceBuffer::from_host(std::vector<float>(4099, 1.f), stream);
    DeviceBuffer output(1);
    DeviceBuffer workspace(ops::reduction_workspace_size(input.size()));
    invalid([&] { ops::reduce<Sum>(input.data(), output.data(), input.size(), DeviceSpan<float>{}, stream); });
    invalid([&] { ops::reduce<Sum>(input.data(), output.data(), -1, stream); });

    for (int n : {0, 1, 31, 32, 33, 255, 256, 257, 4099}) {
        ops::reduce<Sum>(input.data(), output.data(), n, workspace.view(), stream);
        close(output.to_host_scalar(stream), static_cast<float>(n));
        ops::reduce<Sum>(input.data(), output.data(), n, stream);
        close(output.to_host_scalar(stream), static_cast<float>(n));
    }
    close(ReductionBuilder(input).apply<Sum>(workspace.view(), stream).to_host_scalar(stream), 4099);
    auto scaled = ElementwiseBuilder(input).apply(Scale{3}, stream);
    close(scaled.to_host(stream).back(), 3);

    // Capture rejects hidden synchronization. Replay also checks stream ordering
    // and reuse of scratch without allocating on the hot path.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    OPWORKS_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_async(stream, set_value, 1, 1, output.data(), -1.f);
    ops::map(input.data(), input.data(), input.size(), Scale{2}, stream);
    ops::reduce<Sum>(input.data(), output.data(), input.size(), workspace.view(), stream);
    OPWORKS_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    OPWORKS_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    OPWORKS_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    close(output.to_host_scalar(stream), 8198);
    OPWORKS_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    close(output.to_host_scalar(stream), 16396);
    OPWORKS_CUDA_CHECK(cudaGraphExecDestroy(executable));
    OPWORKS_CUDA_CHECK(cudaGraphDestroy(graph));

    launch_sync(stream, set_value, 1, 1, output.data(), 7.f);
    OPWORKS_CUDA_CHECK(cudaStreamQuery(stream));
    close(output.to_host_scalar(stream), 7);
    launch(set_value, 1, 1, output.data(), 8.f);
    synchronize();
    close(output.to_host_scalar(), 8);
    launch_sync(set_value, 1, 1, output.data(), 9.f);
    close(output.to_host_scalar(), 9);
    OPWORKS_CUDA_CHECK(cudaStreamDestroy(stream));
}

void row_ops_and_matmul() {
    cudaStream_t stream;
    OPWORKS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto input = DeviceBuffer::from_host({-2, -1, 0, 1, 2, 3, 4, 5, 6, 7});
    auto gamma = DeviceBuffer::from_host(std::vector<float>(5, 2));
    auto beta = DeviceBuffer::from_host(std::vector<float>(5, 1));
    DeviceBuffer output(10);
    ops::softmax(input.data(), output.data(), 2, 5, stream);
    auto actual = output.to_host(stream);
    float denominator = 0;
    for (int j = 0; j < 5; ++j)
        denominator += std::exp(static_cast<float>(j - 4));
    for (int i = 0; i < 10; ++i)
        close(actual[i], std::exp(static_cast<float>(i % 5 - 4)) / denominator);
    ops::layer_norm(input.data(), output.data(), gamma.data(), beta.data(), 2, 5, 1e-5f, stream);
    actual = output.to_host(stream);
    for (int i = 0; i < 10; ++i)
        close(actual[i], (i % 5 - 2) * 2.f / std::sqrt(2.f + 1e-5f) + 1);

    std::vector<float> host_a(15), host_b(35);
    for (int i = 0; i < 15; ++i)
        host_a[i] = static_cast<float>(i % 7 - 3);
    for (int i = 0; i < 35; ++i)
        host_b[i] = static_cast<float>(i % 5 - 2);
    auto a = DeviceBuffer::from_host(host_a);
    auto b = DeviceBuffer::from_host(host_b);
    DeviceBuffer c(21);
    ops::mat_mul(a.data(), b.data(), c.data(), 3, 5, 7, Scale{2}, stream);
    actual = c.to_host(stream);
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 7; ++col) {
            float expected = 0;
            for (int k = 0; k < 5; ++k)
                expected += host_a[r * 5 + k] * host_b[k * 7 + col];
            close(actual[r * 7 + col], 2 * expected);
        }
    }
    ops::mat_mul(nullptr, nullptr, c.data(), 3, 0, 7, ops::PassThrough{}, stream);
    for (float value : c.to_host(stream))
        close(value, 0);
    OPWORKS_CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    OPWORKS_CUDA_CHECK(status);
    try {
        second_translation_unit();
        buffers_and_contracts();
        elementwise_boundaries();
        streams_and_reduction();
        row_ops_and_matmul();
        std::cout
            << "PASS: contracts, ownership, boundaries, streams, graph replay, reduction, row ops, matmul, multi-TU\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
