#pragma once

#include "builder_common.cuh"

namespace opworks {

enum class MatmulActivation { none, relu, gelu };

class MatmulBuilder {
  public:
    MatmulBuilder(DeviceSpan<const float> a, DeviceSpan<const float> b, DeviceSpan<float> output, int m, int n, int k,
                  cudaStream_t stream = nullptr)
        : a_(a), b_(b), output_(output), m_(m), n_(n), k_(k), stream_(stream) {
        detail::validate_matrix(m, n);
        detail::validate_matrix(n, k);
        detail::validate_matrix(m, k);
        detail::require(a.size() >= m * n && b.size() >= n * k && output.size() >= m * k, "matmul buffer is too small");
    }
    MatmulBuilder &transpose_a(bool value = true) {
        transpose_a_ = value;
        return *this;
    }
    MatmulBuilder &transpose_b(bool value = true) {
        transpose_b_ = value;
        return *this;
    }
    MatmulBuilder &bias(DeviceSpan<const float> value) {
        bias_ = value;
        return *this;
    }
    MatmulBuilder &activation(MatmulActivation value) {
        activation_ = value;
        return *this;
    }
    bool transposed_a() const { return transpose_a_; }
    bool transposed_b() const { return transpose_b_; }
    DeviceSpan<const float> a() const { return a_; }
    DeviceSpan<const float> b() const { return b_; }
    DeviceSpan<float> output() const { return output_; }
    DeviceSpan<const float> bias() const { return bias_; }
    MatmulActivation activation() const { return activation_; }
    int m() const { return m_; }
    int n() const { return n_; }
    int k() const { return k_; }
    cudaStream_t stream() const { return stream_; }

    template <typename Fn> decltype(auto) apply(Fn &&fn) const { return apply_builder(*this, static_cast<Fn &&>(fn)); }

  private:
    DeviceSpan<const float> a_, b_;
    DeviceSpan<float> output_;
    DeviceSpan<const float> bias_;
    int m_, n_, k_;
    bool transpose_a_ = false, transpose_b_ = false;
    MatmulActivation activation_ = MatmulActivation::none;
    cudaStream_t stream_;
};

} // namespace opworks
