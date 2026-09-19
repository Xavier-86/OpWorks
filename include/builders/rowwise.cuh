#pragma once

#include <cmath>

#include "builder_common.cuh"

namespace opworks {

enum class RowwiseOp { softmax, layer_norm, rms_norm };

class RowwiseBuilder {
  public:
    static RowwiseBuilder softmax(DeviceSpan<const float> input, DeviceSpan<float> output, int rows, int cols,
                                  cudaStream_t stream = nullptr) {
        return RowwiseBuilder(RowwiseOp::softmax, input, output, nullptr, nullptr, rows, cols, kLayerNormEps, stream);
    }
    static RowwiseBuilder layer_norm(DeviceSpan<const float> input, DeviceSpan<float> output,
                                     DeviceSpan<const float> gamma, DeviceSpan<const float> beta, int rows, int cols,
                                     float eps = kLayerNormEps, cudaStream_t stream = nullptr) {
        return RowwiseBuilder(RowwiseOp::layer_norm, input, output, gamma.data(), beta.data(), rows, cols, eps, stream);
    }
    static RowwiseBuilder rms_norm(DeviceSpan<const float> input, DeviceSpan<float> output,
                                   DeviceSpan<const float> weight, int rows, int cols, float eps = kLayerNormEps,
                                   cudaStream_t stream = nullptr) {
        return RowwiseBuilder(RowwiseOp::rms_norm, input, output, weight.data(), nullptr, rows, cols, eps, stream);
    }

    RowwiseOp op() const { return op_; }
    DeviceSpan<const float> input() const { return input_; }
    DeviceSpan<float> output() const { return output_; }
    const float *parameter() const { return parameter_; }
    const float *beta() const { return beta_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    float epsilon() const { return eps_; }
    cudaStream_t stream() const { return stream_; }

    template <typename Fn> decltype(auto) apply(Fn &&fn) const { return apply_builder(*this, static_cast<Fn &&>(fn)); }

  private:
    RowwiseBuilder(RowwiseOp op, DeviceSpan<const float> input, DeviceSpan<float> output, const float *parameter,
                   const float *beta, int rows, int cols, float eps, cudaStream_t stream)
        : op_(op), input_(input), output_(output), parameter_(parameter), beta_(beta), rows_(rows), cols_(cols),
          eps_(eps), stream_(stream) {
        detail::validate_matrix(rows, cols);
        detail::require(input.size() >= rows * cols && output.size() >= rows * cols, "rowwise buffer is too small");
        detail::require(rows == 0 || cols > 0, "rowwise columns must be positive");
        detail::require(std::isfinite(eps) && eps > 0, "rowwise epsilon must be finite and positive");
        if (op == RowwiseOp::layer_norm || op == RowwiseOp::rms_norm)
            detail::require(parameter != nullptr || cols == 0, "rowwise parameter is required");
        if (op == RowwiseOp::layer_norm)
            detail::require(beta != nullptr || cols == 0, "layer norm beta is required");
    }

    RowwiseOp op_;
    DeviceSpan<const float> input_;
    DeviceSpan<float> output_;
    const float *parameter_;
    const float *beta_;
    int rows_, cols_;
    float eps_;
    cudaStream_t stream_;
};

} // namespace opworks
