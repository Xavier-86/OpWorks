#pragma once

#include "builder_common.cuh"

namespace opworks {

enum class TransformOp { transpose, reshape, copy };

class TransformBuilder {
  public:
    TransformBuilder(DeviceSpan<const float> input, DeviceSpan<float> output, int rows, int cols,
                     TransformOp op = TransformOp::copy, cudaStream_t stream = nullptr)
        : input_(input), output_(output), rows_(rows), cols_(cols), op_(op), stream_(stream) {
        detail::validate_matrix(rows, cols);
        detail::require(input.size() >= rows * cols && output.size() >= rows * cols, "transform buffers are too small");
    }
    TransformBuilder &op(TransformOp value) {
        op_ = value;
        return *this;
    }
    DeviceSpan<const float> input() const { return input_; }
    DeviceSpan<float> output() const { return output_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    TransformOp operation() const { return op_; }
    cudaStream_t stream() const { return stream_; }

    template <typename Fn> decltype(auto) apply(Fn &&fn) const { return apply_builder(*this, static_cast<Fn &&>(fn)); }

  private:
    DeviceSpan<const float> input_;
    DeviceSpan<float> output_;
    int rows_, cols_;
    TransformOp op_;
    cudaStream_t stream_;
};

} // namespace opworks
