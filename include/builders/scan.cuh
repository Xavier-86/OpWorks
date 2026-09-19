#pragma once

#include "builder_common.cuh"

namespace opworks {

enum class ScanOp { sum, maximum };

class ScanBuilder {
  public:
    ScanBuilder(DeviceSpan<const float> input, DeviceSpan<float> output, int n, ScanOp op = ScanOp::sum,
                bool inclusive = true, cudaStream_t stream = nullptr)
        : input_(input), output_(output), n_(n), op_(op), inclusive_(inclusive), stream_(stream) {
        detail::validate_size(n);
        detail::require(input.size() >= n && output.size() >= n, "scan buffers are too small");
    }
    ScanBuilder &op(ScanOp value) {
        op_ = value;
        return *this;
    }
    ScanBuilder &inclusive(bool value = true) {
        inclusive_ = value;
        return *this;
    }
    DeviceSpan<const float> input() const { return input_; }
    DeviceSpan<float> output() const { return output_; }
    int size() const { return n_; }
    ScanOp operation() const { return op_; }
    bool is_inclusive() const { return inclusive_; }
    cudaStream_t stream() const { return stream_; }

    template <typename Fn> decltype(auto) apply(Fn &&fn) const { return apply_builder(*this, static_cast<Fn &&>(fn)); }

  private:
    DeviceSpan<const float> input_;
    DeviceSpan<float> output_;
    int n_;
    ScanOp op_;
    bool inclusive_;
    cudaStream_t stream_;
};

} // namespace opworks
