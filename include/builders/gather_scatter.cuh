#pragma once

#include "builder_common.cuh"

namespace opworks {

enum class GatherScatterOp { gather, scatter };

class GatherScatterBuilder {
  public:
    GatherScatterBuilder(DeviceSpan<const float> input, DeviceSpan<float> output, DeviceSpan<const int> indices,
                         int count, GatherScatterOp op, cudaStream_t stream = nullptr)
        : input_(input), output_(output), indices_(indices), count_(count), op_(op), stream_(stream) {
        detail::validate_size(count);
        detail::require(indices.size() >= count, "index buffer is too small");
        detail::require(output.size() >= count, "gather/scatter output is too small");
    }
    DeviceSpan<const float> input() const { return input_; }
    DeviceSpan<float> output() const { return output_; }
    DeviceSpan<const int> indices() const { return indices_; }
    int count() const { return count_; }
    GatherScatterOp operation() const { return op_; }
    cudaStream_t stream() const { return stream_; }

    template <typename Fn> decltype(auto) apply(Fn &&fn) const { return apply_builder(*this, static_cast<Fn &&>(fn)); }

  private:
    DeviceSpan<const float> input_;
    DeviceSpan<float> output_;
    DeviceSpan<const int> indices_;
    int count_;
    GatherScatterOp op_;
    cudaStream_t stream_;
};

} // namespace opworks
