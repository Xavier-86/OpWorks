#pragma once

#include "../core/device_buffer.cuh"
#include "../ops/reduction.cuh"

namespace opworks {

class ReductionBuilder {
  public:
    explicit ReductionBuilder(DeviceSpan<const float> in) : in_(in) {}
    explicit ReductionBuilder(const DeviceBuffer &in) : ReductionBuilder(in.view()) {}
    ReductionBuilder(const DeviceBuffer &&) = delete;

    template <typename Op> DeviceBuffer apply(cudaStream_t stream = nullptr) const {
        DeviceBuffer out(1);
        ops::reduce<Op>(in_.data(), out.data(), in_.size(), stream);
        return out;
    }

    template <typename Op> DeviceBuffer apply(DeviceSpan<float> workspace, cudaStream_t stream = nullptr) const {
        DeviceBuffer out(1);
        ops::reduce<Op>(in_.data(), out.data(), in_.size(), workspace, stream);
        return out;
    }

  private:
    DeviceSpan<const float> in_;
};

} // namespace opworks
