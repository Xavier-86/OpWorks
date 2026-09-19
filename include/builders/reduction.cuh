#pragma once

#include "../core/device_buffer.cuh"
#include "../ops/reduction.cuh"

namespace opworks {

class ReductionBuilder {
  public:
    explicit ReductionBuilder(const DeviceBuffer& in) : in_(in.data()), n_(in.size()) {}

    template <typename Op>
    DeviceBuffer apply() {
        DeviceBuffer out(1);
        ops::reduce<Op>(in_, out.data(), n_);
        return out;
    }

  private:
    const float* in_ = nullptr;
    int n_ = 0;
};

}  // namespace opworks
