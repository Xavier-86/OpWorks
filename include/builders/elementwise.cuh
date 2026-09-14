#pragma once

#include "../container/device_buffer.cuh"
#include "../ops/elementwise.cuh"

namespace opworks {

class ElementwiseBuilder {
 public:
  // binary: out[i] = op(a[i], b[i])
  ElementwiseBuilder(const DeviceBuffer& a, const DeviceBuffer& b) : a_(a.data()), b_(b.data()), n_(a.size()) {}
  // unary: out[i] = op(in[i])
  explicit ElementwiseBuilder(const DeviceBuffer& a) : a_(a.data()), n_(a.size()) {}

  // op may carry runtime state (e.g. ScaleAdd{alpha}); it is copied to the
  // kernel by value. Stateless functors can keep calling apply<Op>().
  template <typename Op>
  DeviceBuffer apply(const Op& op = Op{}) {
    DeviceBuffer out(n_);
    if constexpr (Op::kArity == 2) {
      ops::map(a_, b_, out.data(), n_, op);
    } else {
      ops::map(a_, out.data(), n_, op);
    }
    return out;
  }

 private:
  const float* a_ = nullptr;
  const float* b_ = nullptr;
  int n_ = 0;
};

}  // namespace opworks
