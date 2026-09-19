#pragma once

#include "../core/device_buffer.cuh"
#include "../ops/elementwise.cuh"

namespace opworks {

class ElementwiseBuilder {
  public:
    // binary: out[i] = op(a[i], b[i])
    ElementwiseBuilder(DeviceSpan<const float> a, DeviceSpan<const float> b) : a_(a), b_(b), arity_(2) {
        detail::require(a.size() == b.size(), "elementwise inputs must have equal lengths");
    }
    ElementwiseBuilder(const DeviceBuffer &a, const DeviceBuffer &b) : ElementwiseBuilder(a.view(), b.view()) {}
    // unary: out[i] = op(in[i])
    explicit ElementwiseBuilder(DeviceSpan<const float> a) : a_(a) {}
    explicit ElementwiseBuilder(const DeviceBuffer &a) : ElementwiseBuilder(a.view()) {}

    // A retained builder borrows its inputs. Reject immediately dangling owners.
    ElementwiseBuilder(const DeviceBuffer &&) = delete;
    ElementwiseBuilder(const DeviceBuffer &&, const DeviceBuffer &) = delete;
    ElementwiseBuilder(const DeviceBuffer &, const DeviceBuffer &&) = delete;
    ElementwiseBuilder(const DeviceBuffer &&, const DeviceBuffer &&) = delete;

    // op may carry runtime state (e.g. ScaleAdd{alpha}); it is copied to the
    // kernel by value. Stateless functors can keep calling apply<Op>().
    template <typename Op> DeviceBuffer apply(const Op &op = Op{}, cudaStream_t stream = nullptr) const {
        static_assert(Op::kArity == 1 || Op::kArity == 2, "elementwise arity must be 1 or 2");
        detail::require(arity_ == Op::kArity, "functor arity does not match builder inputs");
        DeviceBuffer out(a_.size());
        if constexpr (Op::kArity == 2) {
            ops::map(a_.data(), b_.data(), out.data(), a_.size(), op, stream);
        } else {
            ops::map(a_.data(), out.data(), a_.size(), op, stream);
        }
        return out;
    }

  private:
    DeviceSpan<const float> a_;
    DeviceSpan<const float> b_;
    int arity_ = 1;
};

} // namespace opworks
