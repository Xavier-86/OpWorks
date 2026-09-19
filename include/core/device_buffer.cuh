#pragma once

#include <vector>

#include "device_span.cuh"

namespace opworks {

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(int n) : n_(n) {
        detail::validate_size(n);
        if (n != 0)
            OPWORKS_CUDA_CHECK(cudaMalloc(&ptr_, sizeof(float) * n_));
    }

    ~DeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            if (ptr_)
                cudaFree(ptr_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    // Host copies complete before returning, including on non-default streams.
    static DeviceBuffer from_host(const std::vector<float> &h, cudaStream_t stream = nullptr) {
        detail::require(h.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                        "buffer exceeds the supported INT_MAX elements");
        DeviceBuffer buf(static_cast<int>(h.size()));
        if (!h.empty()) {
            OPWORKS_CUDA_CHECK(
                cudaMemcpyAsync(buf.ptr_, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice, stream));
            synchronize(stream);
        }
        return buf;
    }

    std::vector<float> to_host(cudaStream_t stream = nullptr) const {
        std::vector<float> h(n_);
        if (n_ != 0) {
            OPWORKS_CUDA_CHECK(cudaMemcpyAsync(h.data(), ptr_, sizeof(float) * n_, cudaMemcpyDeviceToHost, stream));
            synchronize(stream);
        }
        return h;
    }

    float to_host_scalar(cudaStream_t stream = nullptr) const {
        detail::require(n_ == 1, "scalar copy requires exactly one element");
        return to_host(stream)[0];
    }

    DeviceSpan<float> view() & { return {ptr_, n_}; }
    DeviceSpan<const float> view() const & { return {ptr_, n_}; }
    DeviceSpan<float> view() && = delete;
    DeviceSpan<const float> view() const && = delete;

    float *data() { return ptr_; }
    const float *data() const { return ptr_; }
    int size() const { return n_; }

  private:
    float *ptr_ = nullptr;
    int n_ = 0;
};

} // namespace opworks
