#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>
#include <vector>

#include "device_span.cuh"

namespace opworks {

// Owning device buffer for the dtypes the model runtime needs (float32,
// int32, bfloat16 weights). Mirrors DeviceBuffer's API but templated on the
// element type.
template <typename T> class TypedDeviceBuffer {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, __nv_bfloat16>,
                  "TypedDeviceBuffer supports float, int32_t and __nv_bfloat16");

  public:
    TypedDeviceBuffer() = default;
    explicit TypedDeviceBuffer(int64_t n) : n_(n) {
        detail::require(n >= 0, "size must be nonnegative");
        if (n_ != 0)
            OPWORKS_CUDA_CHECK(cudaMalloc(&ptr_, sizeof(T) * n_));
    }

    ~TypedDeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }

    TypedDeviceBuffer(const TypedDeviceBuffer &) = delete;
    TypedDeviceBuffer &operator=(const TypedDeviceBuffer &) = delete;

    TypedDeviceBuffer(TypedDeviceBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    TypedDeviceBuffer &operator=(TypedDeviceBuffer &&o) noexcept {
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
    static TypedDeviceBuffer from_host(const std::vector<T> &h, cudaStream_t stream = nullptr) {
        TypedDeviceBuffer buf(static_cast<int64_t>(h.size()));
        if (!h.empty()) {
            OPWORKS_CUDA_CHECK(
                cudaMemcpyAsync(buf.ptr_, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice, stream));
            synchronize(stream);
        }
        return buf;
    }

    static TypedDeviceBuffer from_host(const T *data, int64_t n, cudaStream_t stream = nullptr) {
        TypedDeviceBuffer buf(n);
        if (n != 0) {
            OPWORKS_CUDA_CHECK(cudaMemcpyAsync(buf.ptr_, data, sizeof(T) * n, cudaMemcpyHostToDevice, stream));
            synchronize(stream);
        }
        return buf;
    }

    std::vector<T> to_host(cudaStream_t stream = nullptr) const {
        std::vector<T> h(n_);
        if (n_ != 0) {
            OPWORKS_CUDA_CHECK(cudaMemcpyAsync(h.data(), ptr_, sizeof(T) * n_, cudaMemcpyDeviceToHost, stream));
            synchronize(stream);
        }
        return h;
    }

    T to_host_scalar(cudaStream_t stream = nullptr) const {
        detail::require(n_ == 1, "scalar copy requires exactly one element");
        return to_host(stream)[0];
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    int64_t size() const { return n_; }

  private:
    T *ptr_ = nullptr;
    int64_t n_ = 0;
};

} // namespace opworks
