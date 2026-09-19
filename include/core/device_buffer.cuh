#pragma once

#include <vector>

#include "cuda_utils.cuh"

namespace opworks {

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(int n) : n_(n) {
        OPWORKS_CUDA_CHECK(cudaMalloc(&ptr_, sizeof(float) * n_));
    }

    ~DeviceBuffer() {
        if (ptr_ && owned_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_), owned_(o.owned_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
        o.owned_ = false;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_ && owned_) cudaFree(ptr_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            owned_ = o.owned_;
            o.ptr_ = nullptr;
            o.n_ = 0;
            o.owned_ = false;
        }
        return *this;
    }

    static DeviceBuffer from_host(const std::vector<float>& h) {
        DeviceBuffer buf(static_cast<int>(h.size()));
        OPWORKS_CUDA_CHECK(cudaMemcpy(buf.ptr_, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice));
        return buf;
    }

    // Non-owning view of external device memory; not freed on destruction.
    static DeviceBuffer wrap(const float* p, int n) {
        DeviceBuffer buf;
        buf.ptr_ = const_cast<float*>(p);
        buf.n_ = n;
        buf.owned_ = false;
        return buf;
    }

    std::vector<float> to_host() const {
        std::vector<float> h(n_);
        OPWORKS_CUDA_CHECK(cudaMemcpy(h.data(), ptr_, sizeof(float) * n_, cudaMemcpyDeviceToHost));
        return h;
    }

    float to_host_scalar() const { return to_host()[0]; }

    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    int size() const { return n_; }

  private:
    float* ptr_ = nullptr;
    int n_ = 0;
    bool owned_ = true;
};

}  // namespace opworks
