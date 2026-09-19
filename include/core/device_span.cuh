#pragma once

#include <type_traits>

#include "cuda_utils.cuh"

namespace opworks {

// Borrowed device memory. The owner must outlive every queued use of this view.
template <typename T> class DeviceSpan {
    static_assert(std::is_same_v<std::remove_const_t<T>, float>, "only float32 is supported");

  public:
    DeviceSpan() = default;
    DeviceSpan(T *data, int size) : data_(data), size_(size) {
        detail::validate_size(size);
        detail::require(size == 0 || data != nullptr, "nonempty span requires a pointer");
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
    DeviceSpan(DeviceSpan<U> other) : data_(other.data()), size_(other.size()) {}

    T *data() const { return data_; }
    int size() const { return size_; }

  private:
    T *data_ = nullptr;
    int size_ = 0;
};

} // namespace opworks
