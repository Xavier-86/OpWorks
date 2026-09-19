#pragma once

#include "../core/device_span.cuh"

namespace opworks {

template <typename Builder, typename Fn> decltype(auto) apply_builder(const Builder &builder, Fn &&fn) {
    return static_cast<Fn &&>(fn)(builder);
}

} // namespace opworks
