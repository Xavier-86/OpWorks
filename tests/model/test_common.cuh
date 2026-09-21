#pragma once

// Shared helpers for model tests: fixture loading and FP32 comparison against
// the plan's thresholds (abs_err <= 1e-5 + 1e-4*|ref| for base kernels;
// NRMSE / cosine for wide reductions and logits).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../include/core/typed_device_buffer.cuh"
#include "../include/runtime/pack_reader.cuh"

namespace modeltest {

struct Stats {
    double max_abs_err = 0.0;
    double nrmse = 0.0;
    double cosine = 0.0;
    int64_t worst_index = -1;
};

inline Stats compare(const std::vector<float> &actual, const std::vector<float> &ref) {
    Stats s;
    if (actual.size() != ref.size()) {
        std::fprintf(stderr, "size mismatch: %zu vs %zu\n", actual.size(), ref.size());
        s.max_abs_err = INFINITY;
        return s;
    }
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = actual[i], b = ref[i];
        double e = std::fabs(a - b);
        if (e > s.max_abs_err) {
            s.max_abs_err = e;
            s.worst_index = static_cast<int64_t>(i);
        }
        num += e * e;
        den += b * b;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    double rms_ref = std::sqrt(den / ref.size());
    s.nrmse = std::sqrt(num / ref.size()) / std::max(rms_ref, 1e-6);
    s.cosine = dot / std::max(std::sqrt(na) * std::sqrt(nb), 1e-30);
    return s;
}

// Per-element plan threshold: abs_err <= 1e-5 + 1e-4*|ref|.
inline bool pass_elementwise(const std::vector<float> &actual, const std::vector<float> &ref, Stats &s) {
    s = compare(actual, ref);
    if (s.worst_index >= 0) {
        double ref_v = ref[static_cast<size_t>(s.worst_index)];
        return s.max_abs_err <= 1e-5 + 1e-4 * std::fabs(ref_v);
    }
    return true;
}

inline bool pass_logits(const Stats &s) { return s.nrmse <= 1e-3 && s.cosine >= 0.9999; }

inline void report(const char *name, bool ok, const Stats &s) {
    std::printf("%-28s %s  max_abs=%.3g nrmse=%.3g cos=%.8f (worst@%lld)\n", name, ok ? "PASS" : "FAIL",
                s.max_abs_err, s.nrmse, s.cosine, static_cast<long long>(s.worst_index));
}

inline opworks::TypedDeviceBuffer<float> upload(const std::vector<float> &h) {
    return opworks::TypedDeviceBuffer<float>::from_host(h);
}

struct FixtureTensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

inline FixtureTensor fixture(const opworks::PackReader &pack, const std::string &name) {
    return {pack.read(name), pack.meta(name).shape};
}

inline std::vector<int32_t> fixture_ids(const opworks::PackReader &pack, const std::string &name) {
    return pack.read_int32(name);
}

} // namespace modeltest
