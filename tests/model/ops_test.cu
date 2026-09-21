// Operator-level tests against tests/fixtures/llama/ops_fixtures.pack.
// Usage: ops_test [ops_fixtures.pack]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../include/ops/argmax.cuh"
#include "../include/ops/attention_decode.cuh"
#include "../include/ops/attention_prefill.cuh"
#include "../include/ops/causal_scale.cuh"
#include "../include/ops/embedding.cuh"
#include "../include/ops/linear.cuh"
#include "../include/ops/linear_mma.cuh"
#include "../include/ops/matmul.cuh"
#include "../include/ops/rms_norm.cuh"
#include "../include/ops/rope.cuh"
#include "../include/ops/softmax.cuh"
#include "../include/ops/split_heads.cuh"
#include "../include/ops/swiglu.cuh"
#include "test_common.cuh"

using namespace opworks;
using namespace modeltest;

namespace {

int g_failures = 0;

void check(const char *name, const std::vector<float> &actual, const std::vector<float> &ref) {
    Stats s;
    bool ok = pass_elementwise(actual, ref, s);
    report(name, ok, s);
    if (!ok)
        ++g_failures;
}

void test_rms_norm(const PackReader &p) {
    auto in = fixture(p, "rms_norm.input");
    auto w = fixture(p, "rms_norm.weight");
    auto ref = fixture(p, "rms_norm.output");
    int rows = static_cast<int>(in.shape[0]), cols = static_cast<int>(in.shape[1]);
    auto d_in = upload(in.data);
    auto d_w = upload(w.data);
    TypedDeviceBuffer<float> out(static_cast<int64_t>(rows) * cols);
    ops::rms_norm(d_in.data(), d_w.data(), out.data(), rows, cols, 1e-5f);
    check("rms_norm", out.to_host(), ref.data);
}

void test_linear(const PackReader &p) {
    auto in = fixture(p, "linear.input");
    auto w = fixture(p, "linear.weight");
    auto ref = fixture(p, "linear.output");
    int rows = static_cast<int>(in.shape[0]), in_f = static_cast<int>(in.shape[1]);
    int out_f = static_cast<int>(w.shape[0]);
    auto d_in = upload(in.data);
    auto d_w = upload(w.data);
    TypedDeviceBuffer<float> out(static_cast<int64_t>(rows) * out_f);
    ops::linear(d_in.data(), d_w.data(), out.data(), rows, in_f, out_f);
    check("linear", out.to_host(), ref.data);
}

void test_gemv(const PackReader &p) {
    auto in = fixture(p, "gemv.input");
    auto w = fixture(p, "gemv.weight");
    auto ref = fixture(p, "gemv.output");
    int in_f = static_cast<int>(in.shape[1]);
    int out_f = static_cast<int>(w.shape[0]);
    auto d_in = upload(in.data);
    auto d_w = upload(w.data);
    TypedDeviceBuffer<float> out(out_f);
    ops::linear(d_in.data(), d_w.data(), out.data(), 1, in_f, out_f);
    check("gemv(decode linear)", out.to_host(), ref.data);
}

static uint16_t f2bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<uint16_t>(u >> 16);
}

// Tensor Core linear: weights rounded to bf16 on the host, bf16 tolerance.
void test_linear_mma(const PackReader &p) {
    auto in = fixture(p, "linear.input");
    auto w = fixture(p, "linear.weight");
    auto ref = fixture(p, "linear.output");
    int rows = static_cast<int>(in.shape[0]), in_f = static_cast<int>(in.shape[1]);
    int out_f = static_cast<int>(w.shape[0]);

    std::vector<__nv_bfloat16> w_bf16(w.data.size());
    for (size_t i = 0; i < w.data.size(); ++i) {
        uint16_t bits = f2bf16_bits(w.data[i]);
        std::memcpy(&w_bf16[i], &bits, 2);
    }
    auto d_in = upload(in.data);
    auto d_w = TypedDeviceBuffer<__nv_bfloat16>::from_host(w_bf16.data(), w_bf16.size());
    TypedDeviceBuffer<float> out(static_cast<int64_t>(rows) * out_f);
    ops::linear_mma_bf16(d_in.data(), d_w.data(), out.data(), rows, in_f, out_f);
    Stats s = compare(out.to_host(), ref.data);
    bool ok = s.nrmse <= 1e-2 && s.cosine >= 0.999;
    std::printf("%-28s %s  max_abs=%.3g nrmse=%.3g cos=%.8f\n", "linear_mma_bf16", ok ? "PASS" : "FAIL",
                s.max_abs_err, s.nrmse, s.cosine);
    if (!ok)
        ++g_failures;

    // bf16 SIMT tiled path on the same inputs (rows>1)
    TypedDeviceBuffer<float> out2(static_cast<int64_t>(rows) * out_f);
    ops::linear(d_in.data(), d_w.data(), out2.data(), rows, in_f, out_f);
    Stats s2 = compare(out2.to_host(), ref.data);
    bool ok2 = s2.nrmse <= 1e-2 && s2.cosine >= 0.999;
    std::printf("%-28s %s  max_abs=%.3g nrmse=%.3g cos=%.8f\n", "linear_bf16_tiled", ok2 ? "PASS" : "FAIL",
                s2.max_abs_err, s2.nrmse, s2.cosine);
    if (!ok2)
        ++g_failures;
}

void test_embedding(const PackReader &p) {
    auto table = fixture(p, "embedding.table");
    auto ref = fixture(p, "embedding.output");
    auto ids = fixture_ids(p, "embedding.ids");
    int n = static_cast<int>(ids.size());
    int hidden = static_cast<int>(table.shape[1]);
    auto d_t = upload(table.data);
    auto d_ids = TypedDeviceBuffer<int32_t>::from_host(ids);
    TypedDeviceBuffer<float> out(static_cast<int64_t>(n) * hidden);
    ops::embedding(d_t.data(), d_ids.data(), out.data(), n, hidden);
    check("embedding", out.to_host(), ref.data);
}

void test_rope(const PackReader &p) {
    auto inv = fixture(p, "rope.inv_freq");
    auto q_in = fixture(p, "rope.q_in");
    auto k_in = fixture(p, "rope.k_in");
    auto q_ref = fixture(p, "rope.q_out");
    auto k_ref = fixture(p, "rope.k_out");
    auto pos = fixture_ids(p, "rope.positions");
    int rows = static_cast<int>(q_in.shape[0]);
    int head_dim = static_cast<int>(inv.shape[0]) * 2;
    int qh = static_cast<int>(q_in.shape[1]) / head_dim;
    int kvh = static_cast<int>(k_in.shape[1]) / head_dim;
    // positions are consecutive starting from pos[0]
    auto d_inv = upload(inv.data);
    {
        auto d = upload(q_in.data);
        ops::rope(d.data(), d_inv.data(), rows, qh, head_dim, pos[0]);
        check("rope.q", d.to_host(), q_ref.data);
    }
    {
        auto d = upload(k_in.data);
        ops::rope(d.data(), d_inv.data(), rows, kvh, head_dim, pos[0]);
        check("rope.k", d.to_host(), k_ref.data);
    }
}

void test_swiglu(const PackReader &p) {
    auto x = fixture(p, "swiglu.input");
    auto wg = fixture(p, "swiglu.gate_weight");
    auto wu = fixture(p, "swiglu.up_weight");
    auto ref = fixture(p, "swiglu.output");
    int rows = static_cast<int>(x.shape[0]), H = static_cast<int>(x.shape[1]);
    int I = static_cast<int>(wg.shape[0]);
    auto d_x = upload(x.data);
    auto d_wg = upload(wg.data);
    auto d_wu = upload(wu.data);
    TypedDeviceBuffer<float> gate(static_cast<int64_t>(rows) * I);
    TypedDeviceBuffer<float> up(static_cast<int64_t>(rows) * I);
    ops::linear(d_x.data(), d_wg.data(), gate.data(), rows, H, I);
    ops::linear(d_x.data(), d_wu.data(), up.data(), rows, H, I);
    ops::swiglu(gate.data(), up.data(), gate.data(), static_cast<int64_t>(rows) * I);
    check("swiglu", gate.to_host(), ref.data);
}

// Full prefill attention pipeline as the model runs it: split heads, per-head
// QK^T + causal scale + softmax + PV, merge heads.
void test_attn_prefill(const PackReader &p) {
    auto q = fixture(p, "attn_prefill.q"); // [T, qh, D]
    auto k = fixture(p, "attn_prefill.k"); // [T, kvh, D]
    auto v = fixture(p, "attn_prefill.v");
    auto ref = fixture(p, "attn_prefill.output");
    int T = static_cast<int>(q.shape[0]);
    int qh = static_cast<int>(q.shape[1]), D = static_cast<int>(q.shape[2]);
    int kvh = static_cast<int>(k.shape[1]);
    int rep = qh / kvh;
    float scale = 1.f / std::sqrt(static_cast<float>(D));

    // fixtures are [T, heads, D]; flatten to packed [T, heads*D]
    auto flat = [](const FixtureTensor &t) {
        std::vector<float> out(t.data.size());
        int T = t.shape[0], heads = t.shape[1], D = t.shape[2];
        for (int64_t i = 0; i < static_cast<int64_t>(t.data.size()); ++i) {
            int64_t tt = i / (heads * D);
            int64_t h = (i / D) % heads;
            int64_t d = i % D;
            out[(tt * heads + h) * D + d] = t.data[i];
        }
        return out;
    };
    auto d_q = upload(flat(q));
    auto d_k = upload(flat(k));
    auto d_v = upload(flat(v));

    TypedDeviceBuffer<float> q_heads(static_cast<int64_t>(qh) * T * D);
    TypedDeviceBuffer<float> k_heads(static_cast<int64_t>(kvh) * T * D);
    TypedDeviceBuffer<float> v_heads(static_cast<int64_t>(kvh) * T * D);
    TypedDeviceBuffer<float> scores(static_cast<int64_t>(T) * T);
    TypedDeviceBuffer<float> attn_heads(static_cast<int64_t>(qh) * T * D);
    TypedDeviceBuffer<float> out(static_cast<int64_t>(T) * qh * D);

    ops::split_heads(d_q.data(), q_heads.data(), T, qh, D, T);
    ops::split_heads(d_k.data(), k_heads.data(), T, kvh, D, T);
    ops::split_heads(d_v.data(), v_heads.data(), T, kvh, D, T);
    for (int h = 0; h < qh; ++h) {
        const float *q_h = q_heads.data() + static_cast<int64_t>(h) * T * D;
        const float *k_h = k_heads.data() + static_cast<int64_t>(h / rep) * T * D;
        const float *v_h = v_heads.data() + static_cast<int64_t>(h / rep) * T * D;
        float *o_h = attn_heads.data() + static_cast<int64_t>(h) * T * D;
        ops::linear(q_h, k_h, scores.data(), T, D, T);
        ops::causal_scale(scores.data(), T, T, scale);
        ops::softmax(scores.data(), scores.data(), T, T);
        ops::mat_mul(scores.data(), v_h, o_h, T, T, D);
    }
    ops::merge_heads(attn_heads.data(), out.data(), T, qh, D, T);

    // reference is [T, qh, D]; flatten the same way
    auto ref_flat = flat(ref);
    check("attention_prefill", out.to_host(), ref_flat);
}

void test_attn_decode(const PackReader &p) {
    auto q = fixture(p, "attn_decode.q"); // [1, qh, D]
    auto k = fixture(p, "attn_decode.k"); // [S, kvh, D]
    auto v = fixture(p, "attn_decode.v");
    auto ref = fixture(p, "attn_decode.output"); // [1, qh, D]
    int qh = static_cast<int>(q.shape[1]), D = static_cast<int>(q.shape[2]);
    int S = static_cast<int>(k.shape[0]), kvh = static_cast<int>(k.shape[1]);
    int rep = qh / kvh;

    // move K/V into [kvh, max_seq, D] cache layout
    std::vector<float> kc(static_cast<size_t>(kvh) * S * D), vc(kc.size());
    for (int t = 0; t < S; ++t)
        for (int h = 0; h < kvh; ++h)
            for (int d = 0; d < D; ++d) {
                kc[(static_cast<int64_t>(h) * S + t) * D + d] = k.data[(static_cast<int64_t>(t) * kvh + h) * D + d];
                vc[(static_cast<int64_t>(h) * S + t) * D + d] = v.data[(static_cast<int64_t>(t) * kvh + h) * D + d];
            }
    auto d_q = upload(q.data);
    auto d_k = upload(kc);
    auto d_v = upload(vc);
    TypedDeviceBuffer<float> out(static_cast<int64_t>(qh) * D);
    ops::attention_decode(d_q.data(), d_k.data(), d_v.data(), out.data(), qh, rep, D, S, S);
    check("attention_decode", out.to_host(), ref.data);
}

void test_argmax(const PackReader &p) {
    auto in = fixture(p, "argmax.input");
    auto ref = fixture_ids(p, "argmax.output");
    auto d_in = upload(in.data);
    TypedDeviceBuffer<int32_t> out(1);
    ops::argmax(d_in.data(), static_cast<int>(in.data.size()), out.data());
    int32_t got = out.to_host_scalar();
    bool ok = got == ref[0];
    std::printf("%-28s %s  got=%d ref=%d\n", "argmax", ok ? "PASS" : "FAIL", got, ref[0]);
    if (!ok)
        ++g_failures;
}

// Chunked (online-softmax) prefill attention: same fixture as the explicit
// path, plus a long-sequence self-consistency check against it.
void test_attn_chunked(const PackReader &p) {
    auto q = fixture(p, "attn_prefill.q");
    auto k = fixture(p, "attn_prefill.k");
    auto v = fixture(p, "attn_prefill.v");
    auto ref = fixture(p, "attn_prefill.output");
    int T = static_cast<int>(q.shape[0]);
    int qh = static_cast<int>(q.shape[1]), D = static_cast<int>(q.shape[2]);
    int kvh = static_cast<int>(k.shape[1]);
    int rep = qh / kvh;

    auto flat = [](const FixtureTensor &t) {
        std::vector<float> out(t.data.size());
        int T = t.shape[0], heads = t.shape[1], D = t.shape[2];
        for (int64_t i = 0; i < static_cast<int64_t>(t.data.size()); ++i) {
            int64_t tt = i / (heads * D);
            int64_t h = (i / D) % heads;
            int64_t d = i % D;
            out[(tt * heads + h) * D + d] = t.data[i];
        }
        return out;
    };
    // split into per-head slabs
    auto slab = [](const std::vector<float> &packed, int T, int heads, int D) {
        std::vector<float> out(packed.size());
        for (int64_t i = 0; i < static_cast<int64_t>(packed.size()); ++i) {
            int64_t t = i / (heads * D);
            int64_t h = (i / D) % heads;
            int64_t d = i % D;
            out[(h * static_cast<int64_t>(T) + t) * D + d] = packed[i];
        }
        return out;
    };

    {
        auto d_q = upload(slab(flat(q), T, qh, D));
        auto d_k = upload(slab(flat(k), T, kvh, D));
        auto d_v = upload(slab(flat(v), T, kvh, D));
        TypedDeviceBuffer<float> out(static_cast<int64_t>(qh) * T * D);
        ops::attention_prefill_chunked(d_q.data(), d_k.data(), d_v.data(), out.data(), T, T, 0, qh, rep, D, T);
        // compare against the reference (flattened to [qh, T, D])
        auto ref_slab = slab(flat(ref), T, qh, D);
        check("attn_chunked.fixture", out.to_host(), ref_slab);
    }

    // long sequence: chunked path vs explicit path must agree
    {
        const int Lt = 2500; // crosses the model's 2048-row threshold
        std::vector<float> hq(static_cast<size_t>(qh) * Lt * D), hk(static_cast<size_t>(kvh) * Lt * D), hv(hk.size());
        uint64_t rng = 42;
        auto frand = [&]() {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<float>((rng >> 33) & 0xFFFFFF) / 8388608.f - 1.5f;
        };
        for (auto &x : hq) x = frand();
        for (auto &x : hk) x = frand();
        for (auto &x : hv) x = frand();
        auto d_q = upload(hq);
        auto d_k = upload(hk);
        auto d_v = upload(hv);
        float scale = 1.f / std::sqrt(static_cast<float>(D));

        TypedDeviceBuffer<float> out_chunk(static_cast<int64_t>(qh) * Lt * D);
        ops::attention_prefill_chunked(d_q.data(), d_k.data(), d_v.data(), out_chunk.data(), Lt, Lt, 0, qh, rep, D,
                                       Lt);

        TypedDeviceBuffer<float> out_expl(static_cast<int64_t>(qh) * Lt * D);
        TypedDeviceBuffer<float> scores(static_cast<int64_t>(Lt) * Lt);
        for (int h = 0; h < qh; ++h) {
            const float *q_h = d_q.data() + static_cast<int64_t>(h) * Lt * D;
            const float *k_h = d_k.data() + static_cast<int64_t>(h / rep) * Lt * D;
            const float *v_h = d_v.data() + static_cast<int64_t>(h / rep) * Lt * D;
            float *o_h = out_expl.data() + static_cast<int64_t>(h) * Lt * D;
            ops::linear(q_h, k_h, scores.data(), Lt, D, Lt);
            ops::causal_scale(scores.data(), Lt, Lt, scale);
            ops::softmax(scores.data(), scores.data(), Lt, Lt);
            ops::mat_mul(scores.data(), v_h, o_h, Lt, Lt, D);
        }
        Stats s = compare(out_chunk.to_host(), out_expl.to_host());
        bool ok = s.nrmse <= 1e-5;
        std::printf("%-28s %s  max_abs=%.3g nrmse=%.3g cos=%.8f\n", "attn_chunked.long2500", ok ? "PASS" : "FAIL",
                    s.max_abs_err, s.nrmse, s.cosine);
        if (!ok)
            ++g_failures;
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string path = argc > 1 ? argv[1] : "tests/fixtures/llama/ops_fixtures.pack";
    PackReader pack(path);
    test_rms_norm(pack);
    test_linear(pack);
    test_gemv(pack);
    test_linear_mma(pack);
    test_embedding(pack);
    test_rope(pack);
    test_swiglu(pack);
    test_attn_prefill(pack);
    test_attn_chunked(pack);
    test_attn_decode(pack);
    test_argmax(pack);
    std::printf(g_failures == 0 ? "all operator tests passed\n" : "%d operator test(s) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
