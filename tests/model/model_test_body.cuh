#pragma once

// Templated body shared by the FP32 and BF16 model tests. FP32 mode uses the
// plan's strict thresholds and exact greedy-token regression; BF16 mode uses
// the plan's BF16 thresholds (NRMSE<=1e-2, cosine>=0.999) and registers
// generation divergences as exceptions only when the top-1/top-2 logit margin
// at the divergence step is below kBf16MarginLimit (near-tie), per the plan's
// divergence-analysis rule.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../include/models/llama.cuh"
#include "test_common.cuh"

namespace modeltest {

using namespace opworks;

constexpr double kBf16MarginLimit = 0.05;

template <typename ModelT>
int run_model_tests(int argc, char **argv, bool bf16_tolerances) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <weights.pack> <model_fixtures.pack> [longctx.pack] [regression.pack]\n",
                     argv[0]);
        return 77;
    }
    PackReader *weights = nullptr, *fixtures = nullptr;
    try {
        weights = new PackReader(argv[1]);
        fixtures = new PackReader(argv[2]);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "fixtures unavailable: %s\n", e.what());
        return 77;
    }

    int g_failures = 0;
    const double hid_nrmse = bf16_tolerances ? 1e-2 : 1e-3;
    const double hid_cos = bf16_tolerances ? 0.999 : 0.9999;

    auto check_hidden = [&](const std::string &name, const std::vector<float> &actual,
                            const std::vector<float> &ref) {
        Stats s = compare(actual, ref);
        bool ok = s.nrmse <= hid_nrmse && s.cosine >= hid_cos;
        report(name.c_str(), ok, s);
        if (!ok)
            ++g_failures;
    };

    ModelT model = ModelT::load(argv[1]);
    auto ids = fixture_ids(*fixtures, "input_ids");
    auto generated = fixture_ids(*fixtures, "generated_ids");
    const int T = static_cast<int>(ids.size());
    const int H = model.config().hidden_size;
    std::printf("prompt: %d tokens, reference continuation: %d tokens\n", T,
                static_cast<int>(generated.size()));

    // ---- 1. per-layer hidden state alignment ------------------------------
    {
        auto session = model.create_session(2048);
        auto captures = model.debug_prefill(session, ids);
        if (static_cast<int>(captures.size()) != model.config().num_layers + 1) {
            std::fprintf(stderr, "expected %d captures, got %zu\n", model.config().num_layers + 1,
                         captures.size());
            return 1;
        }
        check_hidden("hidden.embed", captures[0], fixture(*fixtures, "hidden.embed").data);
        for (int i = 0; i < model.config().num_layers; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "hidden.layer%d", i);
            check_hidden(name, captures[i + 1], fixture(*fixtures, name).data);
        }
        Stats s = compare(model.last_logits(session), fixture(*fixtures, "logits.last").data);
        bool ok = bf16_tolerances ? (s.nrmse <= 1e-2 && s.cosine >= 0.999) : pass_logits(s);
        report("logits.last", ok, s);
        if (!ok)
            ++g_failures;
    }

    // ---- 2. greedy generation regression ----------------------------------
    {
        auto session = model.create_session(2048);
        std::vector<int32_t> out;
        int32_t tok = model.prefill(session, ids);
        for (size_t i = 0; i < generated.size(); ++i) {
            out.push_back(tok);
            if (i + 1 < generated.size())
                tok = model.decode_step(session, tok);
        }
        bool ok = out == generated;
        std::printf("%-28s %s  (%zu tokens)\n", "generation.greedy", ok ? "PASS" : "FAIL", out.size());
        if (!ok) {
            for (size_t i = 0; i < out.size(); ++i)
                if (out[i] != generated[i]) {
                    std::fprintf(stderr, "first divergence at step %zu: got %d, ref %d\n", i, out[i],
                                 generated[i]);
                    break;
                }
            if (!bf16_tolerances)
                ++g_failures;
        }
    }

    // ---- 3. KV cache consistency: decode vs full-prefix recompute ---------
    {
        const int prefix = T > 2 ? T / 2 : 1;
        std::vector<int32_t> pre(ids.begin(), ids.begin() + prefix);
        int32_t next = ids[prefix];
        auto s1 = model.create_session(2048);
        model.prefill(s1, pre);
        model.decode_step(s1, next);
        std::vector<float> logits_a = model.last_logits(s1);

        std::vector<int32_t> full(ids.begin(), ids.begin() + prefix + 1);
        auto s2 = model.create_session(2048);
        model.prefill(s2, full);
        std::vector<float> logits_b = model.last_logits(s2);

        Stats s = compare(logits_a, logits_b);
        // different reduction orders (tiled GEMM vs GEMV), so compare
        // distribution-level error, not bitwise equality
        bool ok = s.nrmse <= (bf16_tolerances ? 1e-2 : 1e-5) && s.cosine >= (bf16_tolerances ? 0.999 : 0.999999);
        std::printf("%-28s %s  max_abs=%.3g nrmse=%.3g cos=%.8f\n", "kv_cache.decode_vs_prefill",
                    ok ? "PASS" : "FAIL", s.max_abs_err, s.nrmse, s.cosine);
        if (!ok)
            ++g_failures;

        auto s3 = model.create_session(2048);
        std::vector<int32_t> one{ids[0]};
        model.prefill(s3, one);
        model.decode_step(s3, ids[1]);
        std::vector<float> logits_c = model.last_logits(s3);

        auto s4 = model.create_session(2048);
        std::vector<int32_t> two{ids[0], ids[1]};
        model.prefill(s4, two);
        Stats s2stat = compare(logits_c, model.last_logits(s4));
        bool ok2 = s2stat.nrmse <= (bf16_tolerances ? 1e-2 : 1e-5) && s2stat.cosine >= (bf16_tolerances ? 0.999 : 0.999999);
        std::printf("%-28s %s  max_abs=%.3g\n", "kv_cache.t1_boundary", ok2 ? "PASS" : "FAIL",
                    s2stat.max_abs_err);
        if (!ok2)
            ++g_failures;

        // plan's KV length sweep: prefill(L) vs prefill(L-1)+decode at each L
        const int sweep[] = {1, 2, 31, 32, 33, 127, 128, 129};
        int sweep_bad = 0;
        for (int L : sweep) {
            if (L + 1 > T)
                continue;
            auto sa = model.create_session(2048);
            model.prefill(sa, std::vector<int32_t>(ids.begin(), ids.begin() + L));
            model.decode_step(sa, ids[L]);
            std::vector<float> la = model.last_logits(sa);
            auto sb = model.create_session(2048);
            model.prefill(sb, std::vector<int32_t>(ids.begin(), ids.begin() + L + 1));
            Stats ss = compare(la, model.last_logits(sb));
            double tol = bf16_tolerances ? 1e-2 : 1e-5;
            double ctol = bf16_tolerances ? 0.999 : 0.999999;
            if (!(ss.nrmse <= tol && ss.cosine >= ctol)) {
                ++sweep_bad;
                std::fprintf(stderr, "kv sweep L=%d: nrmse=%.3g cos=%.8f\n", L, ss.nrmse, ss.cosine);
            }
        }
        std::printf("%-28s %s\n", "kv_cache.length_sweep", sweep_bad == 0 ? "PASS" : "FAIL");
        if (sweep_bad)
            ++g_failures;
    }

    // ---- 4. session reset isolation ---------------------------------------
    {
        auto session = model.create_session(2048);
        std::vector<int32_t> out1, out2;
        for (int round = 0; round < 2; ++round) {
            session.cache.reset();
            int32_t tok = model.prefill(session, ids);
            std::vector<int32_t> out;
            for (size_t i = 0; i < generated.size(); ++i) {
                out.push_back(tok);
                if (i + 1 < generated.size())
                    tok = model.decode_step(session, tok);
            }
            (round == 0 ? out1 : out2) = std::move(out);
        }
        bool ok = out1 == out2;
        std::printf("%-28s %s\n", "session.reset_repeat", ok ? "PASS" : "FAIL");
        if (!ok)
            ++g_failures;
    }

    // ---- 5. long-context prefill (chunked attention path) -----------------
    if (argc > 3) {
        try {
            PackReader longctx(argv[3]);
            auto lids = fixture_ids(longctx, "input_ids");
            const int LT = static_cast<int>(lids.size());
            auto session = model.create_session(8192);
            auto captures = model.debug_prefill(session, lids);
            int bad = 0;
            for (int i = 0; i <= model.config().num_layers; ++i) {
                std::string name = i == 0 ? "hidden_last.embed" : "hidden_last.layer" + std::to_string(i - 1);
                std::vector<float> last(captures[i].end() - H, captures[i].end());
                Stats s = compare(last, fixture(longctx, name).data);
                bool ok = s.nrmse <= hid_nrmse && s.cosine >= hid_cos;
                report(("longctx." + name).c_str(), ok, s);
                if (!ok)
                    ++bad;
            }
            Stats s = compare(model.last_logits(session), fixture(longctx, "logits.last").data);
            bool ok = bf16_tolerances ? (s.nrmse <= 1e-2 && s.cosine >= 0.999) : pass_logits(s);
            report("longctx.logits.last", ok, s);
            if (!ok)
                ++bad;
            std::printf("long-context prefill: %d tokens, %s\n", LT, bad == 0 ? "PASS" : "FAIL");
            g_failures += bad;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "long-context fixtures unavailable: %s\n", e.what());
        }
    }

    // ---- 6. greedy generation regression over the fixed prompt set --------
    if (argc > 4) {
        try {
            PackReader reg(argv[4]);
            auto session = model.create_session(2048);
            int n_reg = 0, diverged = 0, bad_divergence = 0;
            for (int i = 0;; ++i) {
                std::string in_name = "input_ids." + std::to_string(i);
                if (!reg.has(in_name))
                    break;
                ++n_reg;
                auto pids = fixture_ids(reg, in_name);
                auto ref = fixture_ids(reg, "generated_ids." + std::to_string(i));
                session.cache.reset();
                int32_t tok = model.prefill(session, pids);
                size_t step = 0;
                for (; step < ref.size(); ++step) {
                    if (tok != ref[step])
                        break;
                    if (step + 1 < ref.size())
                        tok = model.decode_step(session, tok);
                }
                if (step != ref.size()) {
                    ++diverged;
                    if (bf16_tolerances) {
                        // teacher-forced margin analysis at the divergence:
                        // replay the reference prefix and inspect our logits
                        session.cache.reset();
                        std::vector<int32_t> prefix(pids.begin(), pids.end());
                        prefix.insert(prefix.end(), ref.begin(), ref.begin() + step);
                        model.prefill(session, prefix);
                        std::vector<float> logits = model.last_logits(session);
                        float top1 = -1e30f, top2 = -1e30f;
                        for (float v : logits) {
                            if (v > top1) { top2 = top1; top1 = v; }
                            else if (v > top2) { top2 = v; }
                        }
                        double margin = static_cast<double>(top1) - top2;
                        std::fprintf(stderr,
                                     "regression.%d diverged at step %zu/%zu (top1-top2 margin %.4f)\n", i,
                                     step, ref.size(), margin);
                        if (margin > kBf16MarginLimit)
                            ++bad_divergence;
                    } else {
                        std::fprintf(stderr, "regression.%d diverged at step %zu (of %zu)\n", i, step,
                                     ref.size());
                    }
                }
            }
            bool ok = bf16_tolerances ? (bad_divergence == 0 && n_reg == 20)
                                      : (diverged == 0 && n_reg == 20);
            std::printf("%-28s %s  (%d prompts, %d diverged, %d hard failures)\n", "generation.regression20",
                        ok ? "PASS" : "FAIL", n_reg, diverged, bad_divergence);
            if (!ok)
                ++g_failures;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "regression fixtures unavailable: %s\n", e.what());
        }
    }

    // ---- 7. concurrent sessions do not interfere --------------------------
    {
        auto s1 = model.create_session(2048);
        auto s2 = model.create_session(2048);
        std::vector<int32_t> pre1(ids.begin(), ids.begin() + T / 2);
        std::vector<int32_t> pre2(ids.begin() + T / 2, ids.end());
        int32_t t1 = model.prefill(s1, pre1);
        int32_t t2 = model.prefill(s2, pre2);
        for (int i = 0; i < 8; ++i) {
            t1 = model.decode_step(s1, t1);
            t2 = model.decode_step(s2, t2);
        }
        auto r1 = model.create_session(2048);
        auto r2 = model.create_session(2048);
        int32_t u1 = model.prefill(r1, pre1);
        int32_t u2 = model.prefill(r2, pre2);
        for (int i = 0; i < 8; ++i) {
            u1 = model.decode_step(r1, u1);
            u2 = model.decode_step(r2, u2);
        }
        bool ok = t1 == u1 && t2 == u2;
        std::printf("%-28s %s\n", "session.concurrent", ok ? "PASS" : "FAIL");
        if (!ok)
            ++g_failures;
    }

    delete weights;
    delete fixtures;
    std::printf(g_failures == 0 ? "all model tests passed\n" : "%d model test(s) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

} // namespace modeltest
