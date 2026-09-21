#pragma once

// Llama-3.2 decoder forward pass on OpWorks ops (FP32 correctness version).
//
// Layout conventions:
//   - packed row-major activations [T, width]
//   - per-head slabs [heads, rows, head_dim] for attention math
//   - KV cache [num_kv_heads, max_seq_len, head_dim] per layer; K is stored
//     post-RoPE, V as projected
//
// One LlamaModel owns the weights; each Session owns its KV cache, workspace
// and stream. Prefill/decode run on the session's stream; the token result is
// synchronized back to the host each step (greedy decode needs it anyway).

#include <cuda_bf16.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include "../core/cuda_utils.cuh"
#include "../ops/add.cuh"
#include "../ops/argmax.cuh"
#include "../ops/attention_decode.cuh"
#include "../ops/attention_prefill.cuh"
#include "../ops/causal_scale.cuh"
#include "../ops/embedding.cuh"
#include "../ops/linear.cuh"
#include "../ops/linear_mma.cuh"
#include "../ops/matmul.cuh"
#include "../ops/rms_norm.cuh"
#include "../ops/rope.cuh"
#include "../ops/softmax.cuh"
#include "../ops/split_heads.cuh"
#include "../ops/swiglu.cuh"
#include "../runtime/kv_cache.cuh"
#include "../runtime/model_config.cuh"
#include "../runtime/pack_reader.cuh"
#include "../runtime/weights.cuh"
#include "../runtime/workspace.cuh"

namespace opworks {

// Per-session state: KV cache, workspace, stream. Independent of the weight
// storage type, so it lives outside the model template.
struct LlamaSession {
        KvCache cache;
        Workspace workspace;
        cudaStream_t stream = nullptr;

        LlamaSession(const ModelConfig &cfg, int max_seq_len) : cache(cfg, max_seq_len), workspace(cfg, max_seq_len) {
            OPWORKS_CUDA_CHECK(cudaStreamCreate(&stream));
        }
        ~LlamaSession() {
            if (stream)
                cudaStreamDestroy(stream);
        }
        LlamaSession(const LlamaSession &) = delete;
        LlamaSession &operator=(const LlamaSession &) = delete;
        LlamaSession(LlamaSession &&o) noexcept
            : cache(std::move(o.cache)), workspace(std::move(o.workspace)), stream(o.stream) {
            o.stream = nullptr;
        }
        LlamaSession &operator=(LlamaSession &&o) noexcept {
            if (this != &o) {
                if (stream)
                    cudaStreamDestroy(stream);
                cache = std::move(o.cache);
                workspace = std::move(o.workspace);
                stream = o.stream;
                o.stream = nullptr;
            }
            return *this;
        }
};

template <typename WT> class LlamaModelT {
  public:
    using Session = LlamaSession;

    static LlamaModelT load(const std::string &pack_path) {
        PackReader pack(pack_path);
        LlamaModelT m;
        m.cfg_ = ModelConfig::from_manifest(pack.manifest());
        m.weights_ = ModelWeights<WT>::load(pack, m.cfg_);
        return m;
    }

    const ModelConfig &config() const { return cfg_; }

    Session create_session(int max_seq_len) const {
        if (max_seq_len <= 0 || max_seq_len > detail::kAttnDecodeMaxLen)
            throw std::invalid_argument("max_seq_len must be in (0, 8192]");
        return Session(cfg_, max_seq_len);
    }

    // Runs the prompt through the model, fills the KV cache, and returns the
    // first generated token (argmax of the last position's logits).
    int32_t prefill(Session &s, const std::vector<int32_t> &ids) {
        const int n = static_cast<int>(ids.size());
        validate_tokens(ids);
        if (n == 0)
            throw std::invalid_argument("prefill requires at least one token");
        if (n > s.cache.max_seq_len)
            throw std::invalid_argument("prompt exceeds the session capacity");
        if (s.cache.length != 0)
            throw std::invalid_argument("prefill requires a fresh session (reset first)");

        OPWORKS_CUDA_CHECK(cudaMemcpyAsync(s.workspace.tokens.data(), ids.data(), n * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, s.stream));
        forward(s, n, 0);
        s.cache.length = n;
        return last_token(s);
    }

    // Appends one token (produced by the previous step) and returns the next.
    int32_t decode_step(Session &s, int32_t token) {
        if (token < 0 || token >= cfg_.vocab_size)
            throw std::invalid_argument("token id outside the vocabulary");
        const int pos = s.cache.length;
        if (pos >= s.cache.max_seq_len)
            throw std::invalid_argument("session capacity exhausted");

        OPWORKS_CUDA_CHECK(cudaMemcpyAsync(s.workspace.tokens.data(), &token, sizeof(int32_t),
                                           cudaMemcpyHostToDevice, s.stream));
        forward(s, 1, pos);
        s.cache.length = pos + 1;
        return last_token(s);
    }

    bool is_eos(int32_t token) const {
        return std::find(cfg_.eos_token_ids.begin(), cfg_.eos_token_ids.end(), token) != cfg_.eos_token_ids.end();
    }

    // Debug helper: full hidden state after all layers for inspection.
    std::vector<float> hidden_state(Session &s, int rows) const {
        std::vector<float> out(static_cast<size_t>(rows) * cfg_.hidden_size);
        OPWORKS_CUDA_CHECK(cudaMemcpyAsync(out.data(), s.workspace.hidden.data(), out.size() * 4,
                                           cudaMemcpyDeviceToHost, s.stream));
        synchronize(s.stream);
        return out;
    }

    std::vector<float> last_logits(Session &s) const { return s.workspace.logits.to_host(s.stream); }

    // Copies the current last-position logits into a host buffer of size
    // vocab_size (used by the sampling path).
    void copy_last_logits(Session &s, float *out, int n) const {
        if (n != cfg_.vocab_size)
            throw std::invalid_argument("logits buffer must hold vocab_size floats");
        OPWORKS_CUDA_CHECK(cudaMemcpyAsync(out, s.workspace.logits.data(), static_cast<size_t>(n) * 4,
                                           cudaMemcpyDeviceToHost, s.stream));
        synchronize(s.stream);
    }

    // Test/debug entry point: like prefill, but also returns the residual
    // stream after embedding (index 0) and after each decoder layer.
    std::vector<std::vector<float>> debug_prefill(Session &s, const std::vector<int32_t> &ids) {
        const int n = static_cast<int>(ids.size());
        validate_tokens(ids);
        if (n == 0)
            throw std::invalid_argument("prefill requires at least one token");
        if (n > s.cache.max_seq_len)
            throw std::invalid_argument("prompt exceeds the session capacity");
        if (s.cache.length != 0)
            throw std::invalid_argument("prefill requires a fresh session (reset first)");

        std::vector<std::vector<float>> captures;
        OPWORKS_CUDA_CHECK(cudaMemcpyAsync(s.workspace.tokens.data(), ids.data(), n * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, s.stream));
        forward(s, n, 0, [&](int) {
            std::vector<float> h(static_cast<size_t>(n) * cfg_.hidden_size);
            OPWORKS_CUDA_CHECK(cudaMemcpyAsync(h.data(), s.workspace.hidden.data(), h.size() * 4,
                                               cudaMemcpyDeviceToHost, s.stream));
            synchronize(s.stream);
            captures.push_back(std::move(h));
        });
        s.cache.length = n;
        return captures;
    }

  private:
    // Prefill with more rows than this uses the chunked attention kernel
    // instead of materializing the [rows, S] score matrix.
    static constexpr int kExplicitAttnRows = 2048;

    // Projection GEMM dispatch: bf16 weights with multiple rows go to the
    // Tensor Core kernel; everything else uses the SIMT tiled/GEMV kernels.
    static void linear_dispatch(const float *X, const WT *W, float *Y, int rows, int in_f, int out_f,
                                cudaStream_t st) {
        if constexpr (std::is_same_v<WT, __nv_bfloat16>) {
            if (rows >= 16) {
                ops::linear_mma_bf16(X, W, Y, rows, in_f, out_f, st);
                return;
            }
        }
        ops::linear(X, W, Y, rows, in_f, out_f, st);
    }

    ModelConfig cfg_;
    ModelWeights<WT> weights_;

    void validate_tokens(const std::vector<int32_t> &ids) const {
        for (int32_t id : ids)
            if (id < 0 || id >= cfg_.vocab_size)
                throw std::invalid_argument("token id outside the vocabulary");
    }

    // Core forward for `rows` tokens starting at absolute position pos_start.
    // Input token ids are already in ws.tokens. Leaves the residual stream in
    // ws.hidden and last-position logits in ws.logits.
    // layer_hook (tests only) is called with -1 after embedding and with the
    // layer index after each decoder layer; ws.hidden holds the current
    // residual stream at that point.
    void forward(Session &s, int rows, int pos_start, const std::function<void(int)> &layer_hook = nullptr) {
        Workspace &w = s.workspace;
        const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
        const int D = cfg_.head_dim, qh = cfg_.num_q_heads, kvh = cfg_.num_kv_heads;
        const int S = pos_start + rows; // cache entries visible to this pass
        const int cap = s.cache.max_seq_len;
        const float scale = 1.f / std::sqrt(static_cast<float>(D));
        cudaStream_t st = s.stream;

        ops::embedding(weights_.embed_tokens.data(), w.tokens.data(), w.hidden.data(), rows, H, st);
        if (layer_hook)
            layer_hook(-1);

        for (int layer = 0; layer < cfg_.num_layers; ++layer) {
            const LayerWeights<WT> &lw = weights_.layers[layer];
            float *kc = s.cache.k[layer].data();
            float *vc = s.cache.v[layer].data();

            ops::rms_norm(w.hidden.data(), lw.input_norm.data(), w.normed.data(), rows, H, cfg_.rms_norm_eps, st);
            linear_dispatch(w.normed.data(), lw.q_proj.data(), w.q.data(), rows, H, qh * D, st);
            linear_dispatch(w.normed.data(), lw.k_proj.data(), w.k_proj.data(), rows, H, kvh * D, st);
            linear_dispatch(w.normed.data(), lw.v_proj.data(), w.v_proj.data(), rows, H, kvh * D, st);
            ops::rope(w.q.data(), weights_.rope_inv_freq.data(), rows, qh, D, pos_start, st);
            ops::rope(w.k_proj.data(), weights_.rope_inv_freq.data(), rows, kvh, D, pos_start, st);
            ops::split_heads(w.k_proj.data(), kc, rows, kvh, D, cap, pos_start, st);
            ops::split_heads(w.v_proj.data(), vc, rows, kvh, D, cap, pos_start, st);

            if (rows == 1) {
                ops::attention_decode(w.q.data(), kc, vc, w.attn_out.data(), qh, cfg_.kv_rep(), D, S, cap, st);
            } else {
                ops::split_heads(w.q.data(), w.q_heads.data(), rows, qh, D, rows, 0, st);
                if (rows > kExplicitAttnRows) {
                    // long prompts: chunked online-softmax, no [T, S] matrix
                    ops::attention_prefill_chunked(w.q_heads.data(), kc, vc, w.attn_heads.data(), rows, S,
                                                   pos_start, qh, cfg_.kv_rep(), D, cap, st);
                } else {
                    for (int h = 0; h < qh; ++h) {
                        const float *q_h = w.q_heads.data() + static_cast<int64_t>(h) * rows * D;
                        const float *k_h = kc + static_cast<int64_t>(h / cfg_.kv_rep()) * cap * D;
                        const float *v_h = vc + static_cast<int64_t>(h / cfg_.kv_rep()) * cap * D;
                        float *o_h = w.attn_heads.data() + static_cast<int64_t>(h) * rows * D;
                        ops::linear(q_h, k_h, w.scores.data(), rows, D, S, st);
                        ops::causal_scale(w.scores.data(), rows, S, scale, st);
                        ops::softmax(w.scores.data(), w.scores.data(), rows, S, st);
                        ops::mat_mul(w.scores.data(), v_h, o_h, rows, S, D, ops::PassThrough{}, st);
                    }
                }
                ops::merge_heads(w.attn_heads.data(), w.attn_out.data(), rows, qh, D, rows, 0, st);
            }

            linear_dispatch(w.attn_out.data(), lw.o_proj.data(), w.normed.data(), rows, qh * D, H, st);
            ops::add(w.hidden.data(), w.normed.data(), w.hidden.data(), static_cast<int64_t>(rows) * H, st);

            ops::rms_norm(w.hidden.data(), lw.post_norm.data(), w.normed.data(), rows, H, cfg_.rms_norm_eps, st);
            linear_dispatch(w.normed.data(), lw.gate_proj.data(), w.gate.data(), rows, H, I, st);
            linear_dispatch(w.normed.data(), lw.up_proj.data(), w.up.data(), rows, H, I, st);
            ops::swiglu(w.gate.data(), w.up.data(), w.gate.data(), static_cast<int64_t>(rows) * I, st);
            linear_dispatch(w.gate.data(), lw.down_proj.data(), w.normed.data(), rows, I, H, st);
            ops::add(w.hidden.data(), w.normed.data(), w.hidden.data(), static_cast<int64_t>(rows) * H, st);
            if (layer_hook)
                layer_hook(layer);
        }

        ops::rms_norm(w.hidden.data(), weights_.final_norm.data(), w.normed.data(), rows, H, cfg_.rms_norm_eps,
                      st);
        // logits for the last position only (tied embedding as LM head)
        ops::linear(w.normed.data() + static_cast<int64_t>(rows - 1) * H, weights_.embed_tokens.data(),
                    w.logits.data(), 1, H, cfg_.vocab_size, st);
        ops::argmax(w.logits.data(), cfg_.vocab_size, w.argmax_out.data(), st);
    }

    int32_t last_token(Session &s) const { return s.workspace.argmax_out.to_host_scalar(s.stream); }
};

using LlamaModel = LlamaModelT<float>;
using LlamaModelBF16 = LlamaModelT<__nv_bfloat16>;

} // namespace opworks
