// C ABI for the OpWorks Llama runtime. Python (ctypes) drives generation
// through this boundary; all C++ exceptions are captured and returned as
// error strings — no exceptions cross the FFI.

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <variant>
#include <vector>

#include "../include/models/llama.cuh"

namespace {

thread_local std::string g_last_error;

int capture_error(const char *what, char *err, int err_len) {
    g_last_error = what ? what : "unknown error";
    if (err && err_len > 0) {
        std::strncpy(err, g_last_error.c_str(), static_cast<size_t>(err_len) - 1);
        err[err_len - 1] = '\0';
    }
    return -1;
}

template <typename F> int guard(F &&f, char *err, int err_len) {
    try {
        return f();
    } catch (const std::exception &e) {
        return capture_error(e.what(), err, err_len);
    } catch (...) {
        return capture_error("unknown C++ exception", err, err_len);
    }
}

using ModelVariant = std::variant<opworks::LlamaModel, opworks::LlamaModelBF16>;

struct ModelHolder {
    ModelVariant model;
};
struct SessionHolder {
    ModelHolder *owner;
    opworks::LlamaSession session;
    SessionHolder(ModelHolder *o, opworks::LlamaSession s) : owner(o), session(std::move(s)) {}
};

template <typename F> auto with_model(ModelHolder *m, F &&f) {
    return std::visit([&](auto &model) { return f(model); }, m->model);
}

} // namespace

extern "C" {

// Loads a weight pack (float32 or bfloat16). Returns nullptr on failure.
void *opw_llama_load(const char *pack_path, char *err, int err_len) {
    try {
        opworks::PackReader probe(pack_path);
        if (probe.dtype() == "float32")
            return new ModelHolder{opworks::LlamaModel::load(pack_path)};
        if (probe.dtype() == "bfloat16")
            return new ModelHolder{opworks::LlamaModelBF16::load(pack_path)};
        capture_error(("unsupported pack dtype " + probe.dtype()).c_str(), err, err_len);
        return nullptr;
    } catch (const std::exception &e) {
        capture_error(e.what(), err, err_len);
        return nullptr;
    } catch (...) {
        capture_error("unknown C++ exception", err, err_len);
        return nullptr;
    }
}

void opw_llama_free(void *model) { delete static_cast<ModelHolder *>(model); }

void *opw_llama_create_session(void *model, int max_seq_len, char *err, int err_len) {
    if (!model) {
        capture_error("null model", err, err_len);
        return nullptr;
    }
    try {
        auto *m = static_cast<ModelHolder *>(model);
        opworks::LlamaSession s = with_model(m, [&](auto &md) { return md.create_session(max_seq_len); });
        return new SessionHolder(m, std::move(s));
    } catch (const std::exception &e) {
        capture_error(e.what(), err, err_len);
        return nullptr;
    } catch (...) {
        capture_error("unknown C++ exception", err, err_len);
        return nullptr;
    }
}

void opw_llama_destroy_session(void *session) { delete static_cast<SessionHolder *>(session); }

int opw_llama_reset_session(void *session, char *err, int err_len) {
    return guard(
        [&] {
            static_cast<SessionHolder *>(session)->session.cache.reset();
            return 0;
        },
        err, err_len);
}

// Returns the first generated token id, or -1 on error.
int opw_llama_prefill(void *session, const int32_t *tokens, int n, char *err, int err_len) {
    return guard(
        [&] {
            auto *s = static_cast<SessionHolder *>(session);
            if (!tokens && n > 0)
                throw std::invalid_argument("null token buffer");
            std::vector<int32_t> ids(tokens, tokens + static_cast<size_t>(n));
            return with_model(s->owner, [&](auto &m) { return m.prefill(s->session, ids); });
        },
        err, err_len);
}

// Returns the next token id, or -1 on error.
int opw_llama_decode_step(void *session, int32_t prev_token, char *err, int err_len) {
    return guard(
        [&] {
            auto *s = static_cast<SessionHolder *>(session);
            return with_model(s->owner, [&](auto &m) { return m.decode_step(s->session, prev_token); });
        },
        err, err_len);
}

// 1 if token ends generation, 0 otherwise, -1 on error.
int opw_llama_is_eos(void *session, int32_t token, char *err, int err_len) {
    return guard(
        [&] {
            auto *s = static_cast<SessionHolder *>(session);
            return with_model(s->owner, [&](auto &m) { return m.is_eos(token) ? 1 : 0; });
        },
        err, err_len);
}

// Copies the current last-position logits (vocab_size floats) into out.
int opw_llama_last_logits(void *session, float *out, int n, char *err, int err_len) {
    return guard(
        [&] {
            auto *s = static_cast<SessionHolder *>(session);
            if (!out)
                throw std::invalid_argument("null logits buffer");
            with_model(s->owner, [&](auto &m) { m.copy_last_logits(s->session, out, n); });
            return 0;
        },
        err, err_len);
}

int opw_llama_vocab_size(void *session) {
    auto *s = static_cast<SessionHolder *>(session);
    return with_model(s->owner, [&](auto &m) { return m.config().vocab_size; });
}

const char *opw_llama_last_error() { return g_last_error.c_str(); }

} // extern "C"
