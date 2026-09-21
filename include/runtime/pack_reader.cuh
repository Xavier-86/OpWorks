#pragma once

// Host-side reader for OpWorks tensor packs (see scripts/llama/pack_format.py).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.cuh"

namespace opworks {

class PackReader {
  public:
    struct TensorMeta {
        uint64_t offset; // bytes, relative to the blob start
        std::vector<int64_t> shape;
        int64_t numel;
    };

    explicit PackReader(const std::string &path) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f)
            throw std::runtime_error("cannot open pack " + path);

        char magic[8];
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "OPWPACK1", 8) != 0) {
            std::fclose(f);
            throw std::runtime_error(path + ": not an OpWorks pack (bad magic)");
        }
        uint64_t json_len = 0;
        if (std::fread(&json_len, 8, 1, f) != 1 || json_len > (1ull << 30)) {
            std::fclose(f);
            throw std::runtime_error(path + ": corrupt manifest length");
        }
        std::string js(json_len, '\0');
        if (std::fread(js.data(), 1, json_len, f) != json_len) {
            std::fclose(f);
            throw std::runtime_error(path + ": truncated manifest");
        }
        blob_offset_ = 8 + 8 + json_len;

        std::fseek(f, 0, SEEK_END);
        uint64_t file_size = static_cast<uint64_t>(std::ftell(f));
        std::fclose(f);
        if (file_size < blob_offset_)
            throw std::runtime_error(path + ": truncated pack");
        blob_size_ = file_size - blob_offset_;

        manifest_ = json::parse(js);
        if (manifest_.at("format_version").as_int() != 1)
            throw std::runtime_error(path + ": unsupported format version");
        dtype_ = manifest_.at("dtype").as_string();
        if (dtype_ == "float32")
            elem_size_ = 4;
        else if (dtype_ == "bfloat16")
            elem_size_ = 2;
        else
            throw std::runtime_error(path + ": unsupported dtype " + dtype_);

        for (const auto &[name, meta] : manifest_.at("tensors").as_object()) {
            TensorMeta m;
            m.offset = static_cast<uint64_t>(meta.at("offset").as_int());
            m.numel = 1;
            for (const auto &d : meta.at("shape").as_array()) {
                int64_t dim = d.as_int();
                if (dim <= 0)
                    throw std::runtime_error(path + ": non-positive dim in " + name);
                m.numel *= dim;
                if (m.numel > (1ll << 40))
                    throw std::runtime_error(path + ": absurd tensor size in " + name);
                m.shape.push_back(dim);
            }
            uint64_t bytes = static_cast<uint64_t>(m.numel) * elem_size_;
            if (m.offset + bytes > blob_size_)
                throw std::runtime_error(path + ": tensor " + name + " overruns the data blob");
            tensors_.emplace(name, m);
        }
        if (const json::Value *aliases = manifest_.find("aliases")) {
            for (const auto &[alias, target] : aliases->as_object()) {
                const std::string &t = target.as_string();
                if (tensors_.find(t) == tensors_.end())
                    throw std::runtime_error(path + ": alias " + alias + " targets missing tensor " + t);
                aliases_.emplace(alias, t);
            }
        }
        path_ = path;
    }

    bool has(const std::string &name) const {
        return tensors_.count(name) != 0 || aliases_.count(name) != 0;
    }

    const TensorMeta &meta(const std::string &name) const {
        auto it = tensors_.find(resolve(name));
        if (it == tensors_.end())
            throw std::runtime_error(path_ + ": missing tensor " + name);
        return it->second;
    }

    const std::string &dtype() const { return dtype_; }
    size_t elem_size() const { return elem_size_; }

    // Reads one tensor's raw bytes into host memory.
    std::vector<uint8_t> read_raw(const std::string &name) const {
        const TensorMeta &m = meta(name);
        std::vector<uint8_t> out(static_cast<size_t>(m.numel) * elem_size_);
        FILE *f = std::fopen(path_.c_str(), "rb");
        if (!f)
            throw std::runtime_error("cannot reopen pack " + path_);
        if (std::fseek(f, static_cast<long>(blob_offset_ + m.offset), SEEK_SET) != 0 ||
            std::fread(out.data(), 1, out.size(), f) != out.size()) {
            std::fclose(f);
            throw std::runtime_error(path_ + ": failed reading tensor " + name);
        }
        std::fclose(f);
        return out;
    }

    // Reads one float32 tensor into host memory.
    std::vector<float> read(const std::string &name) const {
        if (dtype_ != "float32")
            throw std::runtime_error(path_ + ": read() requires a float32 pack; use read_raw for " + dtype_);
        std::vector<uint8_t> raw = read_raw(name);
        std::vector<float> out(raw.size() / 4);
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }

    // bf16 tensor converted to float32 on the host (for small tensors such as
    // norm weights that stay fp32 in the runtime).
    std::vector<float> read_bf16_as_float(const std::string &name) const {
        if (dtype_ != "bfloat16")
            throw std::runtime_error(path_ + ": read_bf16_as_float requires a bfloat16 pack");
        std::vector<uint8_t> raw = read_raw(name);
        std::vector<float> out(raw.size() / 2);
        for (size_t i = 0; i < out.size(); ++i) {
            uint32_t bits = static_cast<uint32_t>(raw[2 * i]) | (static_cast<uint32_t>(raw[2 * i + 1]) << 8);
            uint32_t f32 = bits << 16;
            std::memcpy(&out[i], &f32, 4);
        }
        return out;
    }

    // Reads raw bytes of an int32 tensor stored via the pack's escape hatch
    // (fixture ids); model packs only contain float32 tensors.
    std::vector<int32_t> read_int32(const std::string &name) const {
        const TensorMeta &m = meta(name);
        std::vector<float> tmp = read(name);
        std::vector<int32_t> out(tmp.size());
        std::memcpy(out.data(), tmp.data(), tmp.size() * 4);
        return out;
    }

    const json::Value &manifest() const { return manifest_; }

  private:
    const std::string &resolve(const std::string &name) const {
        auto it = aliases_.find(name);
        return it == aliases_.end() ? name : it->second;
    }

    std::string path_;
    std::string dtype_;
    size_t elem_size_ = 4;
    json::Value manifest_;
    std::map<std::string, TensorMeta> tensors_;
    std::map<std::string, std::string> aliases_;
    uint64_t blob_offset_ = 0;
    uint64_t blob_size_ = 0;
};

} // namespace opworks
