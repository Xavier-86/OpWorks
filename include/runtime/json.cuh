#pragma once

// Minimal JSON parser for pack manifests: objects, arrays, strings (with the
// common escapes), numbers, booleans, null. Host-only, no external deps.

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace opworks::json {

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    using Data = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    Data data;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    double as_number() const {
        if (!is_number())
            throw std::runtime_error("json: expected number");
        return std::get<double>(data);
    }
    int64_t as_int() const { return static_cast<int64_t>(as_number()); }
    const std::string &as_string() const {
        if (!is_string())
            throw std::runtime_error("json: expected string");
        return std::get<std::string>(data);
    }
    const Array &as_array() const {
        if (!is_array())
            throw std::runtime_error("json: expected array");
        return std::get<Array>(data);
    }
    const Object &as_object() const {
        if (!is_object())
            throw std::runtime_error("json: expected object");
        return std::get<Object>(data);
    }
    const Value &at(const std::string &key) const {
        const auto &obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end())
            throw std::runtime_error("json: missing key \"" + key + "\"");
        return it->second;
    }
    const Value *find(const std::string &key) const {
        if (!is_object())
            return nullptr;
        const auto &obj = std::get<Object>(data);
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
};

class Parser {
  public:
    explicit Parser(const std::string &text) : s_(text) {}

    Value parse() {
        Value v = parse_value();
        skip_ws();
        if (pos_ != s_.size())
            fail("trailing characters");
        return v;
    }

  private:
    const std::string &s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const char *msg) {
        throw std::runtime_error("json parse error at byte " + std::to_string(pos_) + ": " + msg);
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }

    char peek() {
        skip_ws();
        if (pos_ >= s_.size())
            fail("unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c)
            fail("unexpected character");
        ++pos_;
    }

    Value parse_value() {
        char c = peek();
        switch (c) {
        case '{': {
            ++pos_;
            Object obj;
            if (peek() == '}') {
                ++pos_;
                return Value{std::move(obj)};
            }
            while (true) {
                if (peek() != '"')
                    fail("expected object key");
                std::string key = parse_string();
                expect(':');
                obj.emplace(std::move(key), parse_value());
                char next = peek();
                ++pos_;
                if (next == '}')
                    break;
                if (next != ',')
                    fail("expected ',' or '}'");
            }
            return Value{std::move(obj)};
        }
        case '[': {
            ++pos_;
            Array arr;
            if (peek() == ']') {
                ++pos_;
                return Value{std::move(arr)};
            }
            while (true) {
                arr.push_back(parse_value());
                char next = peek();
                ++pos_;
                if (next == ']')
                    break;
                if (next != ',')
                    fail("expected ',' or ']'");
            }
            return Value{std::move(arr)};
        }
        case '"':
            return Value{parse_string()};
        case 't':
            consume_literal("true");
            return Value{true};
        case 'f':
            consume_literal("false");
            return Value{false};
        case 'n':
            consume_literal("null");
            return Value{nullptr};
        default:
            return Value{parse_number()};
        }
    }

    void consume_literal(const char *lit) {
        for (const char *p = lit; *p; ++p) {
            if (pos_ >= s_.size() || s_[pos_] != *p)
                fail("bad literal");
            ++pos_;
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"')
                return out;
            if (c == '\\') {
                if (pos_ >= s_.size())
                    fail("bad escape");
                char e = s_[pos_++];
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                    // BMP escapes only; sufficient for ASCII manifests.
                    if (pos_ + 4 > s_.size())
                        fail("bad \\u escape");
                    {
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = s_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code += h - '0';
                            else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
                            else fail("bad \\u escape");
                        }
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                    }
                    break;
                default:
                    fail("unknown escape");
                }
            } else {
                out += c;
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-')
            ++pos_;
        while (pos_ < s_.size() && (isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.' ||
                                    s_[pos_] == 'e' || s_[pos_] == 'E' || s_[pos_] == '+' || s_[pos_] == '-'))
            ++pos_;
        if (start == pos_)
            fail("expected value");
        try {
            return std::stod(s_.substr(start, pos_ - start));
        } catch (const std::exception &) {
            fail("bad number");
        }
    }
};

inline Value parse(const std::string &text) { return Parser(text).parse(); }

} // namespace opworks::json
