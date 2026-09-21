#!/usr/bin/env python3
"""Tokenizer/chat-template pipeline regression for the OpWorks CLI.

Checks (plan section 8.1 "输入"):
  - identical messages produce identical token ids (determinism)
  - chat template applied exactly once (single BOS, also for multi-turn)
  - empty message, Unicode, newlines, special-token-like text all tokenize
    to in-vocab ids
Exits 77 (CTest skip) when transformers is unavailable.
"""

import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not available")
    sys.exit(77)

BOS = 128000


def chat_ids(tok, messages):
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    return [int(i) for i in ids]


def main():
    tok = AutoTokenizer.from_pretrained("data/llama-3.2-1b-instruct")
    # model vocab (128256) covers the added special tokens; tok.vocab_size
    # (128000) does not
    vocab = 128256
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"{name:34s} {'PASS' if cond else 'FAIL'} {detail}")
        if not cond:
            failures += 1

    cases = {
        "empty": [{"role": "user", "content": ""}],
        "unicode": [{"role": "user", "content": "你好，世界！こんにちは — éèê 🚀"}],
        "newlines": [{"role": "user", "content": "line1\n\nline2\r\nline3"}],
        "special_like_text": [{"role": "user", "content": "what is <|begin_of_text|> <|eot_id|>?"}],
        "multi_turn": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ],
    }
    for name, messages in cases.items():
        ids = chat_ids(tok, messages)
        check(f"tokenize.{name}", len(ids) > 0 and all(0 <= i < vocab for i in ids),
              f"{len(ids)} tokens")

    # determinism
    a = chat_ids(tok, cases["unicode"])
    b = chat_ids(tok, cases["unicode"])
    check("tokenize.deterministic", a == b)

    # template applied exactly once: BOS only at position 0
    for name, messages in cases.items():
        if name == "special_like_text":
            continue  # see special_tokens_parsed below
        ids = chat_ids(tok, messages)
        check(f"single_bos.{name}", ids[0] == BOS and ids.count(BOS) == 1)

    # special-token text: HF's tokenizer parses "<|begin_of_text|>" in plain
    # text into the special id by default. Pin that contract so a tokenizer
    # upgrade that changes it is caught here; the engine consumes the ids
    # verbatim either way.
    ids = chat_ids(tok, cases["special_like_text"])
    check("special_tokens_parsed", ids[0] == BOS and ids.count(BOS) == 2 and 128009 in ids)

    print("all tokenizer tests passed" if failures == 0 else f"{failures} tokenizer test(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
