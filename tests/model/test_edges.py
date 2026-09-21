#!/usr/bin/env python3
"""CLI/session edge-case tests (plan 8.1: 越界 ID、非法 shape、超长拒绝、
max_new_tokens=0/1、容量边界).

Exits 77 (CTest skip) when transformers or the built library is unavailable.
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/llama"))

try:
    from transformers import AutoTokenizer
    from generate import LlamaGenerator
except Exception as e:
    print(f"dependencies unavailable: {e}")
    sys.exit(77)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACK = os.environ.get("OPW_TEST_PACK", os.path.join(ROOT, "data", "llama-bf16.pack"))
SNAPSHOT = os.environ.get("OPW_TEST_TOKENIZER", os.path.join(ROOT, "data", "llama-3.2-1b-instruct"))


def main():
    if not os.path.exists(PACK):
        print("weight pack unavailable")
        return 77
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    enc = tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True)
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    ids = [int(i) for i in ids]

    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"{name:34s} {'PASS' if cond else 'FAIL'} {detail}")
        if not cond:
            failures += 1

    with LlamaGenerator(PACK, max_seq_len=256) as g:
        # max_new_tokens = 0 / 1
        check("max_new_tokens_0", g.generate_ids(ids, 0) == [])
        one = g.generate_ids(ids, 1)
        check("max_new_tokens_1", len(one) == 1 and 0 <= one[0] < 128256)

        # out-of-vocab token id rejected at the FFI boundary
        try:
            g.generate_ids(ids + [128256], 1)
            check("oob_token_rejected", False)
        except RuntimeError:
            check("oob_token_rejected", True)

        # over-capacity rejected with a clear error
        try:
            g.generate_ids(ids, 256 - len(ids) + 1)
            check("over_capacity_rejected", False)
        except ValueError:
            check("over_capacity_rejected", True)

        # exact capacity fill must not error (EOS may legitimately stop early)
        try:
            out = g.generate_ids(ids, 256 - len(ids))
            check("exact_capacity_fill", 1 <= len(out) <= 256 - len(ids), f"{len(out)} tokens")
        except Exception as e:
            check("exact_capacity_fill", False, str(e))

        # empty prompt rejected
        try:
            g.generate_ids([], 1)
            check("empty_prompt_rejected", False)
        except RuntimeError:
            check("empty_prompt_rejected", True)

        # sessions stay usable after errors
        out = g.generate_ids(ids, 4)
        check("usable_after_errors", len(out) >= 1)

    print("all edge tests passed" if failures == 0 else f"{failures} edge test(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
