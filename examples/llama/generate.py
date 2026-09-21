#!/usr/bin/env python3
"""Generate text with the OpWorks Llama runtime (greedy decoding).

The C++/CUDA shared library runs the full forward pass and KV cache; Python
only tokenizes (HF chat template) and detokenizes.

Usage:
    python examples/llama/generate.py \
        --model data/llama-fp32.pack \
        --tokenizer data/llama-3.2-1b-instruct \
        --prompt "Explain what a GPU does." \
        --max-seq-len 2048 --max-new-tokens 128
"""

import argparse
import ctypes
import os
import sys

_ERR_LEN = 512


class _Lib:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        lib.opw_llama_load.restype = ctypes.c_void_p
        lib.opw_llama_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.opw_llama_free.argtypes = [ctypes.c_void_p]
        lib.opw_llama_create_session.restype = ctypes.c_void_p
        lib.opw_llama_create_session.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.opw_llama_destroy_session.argtypes = [ctypes.c_void_p]
        lib.opw_llama_reset_session.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.opw_llama_prefill.restype = ctypes.c_int32
        lib.opw_llama_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.opw_llama_decode_step.restype = ctypes.c_int32
        lib.opw_llama_decode_step.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int]
        lib.opw_llama_is_eos.restype = ctypes.c_int
        lib.opw_llama_is_eos.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int]
        lib.opw_llama_last_logits.restype = ctypes.c_int
        lib.opw_llama_last_logits.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.opw_llama_vocab_size.restype = ctypes.c_int
        lib.opw_llama_vocab_size.argtypes = [ctypes.c_void_p]
        self.lib = lib
        self.err = ctypes.create_string_buffer(_ERR_LEN)

    def check(self, rc, what):
        if rc < 0:
            raise RuntimeError(f"{what} failed: {self.err.value.decode(errors='replace')}")
        return rc


class LlamaGenerator:
    def __init__(self, pack_path, lib_path=None, max_seq_len=2048):
        if lib_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lib_path = os.path.join(root, "build", "libopworks_llama.so")
        self._api = _Lib(lib_path)
        self._model = self._api.lib.opw_llama_load(pack_path.encode(), self._api.err, _ERR_LEN)
        if not self._model:
            raise RuntimeError(f"model load failed: {self._api.err.value.decode(errors='replace')}")
        self._session = self._api.lib.opw_llama_create_session(
            self._model, max_seq_len, self._api.err, _ERR_LEN
        )
        if not self._session:
            self._api.lib.opw_llama_free(self._model)
            raise RuntimeError(f"session creation failed: {self._api.err.value.decode(errors='replace')}")
        self.max_seq_len = max_seq_len
        self._vocab = self._api.lib.opw_llama_vocab_size(self._session)
        self._logits = (ctypes.c_float * self._vocab)()

    def _sample(self, rng, temperature, top_p):
        """Sample one token from the workspace logits (numpy, host side)."""
        import numpy as np

        api = self._api
        api.check(
            api.lib.opw_llama_last_logits(self._session, self._logits, self._vocab, api.err, _ERR_LEN),
            "last_logits",
        )
        logits = np.frombuffer(self._logits, dtype=np.float32).astype(np.float64)
        logits = logits / max(temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            keep = cum <= top_p
            keep[0] = True  # always keep the top token
            # include the token that crosses the threshold (HF semantics)
            cross = np.searchsorted(cum, top_p)
            if cross < len(order):
                keep[cross] = True
            mask = np.zeros_like(probs)
            mask[order[keep]] = probs[order[keep]]
            probs = mask / mask.sum()
        return int(rng.choice(self._vocab, p=probs))

    def generate_ids(self, prompt_ids, max_new_tokens, temperature=0.0, top_p=1.0, seed=None):
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds capacity {self.max_seq_len}"
            )
        import numpy as np

        api = self._api
        sample = temperature > 0.0
        rng = np.random.default_rng(seed) if sample else None
        api.check(api.lib.opw_llama_reset_session(self._session, api.err, _ERR_LEN), "reset")
        buf = (ctypes.c_int32 * max(1, len(prompt_ids)))(*prompt_ids)
        greedy = api.check(api.lib.opw_llama_prefill(self._session, buf, len(prompt_ids), api.err, _ERR_LEN),
                           "prefill")
        out = []
        for _ in range(max_new_tokens):
            tok = self._sample(rng, temperature, top_p) if sample else greedy
            out.append(tok)
            if api.check(api.lib.opw_llama_is_eos(self._session, tok, api.err, _ERR_LEN), "is_eos"):
                break
            if len(prompt_ids) + len(out) >= self.max_seq_len and len(out) < max_new_tokens:
                break  # no room for the next step's input position
            greedy = api.check(api.lib.opw_llama_decode_step(self._session, tok, api.err, _ERR_LEN), "decode")
        return out

    def close(self):
        if self._session:
            self._api.lib.opw_llama_destroy_session(self._session)
            self._session = None
        if self._model:
            self._api.lib.opw_llama_free(self._model)
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpWorks FP32 weight pack")
    ap.add_argument("--tokenizer", required=True, help="HF snapshot dir with tokenizer")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--lib", default=None, help="path to libopworks_llama.so")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--greedy", action="store_true", help="greedy decoding (the default when temperature=0)")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature; 0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling threshold")
    ap.add_argument("--seed", type=int, default=None, help="sampling RNG seed")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    messages = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    ids = [int(i) for i in ids]

    with LlamaGenerator(args.model, args.lib, args.max_seq_len) as gen:
        out_ids = gen.generate_ids(ids, args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                                   seed=args.seed)
    print(tok.decode(out_ids))
    print(f"\n---\nprompt tokens: {len(ids)}, generated tokens: {len(out_ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
