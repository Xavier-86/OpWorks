#!/usr/bin/env python3
"""vLLM baseline for the OpWorks Llama benchmark (same card, same model,
same workload). Run with the vLLM environment's python, e.g.:

    .venv-vllm/bin/python scripts/llama/benchmark_vllm.py \
        --tokenizer data/llama-3.2-1b-instruct --output data/bench_vllm.json

Methodology matches scripts/llama/benchmark.py: batch=1, greedy, fixed
128 generated tokens (ignore_eos), 3 warmup + N reps, median/P95.
TTFT is measured with a max_tokens=1 run; decode tok/s is derived from
(full_run - ttft_run) / (gen_tokens - 1). vLLM runs with its defaults
(CUDA graphs + paged attention), which is the intended framework baseline.
"""

import argparse
import json
import statistics
import sys
import time


def build_prompt_ids(tok, n):
    text = "The quick brown fox jumps over the lazy dog near the river bank. " * (n // 10 + 2)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=True
    )
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    return [int(i) for i in ids][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF snapshot dir (also the model path)")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 1024, 4096, 8000])
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.perf_counter()
    llm = LLM(
        model=args.tokenizer,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,  # reps reuse the same prompt; caching would fake TTFT
    )
    load_s = time.perf_counter() - t0

    results = {"engine": f"vllm", "load_s": load_s, "gen_tokens": args.gen_tokens, "cases": []}
    print(f"vllm load {load_s:.1f}s")

    for plen in args.prompt_lens:
        ids = build_prompt_ids(tok, plen)
        plen = len(ids)
        gen_tokens = min(args.gen_tokens, args.max_model_len - plen)
        if gen_tokens < 2:
            print(f"skip prompt_len={plen}")
            continue
        sp_full = SamplingParams(temperature=0.0, max_tokens=gen_tokens, ignore_eos=True)
        sp_one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        prompt = [{"prompt_token_ids": ids}]

        for _ in range(args.warmup):
            llm.generate(prompt, sp_full, use_tqdm=False)

        ttfts, fulls = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate(prompt, sp_one, use_tqdm=False)
            ttfts.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out = llm.generate(prompt, sp_full, use_tqdm=False)
            fulls.append(time.perf_counter() - t0)
        ttft = statistics.median(ttfts)
        full = statistics.median(fulls)
        decode_tps = (gen_tokens - 1) / max(full - ttft, 1e-9)
        case = {
            "prompt_tokens": plen,
            "gen_tokens": gen_tokens,
            "ttft_s": ttft,
            "e2e_s": full,
            "e2e_tokens_per_s": gen_tokens / full,
            "decode_tokens_per_s": decode_tps,
        }
        results["cases"].append(case)
        print(f"prompt={plen:5d} gen={gen_tokens}  TTFT {ttft*1e3:8.1f} ms  decode ~{decode_tps:7.1f} tok/s  "
              f"e2e {gen_tokens/full:7.1f} tok/s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
