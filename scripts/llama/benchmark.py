#!/usr/bin/env python3
"""OpWorks Llama inference benchmark.

Reports (median and P95 over reps, after warmup):
  - model load time, tokenizer time
  - prefill time / TTFT (first token)
  - decode per-token latency and tokens/s
  - peak GPU memory delta of the process (cudaMemGetInfo)

Prompt lengths default to the plan's matrix; generation length is fixed (no
EOS early stop) so timings are comparable.

Usage:
    python scripts/llama/benchmark.py --model data/llama-bf16.pack \
        --tokenizer data/llama-3.2-1b-instruct
    python scripts/llama/benchmark.py ... --hf-baseline   # needs CUDA torch
"""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/llama"))
from generate import LlamaGenerator


def gpu_mem_used_bytes():
    """Process-visible CUDA memory usage via libcudart (total - free)."""
    lib = ctypes.CDLL("libcudart.so")
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return total.value - free.value


def build_prompt_ids(tok, n):
    # repeated sentence, then truncated to exactly n tokens
    text = "The quick brown fox jumps over the lazy dog near the river bank. " * (n // 10 + 2)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=True
    )
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    ids = [int(i) for i in ids]
    return ids[:n]


def timed_generate(gen, ids, max_new):
    """Returns (prefill_s, decode_s_per_token, n_decode_tokens)."""
    api = gen._api
    session = gen._session
    err = api.err
    buf = (ctypes.c_int32 * len(ids))(*ids)
    api.check(api.lib.opw_llama_reset_session(session, err, 512), "reset")
    t0 = time.perf_counter()
    tok = api.check(api.lib.opw_llama_prefill(session, buf, len(ids), err, 512), "prefill")
    t1 = time.perf_counter()
    n = 0
    for _ in range(max_new - 1):  # first token came from prefill
        tok = api.check(api.lib.opw_llama_decode_step(session, tok, err, 512), "decode")
        n += 1
    t2 = time.perf_counter()
    return (t1 - t0), ((t2 - t1) / max(n, 1)), n


def summarize(name, values):
    med = statistics.median(values)
    p95 = sorted(values)[max(0, int(len(values) * 0.95) - 1)]
    return {"median": med, "p95": p95, "min": min(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 1024, 4096, 8000])
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", default=None, help="write results JSON here")
    ap.add_argument("--hf-baseline", action="store_true", help="also benchmark HF transformers (CUDA)")
    args = ap.parse_args()

    results = {"model": args.model, "gen_tokens": args.gen_tokens, "reps": args.reps, "cases": []}

    t0 = time.perf_counter()
    gen = LlamaGenerator(args.model, args.lib, max_seq_len=max(args.prompt_lens) + args.gen_tokens + 16)
    load_s = time.perf_counter() - t0
    results["model_load_s"] = load_s
    base_mem = gpu_mem_used_bytes()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.perf_counter()
    probe_ids = build_prompt_ids(tok, args.prompt_lens[0])
    results["tokenizer_s"] = time.perf_counter() - t0

    for plen in args.prompt_lens:
        ids = probe_ids if plen == args.prompt_lens[0] else build_prompt_ids(tok, plen)
        plen = len(ids)
        max_new = min(args.gen_tokens, gen.max_seq_len - plen)
        if max_new <= 0:
            print(f"skip prompt_len={plen}: no room to generate")
            continue
        for _ in range(args.warmup):
            timed_generate(gen, ids, max_new)
        prefill, per_tok = [], []
        mem_peak = 0
        for _ in range(args.reps):
            p, d, _ = timed_generate(gen, ids, max_new)
            prefill.append(p)
            per_tok.append(d)
            mem_peak = max(mem_peak, gpu_mem_used_bytes())
        ps = summarize("prefill", prefill)
        ds = summarize("decode", per_tok)
        case = {
            "prompt_tokens": plen,
            "gen_tokens": max_new,
            "ttft_s": ps,  # prefill == time to first token here
            "decode_s_per_token": ds,
            "decode_tokens_per_s": 1.0 / ds["median"],
            "peak_gpu_mem_mib": round((mem_peak - 0) / 2**20),
        }
        results["cases"].append(case)
        print(
            f"prompt={plen:5d} gen={max_new:4d}  TTFT {ps['median']*1e3:8.1f} ms (p95 {ps['p95']*1e3:8.1f})  "
            f"decode {ds['median']*1e6:7.0f} us/tok ({1/ds['median']:7.1f} tok/s, p95 {1/ds['p95']:7.1f})  "
            f"mem {case['peak_gpu_mem_mib']} MiB"
        )
    print(f"model load {load_s:.2f}s, tokenizer {results['tokenizer_s']*1e3:.1f}ms")
    gen.close()

    if args.hf_baseline:
        try:
            results["hf_baseline"] = run_hf_baseline(args)
        except Exception as e:  # report, don't fail the run
            print(f"HF baseline unavailable: {e}", file=sys.stderr)
            results["hf_baseline_error"] = str(e)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output}")


def run_hf_baseline(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda not available")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.tokenizer, dtype=torch.bfloat16).cuda().eval()
    out = {"dtype": "bfloat16", "attention": str(model.config._attn_implementation), "cases": []}
    with torch.no_grad():
        for plen in args.prompt_lens:
            ids = build_prompt_ids(tok, plen)
            plen = len(ids)
            max_new = min(args.gen_tokens, 8192 - plen)
            input_ids = torch.tensor([ids], device="cuda")
            torch.cuda.synchronize()
            for _ in range(args.warmup):
                model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=model.generation_config.pad_token_id)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=model.generation_config.pad_token_id)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) / max_new)
            med = statistics.median(times)
            out["cases"].append({
                "prompt_tokens": plen,
                "gen_tokens": max_new,
                "e2e_s_per_token": med,
                "e2e_tokens_per_s": 1.0 / med,
                "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            })
            print(f"  HF prompt={plen:5d} e2e {1/med:7.1f} tok/s  mem {out['cases'][-1]['peak_gpu_mem_mib']} MiB")
    return out


if __name__ == "__main__":
    main()
