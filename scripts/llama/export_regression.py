#!/usr/bin/env python3
"""Export regression fixtures: greedy continuations for a fixed prompt set,
plus a long-context (crosses the chunked-attention threshold) reference.

Outputs in --output-dir:
    regression_fixtures.pack - per-prompt input_ids.{i} / generated_ids.{i}
    longctx_fixtures.pack    - long prompt: input_ids, per-layer last-position
                               hidden, final logits.last
    regression.json          - prompts, ids, versions
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_format import PackWriter

PROMPTS = [
    "What is the capital of France?",
    "Explain what a GPU does in one sentence.",
    "Write a haiku about the ocean.",
    "What is 17 plus 25?",
    "Name three primary colors.",
    "Give me a one-sentence summary of the plot of Romeo and Juliet.",
    "What is the difference between a list and a tuple in Python?",
    "Translate 'good morning' into German.",
    "What year did the Apollo 11 mission land on the Moon?",
    "Explain photosynthesis to a five-year-old.",
    "Write a Python one-liner that reverses a string.",
    "What is the largest planet in our solar system?",
    "Give two examples of renewable energy sources.",
    "What is the boiling point of water at sea level in Celsius?",
    "Define the word 'algorithm' in one sentence.",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the square root of 144?",
    "List the first five prime numbers.",
    "What does HTTP stand for?",
    "Why is the sky blue? Answer briefly.",
]

# Repetition of a fixed sentence builds a deterministic long prompt without
# needing a corpus.
LONG_SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. "


def chat_ids(tok, content):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True,
    )
    ids = list(enc.ids) if hasattr(enc, "ids") else enc["input_ids"]
    if hasattr(ids[0], "__len__"):
        ids = ids[0]
    return [int(i) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--long-prompt-tokens", type=int, default=2500,
                    help="approximate length; must exceed the explicit-attention threshold (2048)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    pad = model.generation_config.pad_token_id

    os.makedirs(args.output_dir, exist_ok=True)

    reg = PackWriter()
    meta = {"prompts": PROMPTS, "max_new_tokens": args.max_new_tokens, "items": []}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = chat_ids(tok, prompt)
            gen = model.generate(
                input_ids=torch.tensor([ids]), max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=pad,
            )[0].tolist()[len(ids):]
            reg.tensors[f"input_ids.{i}"] = (np.array(ids, dtype="<i4").tobytes(), [len(ids)])
            reg.tensors[f"generated_ids.{i}"] = (np.array(gen, dtype="<i4").tobytes(), [len(gen)])
            meta["items"].append({"prompt": prompt, "input_ids": ids, "generated_ids": gen})
            print(f"[{i + 1}/{len(PROMPTS)}] {len(ids)} prompt tokens -> {len(gen)} generated", flush=True)
    reg.write(os.path.join(args.output_dir, "regression_fixtures.pack"))
    print("wrote regression_fixtures.pack", flush=True)

    # ---- long-context reference (chunked attention path) ----
    long_text = LONG_SENTENCE * (args.long_prompt_tokens // 12 + 1)
    ids = chat_ids(tok, long_text)
    ids = ids[: args.long_prompt_tokens]
    n = len(ids)
    print(f"long prompt: {n} tokens; running reference forward...", flush=True)
    lc = PackWriter()
    lc.tensors["input_ids"] = (np.array(ids, dtype="<i4").tobytes(), [n])
    with torch.no_grad():
        pre_norm = {}
        hook = model.model.norm.register_forward_pre_hook(
            lambda m, inp: pre_norm.setdefault("x", inp[0].detach())
        )
        out = model(
            input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False
        )
        hook.remove()
        hidden = out.hidden_states
        num_layers = len(hidden) - 1
        # last-position hidden per layer keeps the pack small while still
        # localizing the first divergent layer
        for i, h in enumerate(hidden[:-1]):
            name = "hidden_last.embed" if i == 0 else f"hidden_last.layer{i - 1}"
            lc.add_tensor(name, h[0, -1].float().numpy())
        lc.add_tensor(f"hidden_last.layer{num_layers - 1}", pre_norm["x"][0, -1].float().numpy())
        lc.add_tensor("logits.last", out.logits[0, -1].float().numpy())
    lc.write(os.path.join(args.output_dir, "longctx_fixtures.pack"))
    print("wrote longctx_fixtures.pack", flush=True)

    import transformers

    meta["torch"] = torch.__version__
    meta["transformers"] = transformers.__version__
    meta["long_prompt_tokens"] = n
    with open(os.path.join(args.output_dir, "regression.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
