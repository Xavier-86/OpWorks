#!/usr/bin/env python3
"""Export a HF Llama-3.2 checkpoint into an OpWorks FP32 tensor pack.

Usage:
    python scripts/llama/export_weights.py \
        --model-dir data/llama-3.2-1b-instruct \
        --output data/llama-fp32.pack

The pack contains all model weights as float32 plus the model config and the
stop-token set, so the C++ runtime needs nothing else.
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_format import PackWriter

EXPECTED_PER_LAYER = {
    "self_attn.q_proj.weight": lambda c: [c["num_attention_heads"] * c["head_dim"], c["hidden_size"]],
    "self_attn.k_proj.weight": lambda c: [c["num_key_value_heads"] * c["head_dim"], c["hidden_size"]],
    "self_attn.v_proj.weight": lambda c: [c["num_key_value_heads"] * c["head_dim"], c["hidden_size"]],
    "self_attn.o_proj.weight": lambda c: [c["hidden_size"], c["num_attention_heads"] * c["head_dim"]],
    "mlp.gate_proj.weight": lambda c: [c["intermediate_size"], c["hidden_size"]],
    "mlp.up_proj.weight": lambda c: [c["intermediate_size"], c["hidden_size"]],
    "mlp.down_proj.weight": lambda c: [c["hidden_size"], c["intermediate_size"]],
    "input_layernorm.weight": lambda c: [c["hidden_size"]],
    "post_attention_layernorm.weight": lambda c: [c["hidden_size"]],
}


def check_config(cfg):
    required = {
        "model_type": "llama",
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
        "pretraining_tp": 1,
    }
    for key, want in required.items():
        got = cfg.get(key)
        if got != want:
            raise SystemExit(f"unsupported config: {key}={got!r} (expected {want!r})")
    for key in [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
    ]:
        if key not in cfg:
            raise SystemExit(f"config missing required field {key}")
    if cfg["num_attention_heads"] * cfg["head_dim"] != cfg["hidden_size"]:
        raise SystemExit("num_attention_heads * head_dim != hidden_size")
    if cfg["num_attention_heads"] % cfg["num_key_value_heads"] != 0:
        raise SystemExit("num_attention_heads not divisible by num_key_value_heads")
    rs = cfg.get("rope_scaling")
    if rs is not None:
        if rs.get("rope_type") != "llama3":
            raise SystemExit(f"unsupported rope_scaling type {rs.get('rope_type')!r}")
        for key in ["factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"]:
            if key not in rs:
                raise SystemExit(f"rope_scaling missing {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF snapshot directory")
    ap.add_argument("--output", required=True, help="output pack path")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--source", default=None, help="source repo id recorded in the manifest")
    ap.add_argument("--revision", default=None, help="source commit recorded in the manifest")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    check_config(cfg)

    with open(os.path.join(args.model_dir, "generation_config.json")) as f:
        gen_cfg = json.load(f)
    eos = gen_cfg.get("eos_token_id", cfg.get("eos_token_id"))
    eos_ids = sorted(set(eos if isinstance(eos, list) else [eos]))

    st_path = os.path.join(args.model_dir, "model.safetensors")
    with open(st_path, "rb") as f:
        st_sha256 = hashlib.sha256(f.read()).hexdigest()

    writer = PackWriter()
    import torch

    with safe_open(st_path, framework="pt", device="cpu") as f:
        names = set(f.keys())

        def get(name):
            if name not in names:
                raise SystemExit(f"missing tensor {name}")
            t = f.get_tensor(name)
            return t.to(torch.float32).numpy() if args.dtype == "float32" else t

        def put(name, t):
            if args.dtype == "float32":
                writer.add_tensor(name, t)
            else:
                writer.add_tensor_bf16(name, t)

        embed = get("model.embed_tokens.weight")
        if list(embed.shape) != [cfg["vocab_size"], cfg["hidden_size"]]:
            raise SystemExit(f"embed_tokens shape {embed.shape}")
        put("model.embed_tokens.weight", embed)
        if "lm_head.weight" in names:
            head = get("lm_head.weight")
            same = np.array_equal(head, embed) if args.dtype == "float32" else torch.equal(head, embed)
            if not same:
                raise SystemExit("lm_head.weight differs from embed_tokens despite tie_word_embeddings")
        else:
            writer.add_alias("lm_head.weight", "model.embed_tokens.weight")

        norm = get("model.norm.weight")
        if list(norm.shape) != [cfg["hidden_size"]]:
            raise SystemExit(f"model.norm shape {norm.shape}")
        put("model.norm.weight", norm)

        for i in range(cfg["num_hidden_layers"]):
            prefix = f"model.layers.{i}."
            for suffix, shape_fn in EXPECTED_PER_LAYER.items():
                want = shape_fn(cfg)
                t = get(prefix + suffix)
                if list(t.shape) != want:
                    raise SystemExit(f"{prefix}{suffix}: shape {list(t.shape)} != expected {want}")
                put(prefix + suffix, t)

        extra_names = names - set(writer.tensors) - {"lm_head.weight"}
        if extra_names:
            raise SystemExit(f"unexpected tensors in checkpoint: {sorted(extra_names)}")

    slim_config = {
        k: cfg[k]
        for k in [
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "bos_token_id",
        ]
    }
    slim_config["rope_scaling"] = cfg.get("rope_scaling")

    meta = {
        "source": args.source or "unknown",
        "revision": args.revision or "unknown",
        "model_safetensors_sha256": st_sha256,
        "python": platform.python_version(),
    }
    try:
        import torch
        import safetensors
        import transformers

        meta.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            }
        )
    except ImportError:
        pass

    writer.write(
        args.output,
        dtype=args.dtype,
        extra={"config": slim_config, "eos_token_ids": eos_ids, "meta": meta},
    )
    size = os.path.getsize(args.output)
    print(f"wrote {args.output} ({size / 2**30:.2f} GiB, {args.dtype}), eos={eos_ids}")
    print(f"source={meta['source']}@{meta['revision']} sha256={st_sha256}")


if __name__ == "__main__":
    main()
