#!/usr/bin/env python3
"""Export reference fixtures for OpWorks Llama inference.

Produces in --output-dir:
    ops_fixtures.pack   - per-operator input/output pairs (seeded, real shapes)
    model_fixtures.pack - chat-templated token ids, per-layer hidden states,
                          layer-0 intermediates, final logits, greedy tokens
    reference.json      - meta: prompt, versions, tolerances context

Everything runs on CPU in float32 with eager attention and TF32 disabled.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_format import PackWriter

DEFAULT_PROMPT = "Explain what a GPU does in one sentence."


# ---------------------------------------------------------------------------
# Reference math (mirrors HF transformers llama, float32 eager)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def llama3_inv_freq(head_dim, theta, scaling):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    if scaling is None:
        return inv_freq
    factor = scaling["factor"]
    low = scaling["low_freq_factor"]
    high = scaling["high_freq_factor"]
    old_len = scaling["original_max_position_embeddings"]
    low_wl = old_len / low
    high_wl = old_len / high
    wavelen = 2 * math.pi / inv_freq
    out = torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
    smooth = (old_len / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * out / factor + smooth * out
    is_medium = (wavelen >= high_wl) & (wavelen <= low_wl)
    return torch.where(is_medium, smoothed, out)


def apply_rope(x, positions, inv_freq):
    """x: [T, H*D] -> rotated, HF rotate_half convention.
    positions: [T] long tensor of absolute positions."""
    T, HD = x.shape
    D = inv_freq.numel() * 2
    x = x.view(T, -1, D)
    freqs = torch.outer(positions.float(), inv_freq)  # [T, D/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[ :, None, :]  # [T,1,D]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[:, None, :]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    out = x * cos + rotated * sin
    return out.reshape(T, HD)


def causal_attention(q, k, v, num_kv_heads):
    """q: [T, qh, D], k/v: [S, kvh, D] -> [T, qh, D]; query position offset = S - T."""
    T, qh, D = q.shape
    S = k.shape[0]
    rep = qh // num_kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(D)
    qpos = torch.arange(S - T, S)[:, None]
    kpos = torch.arange(S)[None, :]
    mask = kpos > qpos  # [T, S] True = masked
    scores = scores.masked_fill(mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probs, v)


def silu_mul(gate, up):
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------

def build_op_fixtures(cfg):
    torch.manual_seed(1234)
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    D = cfg["head_dim"]
    qh = cfg["num_attention_heads"]
    kvh = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]
    w = PackWriter()

    # rms_norm: [5, H]
    x = torch.randn(5, H) * 3
    weight = torch.randn(H) * 0.5 + 1.0
    w.add_tensor("rms_norm.input", x.numpy())
    w.add_tensor("rms_norm.weight", weight.numpy())
    w.add_tensor("rms_norm.output", rms_norm(x, weight, eps).numpy())

    # linear: X[3, H] @ W[512, H].T
    x = torch.randn(3, H)
    wt = torch.randn(512, H) / math.sqrt(H)
    w.add_tensor("linear.input", x.numpy())
    w.add_tensor("linear.weight", wt.numpy())
    w.add_tensor("linear.output", (x @ wt.T).numpy())

    # linear single-row (decode GEMV path): [1, H] @ [I, H].T
    x = torch.randn(1, H)
    wt = torch.randn(I, H) / math.sqrt(H)
    w.add_tensor("gemv.input", x.numpy())
    w.add_tensor("gemv.weight", wt.numpy())
    w.add_tensor("gemv.output", (x @ wt.T).numpy())

    # embedding: gather rows (incl. id 0 and last row)
    table = torch.randn(cfg["vocab_size"], H) * 0.02
    ids = np.array([0, 5, cfg["vocab_size"] - 1, 128000], dtype=np.int32)
    w.add_tensor("embedding.table", table.numpy())
    w.add_tensor("embedding.output", table[torch.from_numpy(ids.astype(np.int64))].numpy())
    w.tensors["embedding.ids"] = (ids.astype("<i4").tobytes(), [len(ids)])

    # rope: q [17, qh*D], k [17, kvh*D], nonzero positions
    inv_freq = llama3_inv_freq(D, cfg["rope_theta"], cfg.get("rope_scaling"))
    q = torch.randn(17, qh * D)
    k = torch.randn(17, kvh * D)
    pos = torch.arange(17) + 29  # start at 29 to exercise scaling + rotation
    w.add_tensor("rope.inv_freq", inv_freq.numpy())
    w.add_tensor("rope.q_in", q.numpy())
    w.add_tensor("rope.k_in", k.numpy())
    w.add_tensor("rope.q_out", apply_rope(q, pos, inv_freq).numpy())
    w.add_tensor("rope.k_out", apply_rope(k, pos, inv_freq).numpy())
    w.tensors["rope.positions"] = (pos.numpy().astype("<i4").tobytes(), [17])

    # swiglu: [4, H] -> gate/up [I, H]
    x = torch.randn(4, H)
    wg = torch.randn(I, H) / math.sqrt(H)
    wu = torch.randn(I, H) / math.sqrt(H)
    w.add_tensor("swiglu.input", x.numpy())
    w.add_tensor("swiglu.gate_weight", wg.numpy())
    w.add_tensor("swiglu.up_weight", wu.numpy())
    w.add_tensor("swiglu.output", silu_mul(x @ wg.T, x @ wu.T).numpy())

    # attention prefill: T=37 causal
    q = torch.randn(37, qh, D)
    k = torch.randn(37, kvh, D)
    v = torch.randn(37, kvh, D)
    w.add_tensor("attn_prefill.q", q.numpy())
    w.add_tensor("attn_prefill.k", k.numpy())
    w.add_tensor("attn_prefill.v", v.numpy())
    w.add_tensor("attn_prefill.output", causal_attention(q, k, v, kvh).numpy())

    # attention decode: single query at the end of a 37-token context
    q = torch.randn(1, qh, D)
    w.add_tensor("attn_decode.q", q.numpy())
    w.add_tensor("attn_decode.k", k.numpy())
    w.add_tensor("attn_decode.v", v.numpy())
    w.add_tensor("attn_decode.output", causal_attention(q, k, v, kvh).numpy())

    # argmax over vocab-sized vector
    x = torch.randn(cfg["vocab_size"])
    w.add_tensor("argmax.input", x.numpy())
    w.tensors["argmax.output"] = (
        np.array([int(torch.argmax(x))], dtype="<i4").tobytes(),
        [1],
    )
    return w


def build_model_fixtures(model_dir, cfg_json, prompt, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")  # no TF32
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    if hasattr(enc, "ids"):  # tokenizers.Encoding
        ids = list(enc.ids)
    else:
        ids = enc["input_ids"]
        if hasattr(ids[0], "__len__"):  # batched [1, T]
            ids = ids[0]
    ids = [int(i) for i in ids]
    input_ids = torch.tensor([ids], dtype=torch.long)

    w = PackWriter()
    w.tensors["input_ids"] = (np.array(ids, dtype="<i4").tobytes(), [len(ids)])

    with torch.no_grad():
        pre_norm = {}
        hook = model.model.norm.register_forward_pre_hook(
            lambda m, inp: pre_norm.setdefault("x", inp[0].detach())
        )
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hook.remove()
        hidden = out.hidden_states  # (embed_out, layer1_out, ..., layerL_out)
        logits = out.logits[0, -1].float()

        # transformers v5 returns the final-normed tensor as hidden_states[-1];
        # keep the per-layer entries pre-norm for layer-by-layer alignment.
        num_layers = len(hidden) - 1
        for i, h in enumerate(hidden[:-1]):
            name = "hidden.embed" if i == 0 else f"hidden.layer{i - 1}"
            w.add_tensor(name, h[0].float().numpy())
        w.add_tensor(f"hidden.layer{num_layers - 1}", pre_norm["x"][0].float().numpy())
        w.add_tensor("hidden.final_normed", hidden[-1][0].float().numpy())
        w.add_tensor("logits.last", logits.numpy())

        # ---- layer-0 manual intermediates, straight from the state dict ----
        sd = model.state_dict()
        layer = "model.layers.0."
        D = cfg_json.get("head_dim") or cfg_json["hidden_size"] // cfg_json["num_attention_heads"]
        qh, kvh = cfg_json["num_attention_heads"], cfg_json["num_key_value_heads"]
        x = model.model.embed_tokens(input_ids)[0].float()
        T = x.shape[0]
        pos = torch.arange(T)

        u = rms_norm(x, sd[layer + "input_layernorm.weight"].float(), cfg_json["rms_norm_eps"])
        w.add_tensor("layer0.norm1", u.numpy())
        q = u @ sd[layer + "self_attn.q_proj.weight"].float().T
        k = u @ sd[layer + "self_attn.k_proj.weight"].float().T
        v = u @ sd[layer + "self_attn.v_proj.weight"].float().T
        w.add_tensor("layer0.q_pre", q.numpy())
        w.add_tensor("layer0.k_pre", k.numpy())
        w.add_tensor("layer0.v", v.numpy())

        inv_freq = llama3_inv_freq(D, cfg_json["rope_theta"], cfg_json.get("rope_scaling"))
        q_rot = apply_rope(q, pos, inv_freq)
        k_rot = apply_rope(k, pos, inv_freq)
        w.add_tensor("layer0.q_rot", q_rot.numpy())
        w.add_tensor("layer0.k_rot", k_rot.numpy())

        attn = causal_attention(
            q_rot.view(T, qh, D), k_rot.view(T, kvh, D), v.view(T, kvh, D), kvh
        )
        w.add_tensor("layer0.attn_out", attn.reshape(T, qh * D).numpy())
        o = attn.reshape(T, qh * D) @ sd[layer + "self_attn.o_proj.weight"].float().T
        r = x + o
        w.add_tensor("layer0.after_attn", r.numpy())

        u2 = rms_norm(r, sd[layer + "post_attention_layernorm.weight"].float(), cfg_json["rms_norm_eps"])
        w.add_tensor("layer0.norm2", u2.numpy())
        g = silu_mul(u2 @ sd[layer + "mlp.gate_proj.weight"].float().T,
                     u2 @ sd[layer + "mlp.up_proj.weight"].float().T)
        w.add_tensor("layer0.swiglu", g.numpy())
        x1 = r + g @ sd[layer + "mlp.down_proj.weight"].float().T
        w.add_tensor("layer0.output", x1.numpy())

        # greedy continuation
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=model.generation_config.pad_token_id,
        )[0].tolist()
    new_ids = gen[len(ids):]
    w.tensors["generated_ids"] = (np.array(new_ids, dtype="<i4").tobytes(), [len(new_ids)])
    return w, ids, new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)

    ops = build_op_fixtures(cfg)
    ops.write(os.path.join(args.output_dir, "ops_fixtures.pack"))
    print("wrote ops_fixtures.pack")

    model_pack, ids, new_ids = build_model_fixtures(args.model_dir, cfg, args.prompt, args.max_new_tokens)
    model_pack.write(os.path.join(args.output_dir, "model_fixtures.pack"))
    print(f"wrote model_fixtures.pack: {len(ids)} prompt tokens, {len(new_ids)} generated")

    import transformers

    meta = {
        "prompt": args.prompt,
        "prompt_ids": ids,
        "generated_ids": new_ids,
        "max_new_tokens": args.max_new_tokens,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "float32",
        "attention": "eager",
        "tf32": False,
    }
    with open(os.path.join(args.output_dir, "reference.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
