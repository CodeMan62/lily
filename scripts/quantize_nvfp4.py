#!/usr/bin/env python
"""Export an HF checkpoint as compressed-tensors NVFP4 (`nvfp4-pack-quantized`), the
format lily's NVFP4Config loads and ModelOpt/llm-compressor produce.

    uv run python scripts/quantize_nvfp4.py Qwen/Qwen3-0.6B /path/out [--include-lm-head]

Per Linear weight W [N, K] (K % 16 == 0), matching base.py's dequant
`e2m1 * block_scale / global_scale`:
    global_scale = 448 * 6 / amax(|W|)                      fp32 [1]
    block_scale  = e4m3(amax(|block|) * global_scale / 6)   [N, K/16]
    code         = round_to_e2m1(W * global_scale / block_scale)
Codes are packed two per byte, low nibble first. Embeddings, lm_head (unless
--include-lm-head), norms and non-Linear tensors are copied through in their
original dtype. This is plain RTN (no calibration), good enough for engine
testing; use ModelOpt / llm-compressor for production checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from lily.layers.quantization.base import nvfp4_quantize

GROUP = 16

# Linear weights that get quantized; everything else is copied through.
LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


def quantize_nvfp4(w: torch.Tensor):
    """bf16/fp32 [N, K] -> (packed uint8 [N, K/2], scale e4m3 [N, K/16], global_scale fp32 [1]) on CPU."""
    packed, scale, gs = nvfp4_quantize(w.cuda() if torch.cuda.is_available() else w)
    return packed.cpu(), scale.cpu(), gs.cpu()


def is_linear_weight(name: str, lm_head: bool) -> bool:
    if not name.endswith(".weight"):
        return False
    if name.startswith("lm_head."):
        return lm_head
    stem = name[: -len(".weight")].rsplit(".", 1)[-1]
    return stem in LINEAR_SUFFIXES and ".visual." not in name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--include-lm-head", action="store_true", help="also quantize lm_head (saves a 2.5 GB bf16 read per token on 248k vocabs)")
    args = ap.parse_args()

    src = args.model if os.path.isdir(args.model) else snapshot_download(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    assert shards, f"no safetensors in {src}"

    ignore = ["lm_head"] if not args.include_lm_head else []
    out_tensors = {}
    n_quant = 0
    for shard in shards:
        with safe_open(os.path.join(src, shard), framework="pt") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                if is_linear_weight(name, args.include_lm_head):
                    base = name[: -len(".weight")]
                    packed, scale, gs = quantize_nvfp4(t)
                    out_tensors[f"{base}.weight_packed"] = packed
                    out_tensors[f"{base}.weight_scale"] = scale
                    out_tensors[f"{base}.weight_global_scale"] = gs
                    n_quant += 1
                else:
                    out_tensors[name] = t
    save_file(out_tensors, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "symmetric": True,
                        "dynamic": False, "group_size": GROUP},
            "input_activations": None, "output_activations": None}},
        "format": "nvfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "ignore": ignore,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in os.listdir(src):
        if fn.endswith((".json", ".txt", ".jinja", ".model")) and fn != "config.json":
            shutil.copy(os.path.join(src, fn), os.path.join(args.out_dir, fn))
    size = os.path.getsize(os.path.join(args.out_dir, "model.safetensors")) / 2 ** 20
    print(f"quantized {n_quant} Linear weights -> {args.out_dir} ({size:.0f} MiB)")


if __name__ == "__main__":
    main()
