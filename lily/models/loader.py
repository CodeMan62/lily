"""Weight loading. Mirrors sglang/srt/model_loader/loader.py (reduced)."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from typing import Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open
from torch import nn

from lily.distributed import get_tp_info
from lily.models.model_config import ModelConfig
from lily.layers.quantization.base import QuantizationConfig
from lily.models.registry import ModelRegistry
from lily.utils import ceil_div, set_default_torch_dtype

logger = logging.getLogger(__name__)

def _safetensors_files(model_dir: str) -> List[str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_dir, x) for x in files]
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {model_dir}")
    return files

# Checkpoint names whose tensors are split across TP ranks; everything else is replicated.
_ROW_PARALLEL = ("o_proj.weight", "down_proj.weight", "out_proj.weight")  # split dim 1 (input features)
_COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head",
                    "in_proj_z", "in_proj_b", "in_proj_a", "A_log", "dt_bias")  # split dim 0
_KV_PROJ = ("k_proj", "v_proj")

# Pre-quantized checkpoints store several tensors per layer (nvfp4), each packed
# along the K dim; a K-dim slice must be scaled by the packing factor.
_K_PACK_FACTORS = {"weight_packed": 2, "weight_scale": 16}


def read_tp_shard(sf, name: str, num_kv_heads: int) -> torch.Tensor:
    tp_size, tp_rank = get_tp_info()
    if tp_size == 1:
        return sf.get_tensor(name)
    if any(key in name for key in _ROW_PARALLEL):
        dim = 1
        pack = next((f for suffix, f in _K_PACK_FACTORS.items() if name.endswith(suffix)), 1)
    elif any(key in name for key in _COLUMN_PARALLEL):
        dim = 0
        pack = 1
    else:
        return sf.get_tensor(name)
    num_parts = min(tp_size, num_kv_heads) if any(key in name for key in _KV_PROJ) else tp_size
    part = tp_rank * num_parts // tp_size
    tensor = sf.get_slice(name)
    length = tensor.get_shape()[dim]
    part_size = ceil_div(length, num_parts)
    rows = slice(part * part_size * pack, min((part + 1) * part_size * pack, length))
    return tensor[slice(None) if dim == 0 else slice(None), rows] if pack != 1 else (
        tensor[rows] if dim == 0 else tensor[:, rows])


def safetensors_weights_iterator(model_dir: str, num_kv_heads: int) -> Iterator[Tuple[str, torch.Tensor]]:
    """Stream (name, tensor) pairs shard by shard, each already cut to this TP rank's slice."""
    for path in _safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                yield name, read_tp_shard(sf, name, num_kv_heads)

def get_model(
    model_config: ModelConfig,
    device: torch.device | str = "cuda",
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
    quant_config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    dtype = dtype or model_config.dtype
    model_cls = ModelRegistry.resolve(model_config.architectures)
    t0 = time.time()
    with set_default_torch_dtype(dtype), torch.device(device):
        model = model_cls(model_config, quant_config=quant_config)
    loaded, unexpected = model.load_weights(
        safetensors_weights_iterator(model_config.model_path, model_config.num_key_value_heads))
    missing = set(dict(model.named_parameters())) - loaded
    if unexpected:
        logger.warning("ignored %d unexpected checkpoint tensors, e.g. %s", len(unexpected), sorted(unexpected)[:5])
    if missing:
        msg = f"{len(missing)} parameters not found in checkpoint, e.g. {sorted(missing)[:5]}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    for m in model.modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None:
            qm.process_weights_after_loading(m)
    model.eval()
    logger.info("loaded %s in %.1fs (%d tensors)", model_cls.__name__, time.time() - t0, len(loaded))
    return model
