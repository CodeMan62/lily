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

# Default packing factors for pre-quantized checkpoints that store extra tensors
# per layer (uint8-packed NVFP4): a K-dim slice must be scaled by the packing factor.
_QUANT_FORMAT_FACTORS = {"weight_packed": 2, "weight_scale": 16}


SKIP = "skip"   # tp_plan verdict: another rank owns this tensor (expert parallelism)


def read_tp_shard(sf, name: str, num_kv_heads: int, pack_factors: Optional[dict] = None,
                  plan=None) -> Optional[torch.Tensor]:
    """This rank's slice of a checkpoint tensor. With `plan(name, shape)` (a model's `tp_plan`)
    the model decides: None = replicated, SKIP = not this rank's, (dim, start, stop) = slice.
    Without one, the Llama-style name patterns below apply."""
    tp_size, tp_rank = get_tp_info()
    if tp_size == 1:
        return sf.get_tensor(name)
    if plan is not None:
        sl = sf.get_slice(name)
        spec = plan(name, tuple(sl.get_shape()))
        if spec is None:
            return sf.get_tensor(name)
        if spec == SKIP:
            return None
        dim, start, stop = spec
        return sl[start:stop] if dim == 0 else sl[:, start:stop]
    if name.endswith("weight_global_scale"):  # per-tensor scalar: replicated on every rank
        return sf.get_tensor(name)
    if any(key in name for key in _ROW_PARALLEL):
        dim = 1
        factors = pack_factors or _QUANT_FORMAT_FACTORS
        pack = next((f for suffix, f in factors.items() if name.endswith(suffix)), 1)
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


def safetensors_weights_iterator(model_dir: str, num_kv_heads: int, pack_factors: Optional[dict] = None,
                                 plan=None) -> Iterator[Tuple[str, torch.Tensor]]:
    """Stream (name, tensor) pairs shard by shard, each already cut to this TP rank's slice."""
    for path in _safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                t = read_tp_shard(sf, name, num_kv_heads, pack_factors, plan)
                if t is not None:
                    yield name, t

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
    loaded, unexpected = model.load_weights(safetensors_weights_iterator(
        model_config.model_path, model_config.num_key_value_heads,
        pack_factors=getattr(quant_config, "tp_pack_factors", None), plan=getattr(model, "tp_plan", None)))
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # the KV pool is sized from free memory next
    model.eval()
    logger.info("loaded %s in %.1fs (%d tensors)", model_cls.__name__, time.time() - t0, len(loaded))
    return model
