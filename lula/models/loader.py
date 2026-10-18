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

from lula.models.model_config import ModelConfig
from lula.models.registry import ModelRegistry
from lula.utils import set_default_torch_dtype

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

def safetensors_weights_iterator(model_dir: str) -> Iterator[Tuple[str, torch.Tensor]]:
    """Stream (name, tensor) pairs shard by shard; tensors are CPU views until copied."""
    for path in _safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                yield name, sf.get_tensor(name)

def get_model(
    model_config: ModelConfig,
    device: torch.device | str = "cuda",
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> nn.Module:
    dtype = dtype or model_config.dtype
    model_cls = ModelRegistry.resolve(model_config.architectures)
    t0 = time.time()
    with set_default_torch_dtype(dtype), torch.device(device):
        model = model_cls(model_config)
    loaded, unexpected = model.load_weights(safetensors_weights_iterator(model_config.model_path))
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
