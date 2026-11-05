"""Quantization plugin registry for the linear layers."""

from __future__ import annotations

from typing import Optional

from lily.layers.quantization.base import (
    CompressedInt4Config,
    LinearMethodBase,
    NVFP4Config,
    QuantizationConfig,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
    get_linear_method,
)
from lily.layers.quantization.weight_only import FP8Config, W4A16Config, W8A16Config

_QUANT_CONFIGS = {
    "fp8": FP8Config,
    "w8a16": W8A16Config,
    "w4a16": W4A16Config,
}
SUPPORTED_QUANTIZATION = tuple(_QUANT_CONFIGS)

# pre-quantized checkpoint formats: quant_method -> format -> config class
_CHECKPOINT_METHODS = {
    "compressed-tensors": {
        "nvfp4-pack-quantized": NVFP4Config,
        "pack-quantized": CompressedInt4Config,
    },
    # NVIDIA ModelOpt exports (e.g. nvidia/GLM-5.3-Flash-NVFP4): same NVFP4 layout, tensors named
    # weight / weight_scale / weight_scale_2 (remapped by the model loaders), keyed by `quant_algo`.
    "modelopt": {"NVFP4": NVFP4Config},
}


def build_quant_config(quantization: Optional[str], model_config=None) -> Optional[QuantizationConfig]:
    """Resolve the CLI flag / checkpoint `quantization_config` into a config, or None."""
    if quantization is not None:
        cls = _QUANT_CONFIGS.get(quantization)
        if cls is None:
            raise ValueError(
                f"Unknown quantization {quantization!r}; lily supports {', '.join(SUPPORTED_QUANTIZATION)} "
                "(online weight-only, applied to unquantized checkpoints at load time) "
                "or nvfp4 (also usable via the checkpoint's quantization_config)"
            )
        return cls.from_config({})
    checkpoint_cfg = getattr(model_config, "quantization_config", None) if model_config is not None else None
    if checkpoint_cfg:
        method = checkpoint_cfg.get("quant_method", "unknown")
        formats = _CHECKPOINT_METHODS.get(method, {})
        fmt = checkpoint_cfg.get("format") or checkpoint_cfg.get("quant_algo")
        cls = formats.get(fmt)
        if cls is None:
            raise NotImplementedError(
                f"quantized checkpoints are not supported (quant_method={method!r}, format={fmt!r}); "
                f"lily loads the pre-quantized formats {sorted(formats)} or applies online "
                f"{', '.join(SUPPORTED_QUANTIZATION)}: use the unquantized checkpoint and pass --quantization"
            )
        return cls.from_config(checkpoint_cfg)
    return None


__all__ = [
    "SUPPORTED_QUANTIZATION",
    "build_quant_config",
    "get_linear_method",
    "LinearMethodBase",
    "QuantizationConfig",
    "QuantizeMethodBase",
    "UnquantizedLinearMethod",
    "FP8Config",
    "W4A16Config",
    "W8A16Config",
    "NVFP4Config",
    "CompressedInt4Config",
]
