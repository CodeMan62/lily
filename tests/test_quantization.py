"""Phase 9 acceptance: online weight-only quantization.

One test covering, per method (w8a16, fp8, w4a16): output vs bf16 and storage
at the layer level, prefill logits vs the bf16 engine end to end, generation,
and config plumbing.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from lily.engine import Engine, EngineConfig
from lily.layers.quantization import FP8Config, W4A16Config, W8A16Config, build_quant_config
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

METHODS = {"w8a16": W8A16Config(), "fp8": FP8Config(), "w4a16": W4A16Config()}
# (max output MSE relative to the bf16 output's energy, storage ratio vs bf16, min logits cosine)
ACCURACY = {"w8a16": (2e-3, 0.55, 0.999), "fp8": (5e-3, 0.55, 0.995), "w4a16": (3e-2, 0.35, 0.95)}
# w4 is RTN g128, its error compounds over layers, hence the looser cosine

N, K = 64, 256


def make_quantized_layer(name: str, ref_weight: torch.Tensor) -> nn.Module:
    """Layer with `ref_weight` loaded and quantized, on the GPU."""
    method = METHODS[name].get_quant_method(layer=None, prefix="")
    layer = nn.Module()
    layer.input_size = K
    method.create_weights(layer, K, N, torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(ref_weight)
        method.process_weights_after_loading(layer)
    return layer.cuda()


def test_quantization():
    # ---- layer level -------------------------------------------------------
    gen = torch.Generator().manual_seed(0)
    ref_weight = (torch.randn(N, K, generator=gen) * 0.3).bfloat16()
    layers = {}
    for name in METHODS:
        layer = make_quantized_layer(name, ref_weight)
        x = (torch.randn(8, K, generator=gen)).bfloat16().cuda()
        out = METHODS[name].get_quant_method(layer=None, prefix="").apply(layer, x)
        ref_out = F.linear(x, ref_weight.cuda())
        acc, storage_max, _ = ACCURACY[name]
        rel_err = ((out - ref_out) ** 2).mean().item() / ref_out.pow(2).mean().item()
        assert rel_err < acc, f"{name} relative MSE {rel_err:.2e}"
        bytes_ = layer.weight.numel() * layer.weight.element_size()
        scale = getattr(layer, "weight_scale", None)
        if scale is not None:
            bytes_ += scale.numel() * scale.element_size()
        ratio = bytes_ / (2 * N * K)
        assert ratio < storage_max, f"{name} stores {ratio:.2f}x bf16"
        layers[name] = layer
    del layers
    free_cuda()

    # ---- engine level -------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(SMALL_LLAMA)
    prompt_ids = tok.encode("The history of natural language processing began in the 1950s.")[:32]
    engine_cfg = dict(context_length=512, max_running_requests=1, max_total_tokens=2048)

    ref = Engine(EngineConfig(SMALL_LLAMA, dtype="bfloat16", **engine_cfg))
    ref_logits = ref.prefill_logits(prompt_ids).cpu()
    del ref
    free_cuda()

    for name in METHODS:
        eng = Engine(EngineConfig(SMALL_LLAMA, dtype="bfloat16", quantization=name, **engine_cfg))
        quant_logits = eng.prefill_logits(prompt_ids).cpu()
        sim = F.cosine_similarity(ref_logits.float(), quant_logits.float(), dim=-1).item()
        assert sim > ACCURACY[name][2], f"{name} logits cosine {sim:.5f}"
        if name == "w4a16":
            text = eng.generate_text("Write one sentence about the sea.", max_new_tokens=16)
            assert len(text.strip()) > 0
        del eng
        free_cuda()

    # ---- plumbing -----------------------------------------------------------
    assert build_quant_config(None, SimpleNamespace(quantization_config=None)) is None
    try:
        build_quant_config(None, SimpleNamespace(quantization_config={"quant_method": "gptq"}))
        raise AssertionError("expected NotImplementedError for pre-quantized checkpoints")
    except NotImplementedError:
        pass
    assert build_quant_config("w4a16").get_name() == "w4a16"
    try:
        ServerArgs(quantization="awq")
        raise AssertionError("expected ValueError for unknown quantization")
    except ValueError:
        pass
