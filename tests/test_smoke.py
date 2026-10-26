import pytest
import torch

from lily.models.model_config import ModelConfig
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA, SMALL_QWEN3


def test_cuda_available():
    assert torch.cuda.is_available()


def test_server_args_defaults_and_validation():
    sa = ServerArgs.parse(["--model-path", "x", "--speculative-algorithm", "eagle3",
                           "--speculative-draft-model-path", "y"])
    assert sa.speculative_algorithm == "EAGLE3"
    assert sa.speculative_num_steps == 5 and sa.speculative_eagle_topk == 4
    assert sa.speculative_num_draft_tokens == 8

    sa = ServerArgs(model_path="x", speculative_algorithm="nextn")
    assert (sa.speculative_num_steps, sa.speculative_eagle_topk, sa.speculative_num_draft_tokens) == (1, 1, 2)

    with pytest.raises(ValueError):
        ServerArgs(model_path="x", speculative_algorithm="eagle3",
                   speculative_draft_model_path="y", speculative_num_steps=5,
                   speculative_num_draft_tokens=3)


def test_model_config_parses_small_models():
    cfg = ModelConfig.from_pretrained(SMALL_QWEN3)
    assert "Qwen3ForCausalLM" in cfg.architectures
    assert cfg.num_key_value_heads <= cfg.num_attention_heads
    assert cfg.head_dim > 0 and cfg.vocab_size > 0
    assert cfg.eos_token_ids

    cfg = ModelConfig.from_pretrained(SMALL_LLAMA)
    assert "LlamaForCausalLM" in cfg.architectures
    assert cfg.rotary_dim == cfg.head_dim
