"""Server arguments. Mirrors sglang/srt/server_args.py.

Every runtime knob lives here as a dataclass field with a matching CLI flag.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from lily.distributed import DEFAULT_DIST_INIT_ADDR
from lily.layers.quantization import SUPPORTED_QUANTIZATION


@dataclass
class ServerArgs:
    # Model
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    context_length: Optional[int] = None
    revision: Optional[str] = None

    # Backends / quantization
    attention_backend: str = "triton"  # triton | torch_native
    quantization: Optional[str] = None  # fp8 | w8a16 | w4a16 | nf12 (weight-only, online at load time; nf12 is lossless, needs `pip install lily[nf12]`)
    kv_cache_dtype: str = "auto"  # auto | fp8_e4m3
    mamba_ssm_dtype: str = "auto"  # auto (checkpoint config, else float32) | float32 | bfloat16
    quantize_lm_head: Optional[str] = None  # nvfp4: quantize a bf16 lm_head at load (248k-vocab heads read 2.5 GB/token in bf16)
    quantize_draft: Optional[str] = None  # nvfp4: quantize the bf16 MTP head at load (it runs k times per round)

    # Memory
    mem_fraction_static: Optional[float] = None
    max_total_tokens: Optional[int] = None
    max_running_requests: Optional[int] = None
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int = 8192

    # Scheduling
    schedule_policy: str = "fcfs"  # fcfs | lpm
    disable_radix_cache: bool = False
    disable_overlap_schedule: bool = False
    stream_interval: int = 1

    # CUDA graphs
    disable_cuda_graph: bool = False
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None

    # Serving
    host: str = "127.0.0.1"
    port: int = 30000
    served_model_name: Optional[str] = None
    reasoning_parser: Optional[str] = None  # qwen3
    log_level: str = "info"
    random_seed: Optional[int] = None

    # Speculative decoding
    # MTP: the checkpoint's own multi-token-prediction head (NEXTN is an alias).
    # DFLASH: a DFlash 2 block-diffusion drafter from --speculative-draft-model-path.
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_match_window_size: int = 12

    # Device
    device: str = "cuda"

    # Parallelism
    tp_size: int = 1
    base_gpu_id: int = 0
    dist_init_addr: str = DEFAULT_DIST_INIT_ADDR  # host:port for the torch.distributed TCP store
    dist_backend: str = "nccl"                    # "gloo" runs several ranks on one GPU (tests)

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.speculative_algorithm is not None:
            algo = self.speculative_algorithm.upper()
            if algo == "NEXTN":
                algo = "MTP"
            if algo not in ("MTP", "DFLASH"):
                raise ValueError(f"Unknown speculative algorithm {algo!r}; lily supports MTP and DFLASH")
            self.speculative_algorithm = algo
            if algo == "DFLASH" and not self.speculative_draft_model_path:
                raise ValueError("DFLASH requires --speculative-draft-model-path (e.g. z-lab/Qwen3.8-27B-DFlash2)")
            # `num_steps` draft tokens per round, verified together with the bonus
            # position (num_steps + 1 target rows). DFLASH drafts a whole block at once;
            # its default is the drafter's block size - 1, resolved when it loads.
            if self.speculative_num_steps is None:
                self.speculative_num_steps = 3 if algo == "MTP" else 7
            if self.speculative_eagle_topk is None:
                self.speculative_eagle_topk = 1
            if self.speculative_eagle_topk != 1:
                raise ValueError("MTP speculative decoding drafts a single chain (speculative_eagle_topk must be 1)")
            if self.speculative_num_draft_tokens is None:
                if self.speculative_eagle_topk == 1:
                    self.speculative_num_draft_tokens = self.speculative_num_steps + 1
                else:
                    self.speculative_num_draft_tokens = 8
            if self.speculative_num_draft_tokens < self.speculative_num_steps + 1:
                raise ValueError(
                    "speculative_num_draft_tokens must be >= speculative_num_steps + 1"
                )
            if self.speculative_eagle_topk == 1 and (
                self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                raise ValueError(
                    "with topk=1, speculative_num_draft_tokens must equal speculative_num_steps + 1"
                )

        if self.mem_fraction_static is not None and not (0 < self.mem_fraction_static < 1):
            raise ValueError("mem_fraction_static must be in (0, 1)")
        if self.schedule_policy not in ("fcfs", "lpm"):
            raise ValueError(f"Unknown schedule policy {self.schedule_policy!r}")
        if self.attention_backend not in ("triton", "torch_native"):
            raise ValueError(f"Unknown attention backend {self.attention_backend!r}")
        if self.tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if self.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"Unknown dtype {self.dtype!r}")
        if self.quantize_lm_head is not None and self.quantize_lm_head != "nvfp4":
            raise ValueError(f"Unknown quantize_lm_head {self.quantize_lm_head!r}; supported: nvfp4")
        if self.quantize_draft is not None and self.quantize_draft != "nvfp4":
            raise ValueError(f"Unknown quantize_draft {self.quantize_draft!r}; supported: nvfp4")
        if self.mamba_ssm_dtype not in ("auto", "float32", "bfloat16"):
            raise ValueError(f"Unknown mamba_ssm_dtype {self.mamba_ssm_dtype!r}")
        if self.quantization is not None and self.quantization not in SUPPORTED_QUANTIZATION:
            raise ValueError(f"Unknown quantization {self.quantization!r}; supported: {', '.join(SUPPORTED_QUANTIZATION)}")

    # ---- CLI binding -------------------------------------------------------

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        for f in dataclasses.fields(ServerArgs):
            flag = "--" + f.name.replace("_", "-")
            default = f.default if f.default is not dataclasses.MISSING else None
            if f.type in ("bool", bool) or isinstance(default, bool):
                parser.add_argument(flag, action="store_true", default=default)
            elif f.name == "cuda_graph_bs":
                parser.add_argument(flag, type=int, nargs="+", default=default)
            elif isinstance(default, int):
                parser.add_argument(flag, type=int, default=default)
            elif isinstance(default, float):
                parser.add_argument(flag, type=float, default=default)
            elif default is None and f.type in ("Optional[int]",):
                parser.add_argument(flag, type=int, default=None)
            elif default is None and f.type in ("Optional[float]",):
                parser.add_argument(flag, type=float, default=None)
            else:
                parser.add_argument(flag, type=str, default=default)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ServerArgs":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in vars(args).items() if k in names})

    @classmethod
    def parse(cls, argv: Optional[List[str]] = None) -> "ServerArgs":
        parser = argparse.ArgumentParser("lily")
        cls.add_cli_args(parser)
        return cls.from_cli_args(parser.parse_args(argv))
