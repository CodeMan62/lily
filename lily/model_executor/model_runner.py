"""model runner"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from lily.distributed import all_reduce_min_int, init_distributed_environment
from lily.layers.quantization import build_quant_config
from lily.models.model_config import ModelConfig
from lily.layers.attention.torch_native_backend import TorchNativeAttnBackend
from lily.layers.attention.triton_backend import TritonAttnBackend
from lily.layers.logits_processor import LogitsProcessorOutput
from lily.mem_cache.allocator import TokenToKVPoolAllocator
from lily.mem_cache.memory_pool import HybridTokenToKVPool, MHATokenToKVPool, MambaStatePool, ReqToTokenPool
from lily.model_executor.cuda_graph_runner import CudaGraphRunner
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.loader import get_model
from lily.server_args import ServerArgs

logger = logging.getLogger(__name__)

_GB = 1 << 30


def get_available_gpu_memory(device: torch.device) -> float:
    """Free GPU memory in GB as seen by the driver (includes other processes)."""
    torch.cuda.synchronize(device)
    free, _ = torch.cuda.mem_get_info(device)
    return free / _GB


def get_total_gpu_memory(device: torch.device) -> float:
    _, total = torch.cuda.mem_get_info(device)
    return total / _GB


class ModelRunner:
    def __init__(self, model_config: ModelConfig, server_args: ServerArgs, device: str = "cuda", tp_rank: int = 0):
        self.model_config = model_config
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.device = torch.device(device)
        if self.tp_size > 1 and self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", server_args.base_gpu_id + tp_rank)
        init_distributed_environment(self.tp_size, tp_rank, server_args.dist_init_addr, self.device)
        self.num_attention_heads = model_config.get_num_attention_heads(self.tp_size)
        self.num_kv_heads = model_config.get_num_kv_heads(self.tp_size)
        self.dtype = model_config.dtype
        self.kv_cache_dtype = self._resolve_kv_dtype(server_args.kv_cache_dtype)
        if server_args.random_seed is not None:
            torch.manual_seed(server_args.random_seed)

        self.mem_fraction_static = server_args.mem_fraction_static or 0.85
        total_gpu_memory = get_total_gpu_memory(self.device)

        self.quant_config = build_quant_config(server_args.quantization, model_config)
        if self.quant_config is not None:
            logger.info("weights quantized online with %s", self.quant_config.get_name())
        self.model = get_model(model_config, device=self.device, dtype=self.dtype, quant_config=self.quant_config)
        self.init_memory_pool(total_gpu_memory, server_args.max_running_requests, server_args.max_total_tokens)
        self.attn_backend = self._init_attention_backend()
        self.cuda_graph_runner = self._init_cuda_graph_runner()

    # ---- memory ------------------------------------------------------------

    def _resolve_kv_dtype(self, name: str) -> torch.dtype:
        if name == "auto":
            return self.dtype
        if name in ("fp8_e5m2",):
            return torch.float8_e5m2
        if name in ("fp8_e4m3",):
            return torch.float8_e4m3fn
        raise ValueError(f"unknown kv_cache_dtype {name!r}")

    def kv_cell_size_bytes(self) -> int:
        cfg = self.model_config
        return MHATokenToKVPool.cell_size_bytes(
            self.num_kv_heads, cfg.head_dim, len(cfg.full_attention_layer_ids), self.kv_cache_dtype
        )

    def profile_max_num_token(self, total_gpu_memory: float) -> int:
        available = get_available_gpu_memory(self.device)
        rest_memory = available - total_gpu_memory * (1 - self.mem_fraction_static)
        max_num_token = int(rest_memory * _GB // self.kv_cell_size_bytes())
        logger.info(
            "memory: total %.2f GB, free after load %.2f GB, mem_fraction_static %.2f -> %.2f GB for KV, %d tokens",
            total_gpu_memory, available, self.mem_fraction_static, rest_memory, max_num_token,
        )
        return max_num_token

    def init_memory_pool(self, total_gpu_memory: float, max_num_reqs: Optional[int], max_total_tokens: Optional[int]):
        # All ranks must agree on the pool size: the scheduler on every rank
        # replays the same allocations and expects the same free lists.
        self.max_total_num_tokens = all_reduce_min_int(self.profile_max_num_token(total_gpu_memory))
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logger.warning("max_total_tokens=%d exceeds profiled %d; capping", max_total_tokens, self.max_total_num_tokens)
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)
        if self.max_total_num_tokens <= 0:
            raise RuntimeError("Not enough memory for the KV cache. Lower mem_fraction_static or use a smaller model.")

        if max_num_reqs is None:
            max_num_reqs = min(max(self.max_total_num_tokens // 2, 1), 512)
        self.max_running_requests = max_num_reqs

        cfg = self.model_config
        if cfg.is_hybrid:
            # Linear-attention state is tied 1:1 to request slots and cannot be
            # prefix-cached, so the radix cache must stay off for hybrid models.
            if self.server_args.disable_radix_cache is not True:
                logger.warning("hybrid linear-attention model: disabling radix cache (states are not prefix-cacheable)")
                self.server_args.disable_radix_cache = True
            max_num_reqs = self._cap_requests_for_mamba(max_num_reqs, total_gpu_memory)

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1, max_context_len=cfg.context_len + 4, device=self.device
        )
        full_layer_ids = cfg.full_attention_layer_ids
        if cfg.is_hybrid:
            self.token_to_kv_pool = HybridTokenToKVPool(
                full_layer_ids,
                size=self.max_total_num_tokens,
                page_size=1,
                dtype=self.kv_cache_dtype,
                head_num=self.num_kv_heads,
                head_dim=cfg.head_dim,
                device=self.device,
            )
            # Mamba states are allocated per request slot, independent of tokens.
            self.mamba_pool = MambaStatePool.for_model_config(cfg, size=max_num_reqs + 1, device=self.device)
            self.token_to_kv_pool.mamba_pool = self.mamba_pool
        else:
            self.mamba_pool = None
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=1,
                dtype=self.kv_cache_dtype,
                head_num=self.num_kv_heads,
                head_dim=cfg.head_dim,
                layer_num=cfg.num_hidden_layers,
                device=self.device,
            )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            self.max_total_num_tokens, dtype=self.kv_cache_dtype, device=self.device, kvcache=self.token_to_kv_pool
        )
        logger.info("pools: max_total_num_tokens=%d max_running_requests=%d", self.max_total_num_tokens, max_num_reqs)

    def _cap_requests_for_mamba(self, max_num_reqs: int, total_gpu_memory: float) -> int:
        """Bound concurrent requests so the per-request mamba states fit in a
        share of the memory left after weights, alongside the token KV pool."""
        available = get_available_gpu_memory(self.device)
        rest_memory = available - total_gpu_memory * (1 - self.mem_fraction_static)
        if self.server_args.max_running_requests is not None:
            num_reqs = max_num_reqs
        else:
            mamba_bytes = MambaStatePool.per_slot_bytes_for_model_config(self.model_config)
            mamba_budget = rest_memory * 0.25
            num_reqs = max(2, min(512, int(mamba_budget / max(mamba_bytes, 1))))
        if num_reqs < max_num_reqs:
            logger.warning(
                "hybrid model: capping max_running_requests %d -> %d for mamba state memory", max_num_reqs, num_reqs
            )
        return num_reqs

    # ---- backend -----------------------------------------------------------

    def _init_attention_backend(self):
        name = self.server_args.attention_backend
        if name == "triton":
            return TritonAttnBackend(self)
        if name == "torch_native":
            return TorchNativeAttnBackend(self)
        raise ValueError(f"unknown attention backend {name!r}")

    # ---- CUDA graphs -------------------------------------------------------

    def _init_cuda_graph_runner(self):
        if self.server_args.disable_cuda_graph or not torch.cuda.is_available():
            return None
        if not getattr(self.attn_backend, "supports_cuda_graph", lambda: True)():
            logger.info("CUDA graphs disabled: backend %s is not capture-safe", type(self.attn_backend).__name__)
            return None
        try:
            return CudaGraphRunner(self)
        except Exception as exc:
            logger.warning("CUDA graph capture failed; falling back to eager: %s", exc)
            return None

    # ---- forward -----------------------------------------------------------

    def _attach(self, forward_batch: ForwardBatch) -> None:
        forward_batch.attn_backend = self.attn_backend
        forward_batch.req_to_token_pool = self.req_to_token_pool
        forward_batch.token_to_kv_pool = self.token_to_kv_pool
        self.attn_backend.init_forward_metadata(forward_batch)

    @torch.inference_mode()
    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if self.cuda_graph_runner is not None and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)
        self._attach(forward_batch)
        return self.model(forward_batch.input_ids, forward_batch.positions, forward_batch)
