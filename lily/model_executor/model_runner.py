"""model runner"""
from __future__ import annotations

import logging
import os
import time
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
        init_distributed_environment(self.tp_size, tp_rank, server_args.dist_init_addr, self.device,
                                     backend=server_args.dist_backend)
        self.num_attention_heads = model_config.get_num_attention_heads(self.tp_size)
        self.num_kv_heads = model_config.get_num_kv_heads(self.tp_size)
        self.dtype = model_config.dtype
        self.kv_cache_dtype = self._resolve_kv_dtype(server_args.kv_cache_dtype)
        if server_args.random_seed is not None:
            torch.manual_seed(server_args.random_seed)

        self.mem_fraction_static = server_args.mem_fraction_static or 0.85
        total_gpu_memory = get_total_gpu_memory(self.device)

        self.quant_config = build_quant_config(server_args.quantization, model_config)
        model_config.lm_head_quantization = server_args.quantize_lm_head
        model_config.enable_mtp = (
            server_args.speculative_algorithm in ("NEXTN", "MTP") and model_config.mtp_num_hidden_layers >= 1)
        if server_args.speculative_algorithm in ("NEXTN", "MTP") and not model_config.enable_mtp:
            raise ValueError("--speculative-algorithm MTP needs a checkpoint with an MTP head (mtp_num_hidden_layers >= 1)")
        if self.quant_config is not None:
            logger.info("weights quantized online with %s", self.quant_config.get_name())
        self.model = get_model(model_config, device=self.device, dtype=self.dtype, quant_config=self.quant_config)
        self.drafter = None
        if server_args.speculative_algorithm == "DFLASH":
            from lily.models.dflash import DFlash2Drafter, resolve_draft_path

            self.drafter = DFlash2Drafter.load(resolve_draft_path(server_args.speculative_draft_model_path),
                                               model_config.hidden_size, self.device)
            if server_args.speculative_num_steps > self.drafter.num_drafts:
                logger.warning("DFLASH drafts a block of %d: speculative_num_steps %s -> %d",
                               self.drafter.block_size, server_args.speculative_num_steps, self.drafter.num_drafts)
                server_args.speculative_num_steps = self.drafter.num_drafts
                server_args.speculative_num_draft_tokens = self.drafter.num_drafts + 1
            logger.info("DFlash2 drafter: %d layers, block %d, context from target layers %s",
                        self.drafter.cfg.num_hidden_layers, self.drafter.block_size, self.drafter.cfg.target_layer_ids)
        self.init_memory_pool(total_gpu_memory, server_args.max_running_requests, server_args.max_total_tokens)
        self.attn_backend = self._init_attention_backend()
        self.mtp_kv_pool = self.mtp_attn_backend = None
        self.dflash_kv_pool = None
        if self.drafter is not None:
            dc = self.drafter.cfg
            # one context entry per cached target position, same slots as the target
            self.dflash_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens, page_size=1, dtype=self.dtype, head_num=dc.num_key_value_heads,
                head_dim=dc.head_dim, layer_num=dc.num_hidden_layers, device=self.device)
        if getattr(self.model, "mtp", None) is not None:
            # The draft layer keeps its own K/V, indexed by the same KV slots as the
            # target so allocation, req_to_token rows and rollback are shared.
            make_mtp_pool = getattr(self.model, "make_mtp_pool", None)
            if make_mtp_pool is not None:
                self.mtp_kv_pool = make_mtp_pool(self.max_total_num_tokens, self.device, self.kv_cache_dtype)
            else:
                self.mtp_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens, page_size=1, dtype=self.kv_cache_dtype, head_num=self.num_kv_heads,
                    head_dim=model_config.head_dim, layer_num=1, device=self.device,
                    start_layer=model_config.num_hidden_layers)
            if self.attn_backend is not None:
                self.mtp_attn_backend = TritonAttnBackend(self, kv_pool=self.mtp_kv_pool) \
                    if server_args.attention_backend == "triton" else TorchNativeAttnBackend(self, kv_pool=self.mtp_kv_pool)
        self.cuda_graph_runner = self._init_cuda_graph_runner()
        self._warmup()

    def _warmup(self) -> None:
        """One throwaway prefill (and MTP extend) so the Triton kernels the decode graph
        does not cover (extend attention, the M > 4 dequant GEMM) compile now instead
        of on the first user request, which otherwise pays 10+ s of time to first token."""
        if self.device.type != "cuda" or not torch.cuda.is_available() or os.environ.get("LILY_DISABLE_WARMUP"):
            return
        t0 = time.time()
        n = 16
        slot = self.req_to_token_pool.alloc(1)
        loc = self.token_to_kv_pool_allocator.alloc(n)
        if slot is None or loc is None:
            return
        slot = slot[0]
        try:
            self.req_to_token_pool.write((slot, slice(0, n)), loc.to(torch.int32))
            ids = [min(10, self.model_config.vocab_size - 1)] * n
            fb = ForwardBatch.for_extend([ids], [0], [slot], self.device, out_cache_loc=loc)
            fb.capture_hidden_states = self.mtp_kv_pool is not None
            out = self.forward(fb)
            if self.mtp_kv_pool is not None:
                fbm = ForwardBatch.for_extend([ids], [0], [slot], self.device, out_cache_loc=loc)
                self.forward_mtp(fbm, out.hidden_states)
            torch.cuda.synchronize(self.device)
        finally:
            self.token_to_kv_pool_allocator.free(loc)
            self.req_to_token_pool.free(slot)
        logger.info("warmup prefill done in %.1fs", time.time() - t0)

    # ---- memory ------------------------------------------------------------

    def _resolve_kv_dtype(self, name: str) -> torch.dtype:
        if name == "auto":
            return self.dtype
        if name in ("fp8_e5m2",):
            return torch.float8_e5m2
        if name in ("fp8_e4m3",):
            return torch.float8_e4m3fn
        raise ValueError(f"unknown kv_cache_dtype {name!r}")

    def _resolve_mamba_dtype(self) -> torch.dtype:
        name = self.server_args.mamba_ssm_dtype
        if name == "auto":
            return self.model_config.mamba_state_dtype
        return torch.float32 if name == "float32" else torch.bfloat16

    def kv_cell_size_bytes(self) -> int:
        cfg = self.model_config
        custom = getattr(self.model, "token_cell_bytes", None)   # models with their own cache layout (MLA)
        if custom is not None:
            cell = custom(self.kv_cache_dtype)
        else:
            layers = len(cfg.full_attention_layer_ids) + (1 if cfg.enable_mtp else 0)  # + the MTP draft layer
            cell = MHATokenToKVPool.cell_size_bytes(self.num_kv_heads, cfg.head_dim, layers, self.kv_cache_dtype)
        if self.drafter is not None:  # + the DFlash drafter's context entries
            dc = self.drafter.cfg
            cell += MHATokenToKVPool.cell_size_bytes(dc.num_key_value_heads, dc.head_dim, dc.num_hidden_layers, self.dtype)
        return cell

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
        make_pool = getattr(self.model, "make_token_to_kv_pool", None)
        if make_pool is not None:
            self.token_to_kv_pool = make_pool(self.max_total_num_tokens, self.device, self.kv_cache_dtype)
        elif cfg.is_hybrid:
            self.token_to_kv_pool = HybridTokenToKVPool(
                cfg.full_attention_layer_ids,
                size=self.max_total_num_tokens,
                page_size=1,
                dtype=self.kv_cache_dtype,
                head_num=self.num_kv_heads,
                head_dim=cfg.head_dim,
                device=self.device,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=1,
                dtype=self.kv_cache_dtype,
                head_num=self.num_kv_heads,
                head_dim=cfg.head_dim,
                layer_num=cfg.num_hidden_layers,
                device=self.device,
            )
        self.mamba_pool = None
        if cfg.is_hybrid:
            # Linear-attention states are allocated per request slot, independent of tokens.
            self.mamba_pool = MambaStatePool.for_model_config(
                cfg, size=max_num_reqs + 1, device=self.device, dtype=self._resolve_mamba_dtype())
            self.token_to_kv_pool.mamba_pool = self.mamba_pool
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
            mamba_bytes = MambaStatePool.per_slot_bytes_for_model_config(self.model_config, self._resolve_mamba_dtype())
            mamba_budget = rest_memory * 0.25
            num_reqs = max(2, min(512, int(mamba_budget / max(mamba_bytes, 1))))
        if num_reqs < max_num_reqs:
            logger.warning(
                "hybrid model: capping max_running_requests %d -> %d for mamba state memory", max_num_reqs, num_reqs
            )
        return num_reqs

    # ---- backend -----------------------------------------------------------

    def _init_attention_backend(self):
        if not getattr(self.model, "uses_attention_backend", True):
            return None    # the model runs its own attention (MLA over the latent pool)
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
        if not getattr(self.model, "supports_cuda_graph", True):
            logger.info("CUDA graphs disabled: %s decodes eagerly", type(self.model).__name__)
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
        if self.attn_backend is not None:
            self.attn_backend.init_forward_metadata(forward_batch)

    @torch.inference_mode()
    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if self.cuda_graph_runner is not None and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)
        self._attach(forward_batch)
        return self.model(forward_batch.input_ids, forward_batch.positions, forward_batch)

    @torch.inference_mode()
    def forward_mtp(self, forward_batch: ForwardBatch, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the MTP draft layer over `forward_batch` (its own KV pool, shared request
        rows). `hidden_states` [num_tokens, hidden] are the target's final hidden states
        paired with each input token. Returns the draft hidden states [num_tokens, hidden];
        logits come from `model.lm_head.compute_logits` on the rows the caller needs."""
        forward_batch.attn_backend = self.mtp_attn_backend
        forward_batch.req_to_token_pool = self.req_to_token_pool
        forward_batch.token_to_kv_pool = self.mtp_kv_pool
        if self.mtp_attn_backend is not None:
            self.mtp_attn_backend.init_forward_metadata(forward_batch)
        return self.model.mtp(forward_batch.input_ids, forward_batch.positions, hidden_states, forward_batch,
                              self.model.model.embed_tokens)
