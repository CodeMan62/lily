"""all distributed utils"""

from __future__ import annotations

import pickle
from typing import Any, List, Optional

import torch
import torch.distributed as dist

DEFAULT_DIST_INIT_ADDR = "127.0.0.1:2333"

_TP_SIZE = 1
_TP_RANK = 0
_BACKEND = "nccl"
_DEVICE_GROUP: Optional[dist.ProcessGroup] = None
_CPU_GROUP: Optional[dist.ProcessGroup] = None


def init_distributed_environment(
    tp_size: int,
    tp_rank: int,
    dist_init_addr: str = DEFAULT_DIST_INIT_ADDR,
    device: Optional[torch.device] = None,
    backend: str = "nccl",
) -> None:
    """`backend` "gloo" lets several ranks share one GPU (tests of the sharding logic)."""
    global _TP_SIZE, _TP_RANK, _DEVICE_GROUP, _CPU_GROUP, _BACKEND
    _TP_SIZE, _TP_RANK, _BACKEND = tp_size, tp_rank, backend
    if tp_size == 1:
        return
    if device is not None and device.type == "cuda":
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{dist_init_addr}",
            world_size=tp_size,
            rank=tp_rank,
        )
    _DEVICE_GROUP = dist.group.WORLD
    _CPU_GROUP = dist.new_group(list(range(tp_size)), backend="gloo")


def destroy_distributed_environment() -> None:
    global _TP_SIZE, _TP_RANK, _DEVICE_GROUP, _CPU_GROUP
    if dist.is_initialized():
        dist.destroy_process_group()
    _TP_SIZE, _TP_RANK, _DEVICE_GROUP, _CPU_GROUP = 1, 0, None, None


def get_tp_info() -> tuple[int, int]:
    """(tp_size, tp_rank) of this process; (1, 0) before distributed init."""
    return _TP_SIZE, _TP_RANK


# ---- activation collectives (NCCL) ----------------------------------------


def tensor_model_parallel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    if _TP_SIZE == 1:
        return x
    dist.all_reduce(x, group=_DEVICE_GROUP)
    return x


def tensor_model_parallel_all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Concatenate each rank's `x` along `dim` (rank order)."""
    if _TP_SIZE == 1:
        return x
    dim = dim % x.dim()
    x = x.contiguous()
    out = torch.empty((_TP_SIZE,) + tuple(x.shape), dtype=x.dtype, device=x.device)
    if _BACKEND == "gloo":      # no all_gather_into_tensor for CUDA tensors on gloo
        dist.all_gather(list(out.unbind(0)), x, group=_DEVICE_GROUP)
    else:
        dist.all_gather_into_tensor(out, x, group=_DEVICE_GROUP)
    # [tp, ..., d, ...] -> [..., tp * d, ...]
    out = out.movedim(0, dim)
    return out.reshape(x.shape[:dim] + (_TP_SIZE * x.shape[dim],) + x.shape[dim + 1:])


def tensor_model_parallel_broadcast(x: torch.Tensor, src: int = 0) -> torch.Tensor:
    if _TP_SIZE == 1:
        return x
    dist.broadcast(x, src=src, group=_DEVICE_GROUP)
    return x


# ---- control plane (gloo) --------------------------------------------------


def broadcast_pyobj(data: List[Any], rank: int, src: int = 0) -> List[Any]:
    """Broadcast a picklable list from `src` to every TP rank over the CPU group."""
    if _TP_SIZE == 1:
        return data
    if rank == src:
        payload = pickle.dumps(data) if data else b""
        size = torch.tensor([len(payload)], dtype=torch.long)
        dist.broadcast(size, src=src, group=_CPU_GROUP)
        if payload:
            buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            dist.broadcast(buf, src=src, group=_CPU_GROUP)
        return data
    size = torch.tensor([0], dtype=torch.long)
    dist.broadcast(size, src=src, group=_CPU_GROUP)
    if size.item() == 0:
        return []
    buf = torch.empty(int(size.item()), dtype=torch.uint8)
    dist.broadcast(buf, src=src, group=_CPU_GROUP)
    return pickle.loads(buf.numpy().tobytes())


def all_reduce_min_int(value: int) -> int:
    """Minimum of a host integer across TP ranks (e.g. the profiled KV pool size)."""
    if _TP_SIZE == 1:
        return value
    t = torch.tensor([value], dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=_CPU_GROUP)
    return int(t.item())
