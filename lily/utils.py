from __future__ import annotations

import contextlib
import functools
from typing import Sequence

import torch


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def new_generator(seed: int | None = None, device: torch.device | str = "cpu") -> torch.Generator:
    """A torch generator seeded from `seed` (or a fresh random seed)."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed if seed is not None else int(torch.initial_seed() & 0x7FFF_FFFF))
    return generator


def longest_partial_suffix(text: str, delimiter: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `delimiter`.

    Used by streaming text consumers (detokenizer stop strings, reasoning parser)
    to hold back a delimiter that may still be completed by the next chunk.
    """
    for length in range(min(len(text), len(delimiter) - 1), 0, -1):
        if text.endswith(delimiter[:length]):
            return length
    return 0


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("b can't be zero")
    return -(-a // b)


class _PinnedRing:
    """Pinned staging rows for small host -> device copies. `torch.tensor(values, device=...)` is a
    blocking pageable copy (~250 us in a sandboxed container, ~14 per speculative round); from pinned
    memory the copy is async. A row is rewritten after SLOTS more copies, which the per-round device
    sync keeps far behind."""

    SLOTS, WIDTH = 1024, 1024

    def __init__(self):
        self.rows = torch.empty(self.SLOTS, self.WIDTH, dtype=torch.int64, pin_memory=True)
        self.view = self.rows.numpy()
        self.next = 0

    def put(self, values: Sequence[int], device) -> torch.Tensor:
        i, n = self.next, len(values)
        self.next = (i + 1) % self.SLOTS
        self.view[i, :n] = values
        return self.rows[i, :n].to(device, non_blocking=True)


@functools.lru_cache(maxsize=None)
def _pinned_ring() -> _PinnedRing:
    return _PinnedRing()


def int_tensor(values: Sequence[int], device, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    """A small integer list on `device`, without blocking the host on CUDA."""
    if torch.device(device).type != "cuda" or len(values) > _PinnedRing.WIDTH:
        return torch.tensor(values, dtype=dtype, device=device)
    out = _pinned_ring().put(values, device)
    return out if dtype == torch.int64 else out.to(dtype)


@functools.cache
def _side_stream() -> torch.cuda.Stream:
    return torch.cuda.Stream()


def run_beside(side_fn, main_fn):
    """(side_fn(), main_fn()), concurrently while capturing a CUDA graph: small kernels leave most of
    the GPU idle. Sequential otherwise, where the allocator would need the tensors' streams recorded."""
    if not torch.cuda.is_current_stream_capturing():
        return side_fn(), main_fn()
    main, side = torch.cuda.current_stream(), _side_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        a = side_fn()
    b = main_fn()
    main.wait_stream(side)
    return a, b
