"""Small helpers shared across the test suite."""

from __future__ import annotations

import gc

import torch


def free_cuda() -> None:
    """Drop unreferenced objects and return cached blocks to the CUDA allocator."""
    gc.collect()
    torch.cuda.empty_cache()
