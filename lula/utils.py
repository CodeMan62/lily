from __future__ import annotations

import contextlib

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
