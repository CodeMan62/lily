"""Phase 4: RadixCache prefix caching.

Two acceptance tests:
  1. fuzz insert/match/evict against a dict-of-prefixes reference (pure CPU);
  2. through the scheduler: 32 requests sharing a system prompt compute prefill
     only for the unique tokens, and outputs stay identical under forced
     eviction.
"""

import random
from typing import List

import torch

from lily.models.model_config import ModelConfig
from lily.scheduler.scheduler import Scheduler
from lily.mem_cache.radix_cache import RadixCache
from lily.sampling.sampling_params import SamplingParams
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

VOCAB = 49152


# ---- reference implementation (dict of prefixes) ---------------------------

_VAL: dict = {}


def prefix_val(pref: tuple) -> int:
    if pref not in _VAL:
        _VAL[pref] = (hash(pref) % (2**31 - 1)) + 1
    return _VAL[pref]


def make_value(seq: List[int]) -> torch.Tensor:
    return torch.tensor([prefix_val(tuple(seq[: i + 1])) for i in range(len(seq))], dtype=torch.int64)


def ref_match(seqs: List[List[int]], key: List[int]) -> int:
    best = 0
    for s in seqs:
        common = 0
        while common < len(s) and common < len(key) and s[common] == key[common]:
            common += 1
        best = max(best, common)
    return best


class FakeAllocator:
    def __init__(self):
        self._next = 1
        self.live = set()
        self.freed: List[int] = []

    def alloc(self, n: int) -> List[int]:
        out = list(range(self._next, self._next + n))
        self._next += n
        self.live.update(out)
        return out

    def free(self, idx) -> None:
        for i in idx.tolist() if hasattr(idx, "tolist") else list(idx):
            self.freed.append(i)
            self.live.discard(i)


def _tree_indices(root) -> set:
    out = set()
    stack = [root]
    while stack:
        node = stack.pop()
        out.update(int(x) for x in node.value.tolist())
        stack.extend(node.children.values())
    return out


def _evictable_size(root) -> int:
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node is not root and node.lock_ref == 0:
            total += node.value_len
        stack.extend(node.children.values())
    return total


def test_fuzz_insert_match_evict_against_reference():
    rng = random.Random(1)
    alloc = FakeAllocator()
    tree = RadixCache(alloc)
    ever: List[List[int]] = []

    for _ in range(600):
        r = rng.random()
        if r < 0.5:
            seq = [rng.randint(0, 5) for _ in range(rng.randint(1, 10))]
            kv = torch.tensor(alloc.alloc(len(seq)))
            duplicated, _ = tree.insert(seq, kv)
            if duplicated:
                alloc.free(kv[:duplicated])
            ever.append(seq)
        elif r < 0.75:
            tree.evict(rng.randint(1, 8))
        else:
            key = [rng.randint(0, 5) for _ in range(rng.randint(0, 10))]
            matched, _ = tree.match_prefix(key)
            assert len(matched) <= ref_match(ever, key)
            assert set(matched) <= alloc.live

        assert tree.evictable_size == _evictable_size(tree.root_node)
        assert _tree_indices(tree.root_node) <= alloc.live

    # Lock refs protect a matched prefix from eviction.
    seq = [9, 9, 9]
    kv = torch.tensor(alloc.alloc(3))
    tree.insert(seq, kv)
    matched, node = tree.match_prefix(seq)
    assert matched == kv.tolist()
    tree.inc_lock_ref(node)
    tree.evict(10 ** 6)
    assert tree.match_prefix(seq)[0] == kv.tolist()
    tree.dec_lock_ref(node)
    tree.evict(10 ** 6)
    assert tree.match_prefix(seq)[0] == []


# ---- scheduler-level prefix caching ---------------------------------------

def _new_scheduler(max_total, max_reqs, disable_radix, chunked=8192):
    cfg = ModelConfig.from_pretrained(SMALL_LLAMA, dtype="float32", context_length=2048)
    sa = ServerArgs(model_path=SMALL_LLAMA, dtype="float32", attention_backend="triton",
                    context_length=2048, max_running_requests=max_reqs, max_total_tokens=max_total,
                    chunked_prefill_size=chunked, disable_radix_cache=disable_radix)
    return Scheduler(cfg, sa)


def _run(sch, prompts, max_new_tokens):
    sp = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=True)
    for i, p in enumerate(prompts):
        sch.add_request(f"r{i}", p, sp)
    while sch.step(): pass
    return [list(sch._requests[f"r{i}"].result.output_ids) for i in range(len(prompts))]


def test_shared_system_prompt_and_forced_eviction():
    rng = random.Random(0)
    system = [rng.randrange(1000, VOCAB - 1000) for _ in range(48)]
    unique = [[rng.randrange(1000, VOCAB - 1000) for _ in range(6)] for _ in range(32)]
    prompts = [system + u for u in unique]

    # Cached: warm the tree, then the 32 requests prefill only their unique tail.
    sch = _new_scheduler(max_total=8192, max_reqs=32, disable_radix=False)
    sch.add_request("warm", prompts[0], SamplingParams(max_new_tokens=6, ignore_eos=True))
    while sch.step(): pass
    warm = sch.prefill_tokens_computed
    sch.prefill_tokens_computed = 0
    shared = _run(sch, prompts, 6)
    cached = sch.prefill_tokens_computed
    assert warm + cached <= len(system) + 32 * 6 + 16
    assert cached <= 32 * 6 + 16
    sch = None
    free_cuda()

    # Forced eviction with a tiny pool: outputs must be unchanged.
    prompts2 = [[rng.randrange(1000, VOCAB - 1000) for _ in range(rng.randint(10, 40))] for _ in range(16)]
    sch = _new_scheduler(max_total=384, max_reqs=8, disable_radix=False, chunked=64)
    evicted = _run(sch, prompts2, 6)
    assert sch.token_to_kv_pool_allocator.available_size() + sch.tree_cache.evictable_size <= 384
    sch = None
    free_cuda()

    sch = _new_scheduler(max_total=8192, max_reqs=1, disable_radix=True)
    reference = _run(sch, prompts2, 6)
    sch = None
    free_cuda()
    assert evicted == reference
