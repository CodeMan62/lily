"""Prefix-caching radix tree. Mirrors sglang/srt/mem_cache/radix_cache.py (reduced).

Ownership model
---------------
The tree owns the KV-pool indices stored in its nodes. `match_prefix` returns the
tree's indices, which a request may only read (borrow) while it holds a lock via
`inc_lock_ref`. `insert` takes ownership of the indices it newly stores and returns
the number of leading tokens that were already present (duplicates); the caller owns
those duplicate indices and must free them.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# Logical clock for LRU instead of wall time: the order is the same, but it is
# reproducible, so TP ranks replaying the same scheduler steps evict the same nodes.
_access_clock = itertools.count()
_node_ids = itertools.count()


class TreeNode:
    def __init__(
        self,
        key: Optional[Iterable[int]] = None,
        value: Optional[torch.Tensor] = None,
        parent: Optional["TreeNode"] = None,
    ):
        self.key: List[int] = list(key or [])
        self.value: torch.Tensor = value if value is not None else torch.empty(0, dtype=torch.int64)
        self.parent: Optional[TreeNode] = parent
        self.children: Dict[int, "TreeNode"] = {}
        self.lock_ref: int = 0
        self.node_id: int = next(_node_ids)
        self.last_access_time: int = next(_access_clock)

    @property
    def value_len(self) -> int:
        return int(self.value.numel())

    def touch(self) -> None:
        self.last_access_time = next(_access_clock)


class RadixCache:
    def __init__(self, token_to_kv_pool_allocator: Any):
        self.root_node = TreeNode()
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self._evictable_size = 0

    # ---- introspection -----------------------------------------------------

    @property
    def evictable_size(self) -> int:
        return self._evictable_size

    def total_size(self) -> int:
        total = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            total += node.value_len
            stack.extend(node.children.values())
        return total

    # ---- match -------------------------------------------------------------

    def match_prefix(self, key: List[int]) -> Tuple[List[int], Optional[TreeNode]]:
        """Longest prefix of `key` present in the tree.

        Returns the flattened KV indices and the deepest node touched (None when
        nothing matched). The returned indices are borrowed, not owned.
        """
        matched: List[int] = []
        node = self.root_node
        last_node: Optional[TreeNode] = None
        offset = 0
        while offset < len(key):
            child = node.children.get(key[offset])
            if child is None:
                break
            common = 0
            limit = min(len(child.key), len(key) - offset)
            while common < limit and child.key[common] == key[offset + common]:
                common += 1
            if common == 0:
                break
            matched.extend(int(x) for x in child.value[:common].tolist())
            offset += common
            child.touch()
            last_node = child
            if common < len(child.key):
                break
            node = child
        return matched, last_node

    # ---- lock refs ---------------------------------------------------------

    def inc_lock_ref(self, node: Optional[TreeNode]) -> None:
        while node is not None and node is not self.root_node:
            if node.lock_ref == 0:
                self._evictable_size -= node.value_len
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: Optional[TreeNode]) -> None:
        while node is not None and node is not self.root_node:
            if node.lock_ref <= 0:
                raise RuntimeError("radix cache lock underflow")
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self._evictable_size += node.value_len
            node = node.parent

    # ---- insert ------------------------------------------------------------

    def insert(self, key: List[int], value: torch.Tensor) -> Tuple[int, Optional[TreeNode]]:
        """Insert `key` with KV indices `value`.

        Returns `(duplicated, last_node)`: `duplicated` leading tokens were already
        in the tree (the caller still owns `value[:duplicated]` and should free
        them); the rest of `value` is now owned by the tree.
        """
        if len(key) == 0:
            return 0, self.root_node
        node = self.root_node
        offset = 0
        duplicated = 0
        while offset < len(key):
            child = node.children.get(key[offset])
            if child is None:
                new_node = TreeNode(key[offset:], value[offset:].clone(), node)
                node.children[key[offset]] = new_node
                self._evictable_size += new_node.value_len
                return duplicated, new_node

            common = 0
            limit = min(len(child.key), len(key) - offset)
            while common < limit and child.key[common] == key[offset + common]:
                common += 1

            if common < len(child.key):
                # Split the existing node at the divergence point.
                split = TreeNode(child.key[:common], child.value[:common].clone(), node)
                split.lock_ref = child.lock_ref
                split.last_access_time = child.last_access_time
                child.parent = split
                child.key = child.key[common:]
                child.value = child.value[common:]
                split.children[child.key[0]] = child
                node.children[key[offset]] = split
                child = split

            duplicated += common
            node = child
            offset += common
            child.touch()
        return duplicated, node

    # ---- evict -------------------------------------------------------------

    def _collect_leaves(self) -> List[TreeNode]:
        leaves: List[TreeNode] = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node and not node.children:
                leaves.append(node)
            stack.extend(node.children.values())
        return leaves

    def evict(self, num_tokens: int) -> int:
        """Free at least `num_tokens` worth of unlocked KV; returns tokens freed."""
        if num_tokens <= 0:
            return 0
        freed = 0
        heap = [(leaf.last_access_time, leaf.node_id, leaf) for leaf in self._collect_leaves()]
        heapq.heapify(heap)
        while heap and freed < num_tokens:
            _, _, node = heapq.heappop(heap)
            if node.lock_ref or node.parent is None:
                continue
            parent = node.parent
            self.token_to_kv_pool_allocator.free(node.value)
            freed += node.value_len
            self._evictable_size -= node.value_len
            if parent.children.get(node.key[0]) is node:
                del parent.children[node.key[0]]
            node.parent = None
            if parent is not self.root_node and not parent.children and parent.lock_ref == 0:
                heapq.heappush(heap, (parent.last_access_time, parent.node_id, parent))
        return freed

    def clear(self) -> None:
        self.evict(self._evictable_size)


class ChunkCache:
    """No-op cache used with `--disable-radix-cache`; chunked prefill still works."""

    evictable_size = 0

    def match_prefix(self, key: List[int]) -> Tuple[List[int], None]:
        return [], None

    def insert(self, key: List[int], value: torch.Tensor) -> Tuple[int, None]:
        return 0, None

    def inc_lock_ref(self, node: Any) -> None:
        return None

    def dec_lock_ref(self, node: Any) -> None:
        return None

    def evict(self, num_tokens: int) -> int:
        return 0

    def clear(self) -> None:
        return None
