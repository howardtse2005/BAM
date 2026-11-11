"""Perceptual hashing and duplicate grouping.

Near-duplicate images appear across datasets (COCO and Open Images overlap in
content). If a duplicate lands in both train and test, every reported metric is
inflated. Duplicates are therefore grouped and the whole group is assigned to a
single split.

The 64-bit hash is split into four 16-bit bands (LSH). Candidates that share a
band are compared exactly, which keeps this near-linear instead of O(n^2).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np


def phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit DCT perceptual hash as a Python int."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    size = hash_size * highfreq_factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


def group_duplicates(
    items: Iterable[tuple[str, int]], max_distance: int = 6
) -> dict[str, str]:
    """Group near-duplicate keys. Returns ``key -> group_id``.

    ``group_id`` is the smallest key in the group, so it is stable across runs.
    """
    items = [(k, int(h)) for k, h in items]
    if not items:
        return {}

    uf = _UnionFind()
    for key, _ in items:
        uf.find(key)

    bands: list[dict[int, list[tuple[str, int]]]] = [defaultdict(list) for _ in range(4)]
    for key, value in items:
        for i in range(4):
            band = (value >> (16 * i)) & 0xFFFF
            bands[i][band].append((key, value))

    for band_map in bands:
        for bucket in band_map.values():
            if len(bucket) < 2:
                continue
            for i in range(len(bucket)):
                key_i, hash_i = bucket[i]
                for j in range(i + 1, len(bucket)):
                    key_j, hash_j = bucket[j]
                    if hamming(hash_i, hash_j) <= max_distance:
                        uf.union(key_i, key_j)

    return {key: uf.find(key) for key, _ in items}


def stable_bucket(key: str, salt: str = "") -> float:
    """Deterministic value in [0, 1) from a string, independent of Python's hash."""
    import hashlib

    digest = hashlib.sha256((salt + key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


__all__ = ["phash", "hamming", "group_duplicates", "stable_bucket"]
