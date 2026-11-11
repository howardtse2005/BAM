"""Small thread-safe LRU cache.

Two uses:
  * decoded source images, so repeated strokes on one image do not re-read disk;
  * crop predictions, so undo/redo and repeated strokes are instant.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Callable


class LRUCache:
    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = max(1, int(maxsize))
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return default

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def get_or_create(self, key: Any, factory: Callable[[], Any]) -> Any:
        sentinel = object()
        found = self.get(key, sentinel)
        if found is not sentinel:
            return found
        value = factory()
        self.put(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


def key_from(*parts: Any) -> str:
    """Stable cache key from strings, bytes and numbers."""
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, bytes):
            digest.update(part)
        elif isinstance(part, str):
            digest.update(part.encode("utf-8"))
        else:
            digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = ["LRUCache", "key_from"]
