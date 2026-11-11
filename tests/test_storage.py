import base64

import cv2
import numpy as np
import pytest

from backend.services.cache import LRUCache, key_from
from backend.services.storage import (
    StorageError,
    decode_data_url,
    decode_mask_bytes,
    encode_png_base64,
    safe_join,
    sniff_image,
)


def test_safe_join_allows_children(tmp_path):
    path = safe_join(tmp_path, "a", "b.png")
    assert path.parent.name == "a"


@pytest.mark.parametrize("parts", [("..", "x.png"), ("a", "..", "..", "x"), ("/etc/passwd",)])
def test_safe_join_rejects_escapes(tmp_path, parts):
    with pytest.raises(StorageError):
        safe_join(tmp_path, *parts)


def test_decode_data_url_handles_prefix_and_raw():
    payload = base64.b64encode(b"hello").decode()
    assert decode_data_url(f"data:image/png;base64,{payload}") == b"hello"
    assert decode_data_url(payload) == b"hello"


def test_decode_data_url_rejects_empty():
    with pytest.raises(StorageError):
        decode_data_url("")


def test_png_roundtrip():
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    url = encode_png_base64(mask)
    assert url.startswith("data:image/png;base64,")

    decoded = decode_mask_bytes(decode_data_url(url))
    assert np.array_equal(decoded, mask)


def test_decode_mask_accepts_rgba_alpha():
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[2:6, 2:6, 3] = 255
    ok, buf = cv2.imencode(".png", rgba)
    assert ok
    decoded = decode_mask_bytes(buf.tobytes())
    assert decoded[3, 3] == 255
    assert decoded[0, 0] == 0


def test_sniff_image_detects_png():
    ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    assert ok
    assert sniff_image(buf.tobytes()) == "png"


def test_sniff_image_rejects_garbage():
    with pytest.raises(StorageError):
        sniff_image(b"not an image at all")


def test_lru_evicts_oldest():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_lru_stats():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.get("a")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_key_from_is_stable():
    assert key_from("a", b"b", 1) == key_from("a", b"b", 1)
    assert key_from("a") != key_from("b")
