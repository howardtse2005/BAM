"""Single model holder for the whole process.

The reference implementation builds ``Predictor`` twice (once in ``app.py`` and
once in ``magic_pen_router.py``), which loads the weights twice and doubles peak
memory. This registry is created once, warmed up once, and shared.

It also refuses to load a checkpoint whose ``data_tier`` is not the configured
tier, so a research-only model cannot be shipped by accident.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config import RGB_MEAN, RGB_STD

from .predictor import MockBackend, OnnxBackend, Predictor, TorchBackend


class ModelUnavailable(RuntimeError):
    """Raised when no usable backend could be constructed."""


class ModelRegistry:
    _instance: "ModelRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.predictor: Predictor | None = None
        self.error: str | None = None
        self.attempts: list[str] = []
        self.meta: dict = {}
        self._infer_lock = threading.Lock()
        self.load()

    # -- lifecycle --------------------------------------------------------- #

    @classmethod
    def instance(cls, cfg=None) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                if cfg is None:
                    raise RuntimeError("first call to ModelRegistry.instance needs a config")
                cls._instance = cls(cfg)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    # -- loading ----------------------------------------------------------- #

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.cfg.path(value)

    def _checkpoint_meta(self, path: Path) -> dict:
        try:
            import torch

            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return {
                "epoch": ckpt.get("epoch"),
                "data_tier": ckpt.get("data_tier", "unknown"),
                "manifest_hash": ckpt.get("manifest_hash"),
                "created_at": ckpt.get("created_at"),
            }
        except Exception:
            return {}

    def _build_torch(self):
        path = self.cfg.model_file
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        backend = TorchBackend(
            path,
            device=self.cfg.infer_device,
            threads=self.cfg.threads,
        )
        return backend, self._checkpoint_meta(path)

    def _build_onnx(self, which: str):
        raw = (
            self.cfg.onnx_int8_path if which == "onnx_int8" else self.cfg.onnx_path
        )
        path = self._resolve(raw)
        if not path.is_file():
            raise FileNotFoundError(f"onnx model not found: {path}")
        return OnnxBackend(path, name=which, threads=self.cfg.threads), {"data_tier": "unknown"}

    def load(self) -> None:
        self.attempts = []
        want = self.cfg.backend

        if want == "mock":
            order = ["mock"]
        elif want == "torch":
            order = ["torch"]
        elif want == "onnx_int8":
            order = ["onnx_int8", "onnx_fp32"]
        elif want == "onnx_fp32":
            order = ["onnx_fp32", "onnx_int8"]
        else:
            order = ["onnx_int8", "onnx_fp32", "torch", "mock"]

        for name in order:
            try:
                if name == "mock":
                    if not self.cfg.allow_mock:
                        raise ModelUnavailable("mock backend disabled by config")
                    backend, meta = MockBackend(), {"data_tier": "none"}
                elif name == "torch":
                    backend, meta = self._build_torch()
                else:
                    backend, meta = self._build_onnx(name)

                tier = str(meta.get("data_tier", "unknown"))
                if tier == "B" and self.cfg.tier != "B":
                    raise ModelUnavailable(
                        "checkpoint was trained on research-tier data (B); "
                        "set data.tier=B explicitly to load it"
                    )

                predictor = Predictor(
                    backend,
                    mean=RGB_MEAN,
                    std=RGB_STD,
                    batch_size=self.cfg.infer_batch,
                )
                predictor.backend.warmup(self.cfg.crop_size)

                self.predictor = predictor
                self.meta = {**meta, **backend.info()}
                self.error = None
                self.attempts.append(f"{name}: ok")
                return
            except Exception as exc:  # keep trying the next backend
                self.attempts.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

        self.predictor = None
        self.error = "no backend available; " + " | ".join(self.attempts)

    # -- inference --------------------------------------------------------- #

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def predict(
        self, rgb_crops: list[np.ndarray], stroke_crops: list[np.ndarray]
    ) -> np.ndarray:
        if self.predictor is None:
            raise ModelUnavailable(self.error or "model unavailable")
        with self._infer_lock:
            return self.predictor.predict(rgb_crops, stroke_crops)

    def warmup(self) -> float:
        if self.predictor is None:
            raise ModelUnavailable(self.error or "model unavailable")
        import time

        size = self.cfg.crop_size
        rgb = np.zeros((size, size, 3), np.uint8)
        stroke = np.zeros((size, size), np.uint8)
        stroke[size // 2 - 2 : size // 2 + 2, 20:-20] = 255
        started = time.perf_counter()
        with self._infer_lock:
            self.predictor.predict([rgb], [stroke])
        return (time.perf_counter() - started) * 1000.0

    def info(self) -> dict:
        payload = {
            "available": self.available,
            "model": self.meta.get("backend", "none"),
            "is_mock": bool(self.meta.get("is_mock", False)),
            "device": self.cfg.infer_device,
            "data_tier": self.meta.get("data_tier", "unknown"),
            "epoch": self.meta.get("epoch"),
            "manifest_hash": self.meta.get("manifest_hash"),
            "attempts": self.attempts,
            "error": self.error,
            "crop_size": self.cfg.crop_size,
            "batch_size": self.cfg.infer_batch,
        }
        if self.predictor is not None:
            payload["last_ms"] = round(self.predictor.last_ms, 2)
            payload["last_batches"] = self.predictor.last_batches
        return payload


__all__ = ["ModelRegistry", "ModelUnavailable"]
