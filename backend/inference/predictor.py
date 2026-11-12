"""Batched inference for the magic brush.

Backends share one contract: ``__call__(batch) -> probabilities`` where
``batch`` is ``[N, 4, H, W]`` float32 already normalised, and the output is
``[N, 1, H, W]`` in [0, 1].

Three backends are provided:

  * ``torch``  -- reference implementation, works everywhere PyTorch installs;
  * ``onnx``   -- 2-4x faster on CPU, needs onnxruntime;
  * ``mock``   -- deterministic stand-in used when no checkpoint exists yet, so
                  the full request path can be exercised before training.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class BaseBackend:
    name = "base"
    is_mock = False

    def __call__(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "is_mock": self.is_mock}

    def warmup(self, crop_size: int) -> None:
        dummy = np.zeros((1, 4, crop_size, crop_size), np.float32)
        self(dummy)


class TorchBackend(BaseBackend):
    name = "torch"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        threads: int = 8,
        channels_last: bool = True,
    ) -> None:
        import torch

        from training.unet import load_checkpoint_model

        self._torch = torch
        self.device = torch.device(device)
        self.channels_last = bool(channels_last and device == "cpu")
        if device == "cpu":
            torch.set_num_threads(max(1, int(threads)))

        self.model = load_checkpoint_model(checkpoint, map_location=self.device)
        self.model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        if self.channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)
        return probs.float().cpu().numpy()

    def info(self) -> dict:
        return {
            "backend": self.name,
            "is_mock": False,
            "device": str(self.device),
            "params": int(self.params),
            "channels_last": self.channels_last,
        }


class OnnxBackend(BaseBackend):
    def __init__(self, path: str | Path, name: str = "onnx_fp32", threads: int = 8) -> None:
        import onnxruntime as ort

        self.name = name
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.path = str(path)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: batch})[0]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def info(self) -> dict:
        return {"backend": self.name, "is_mock": False, "path": self.path}


class MockBackend(BaseBackend):
    """Deterministic stand-in: predicts a band around the brush channel.

    This is not a model. It exists so the API, the merge, the line and the fill
    can be tested end to end before the first checkpoint exists. It is never
    selected silently: ``/api/health`` reports ``model: mock`` and the UI shows a
    warning.
    """

    name = "mock"
    is_mock = True

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        outputs = []
        for i in range(batch.shape[0]):
            brush = np.clip(batch[i, 3], 0.0, 1.0).astype(np.float32)
            band = cv2.GaussianBlur(brush, (0, 0), 3.0)
            peak = float(band.max())
            if peak > 1e-6:
                band = band / peak
            prob = np.clip(band * 1.25 - 0.15, 0.0, 1.0)
            outputs.append(prob[None])
        return np.stack(outputs).astype(np.float32)

    def info(self) -> dict:
        return {"backend": self.name, "is_mock": True, "note": "no checkpoint loaded"}


class Predictor:
    """Normalisation + batching in front of a backend."""

    def __init__(
        self,
        backend: BaseBackend,
        mean: tuple | None = None,
        std: tuple | None = None,
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        from config import RGB_MEAN, RGB_STD

        self.mean = np.asarray(mean if mean is not None else RGB_MEAN, np.float32).reshape(1, 4, 1, 1)
        self.std = np.asarray(std if std is not None else RGB_STD, np.float32).reshape(1, 4, 1, 1)
        self.batch_size = max(1, int(batch_size))
        self.last_ms = 0.0
        self.last_batches = 0

    def preprocess(
        self, rgb_crops: list[np.ndarray], brush_crops: list[np.ndarray]
    ) -> np.ndarray:
        if len(rgb_crops) != len(brush_crops):
            raise ValueError("rgb and brush crop counts differ")
        if not rgb_crops:
            raise ValueError("no crops supplied")

        height, width = rgb_crops[0].shape[:2]
        batch = np.zeros((len(rgb_crops), 4, height, width), np.float32)

        for i, (rgb, brush) in enumerate(zip(rgb_crops, brush_crops)):
            if rgb.shape[:2] != (height, width):
                raise ValueError("all crops must share one size")
            if brush.shape[:2] != (height, width):
                raise ValueError("brush crop size mismatch")

            rgb_f = rgb.astype(np.float32) / 255.0
            brush_f = brush.astype(np.float32)
            if brush_f.max(initial=0) > 1.5:
                brush_f = brush_f / 255.0

            batch[i, :3] = rgb_f.transpose(2, 0, 1)
            batch[i, 3] = brush_f

        return (batch - self.mean) / self.std

    def predict(
        self, rgb_crops: list[np.ndarray], brush_crops: list[np.ndarray]
    ) -> np.ndarray:
        batch = self.preprocess(rgb_crops, brush_crops)
        outputs = []
        started = time.perf_counter()
        self.last_batches = 0

        for start in range(0, batch.shape[0], self.batch_size):
            chunk = batch[start : start + self.batch_size]
            probs = self.backend(chunk)
            if probs.ndim == 3:
                probs = probs[:, None]
            outputs.append(probs.astype(np.float32))
            self.last_batches += 1

        self.last_ms = (time.perf_counter() - started) * 1000.0
        return np.concatenate(outputs, axis=0)


__all__ = [
    "BaseBackend",
    "TorchBackend",
    "OnnxBackend",
    "MockBackend",
    "Predictor",
]
