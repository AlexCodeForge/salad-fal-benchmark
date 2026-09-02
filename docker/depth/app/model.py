"""Depth-Anything-V2 inference and MOCK_INFERENCE gradient stub."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
PROD_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


@dataclass
class DepthResult:
    raw_u8: np.ndarray  # H×W uint8 grayscale, higher = nearer

    @property
    def height(self) -> int:
        return int(self.raw_u8.shape[0])

    @property
    def width(self) -> int:
        return int(self.raw_u8.shape[1])


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_id() -> str:
    return os.getenv("MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def _resolve_dtype(device: torch.device) -> torch.dtype:
    raw = os.getenv("TORCH_DTYPE", "float16" if device.type == "cuda" else "float32")
    if raw.lower() in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.float32


def _minmax_uint8(depth: torch.Tensor) -> np.ndarray:
    d = depth.detach().float().cpu()
    d_min = float(d.min())
    d_max = float(d.max())
    if d_max > d_min:
        scaled = (d - d_min) / (d_max - d_min) * 255.0
    else:
        scaled = torch.zeros_like(d)
    return scaled.clamp(0, 255).byte().numpy()


def mock_depth_u8(width: int, height: int) -> np.ndarray:
    """Deterministic gradient fake depth (uint8), higher = nearer (left→right ramp)."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    row = x[np.newaxis, :]
    gradient = np.broadcast_to(row, (height, width)).astype(np.uint8)
    return gradient


class DepthModel:
    def __init__(self) -> None:
        self.mock = _env_bool("MOCK_INFERENCE")
        self.model_id = _resolve_model_id()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(self.device)
        self._processor = None
        self._model = None
        self.ready = False

    def load(self) -> None:
        if self.mock:
            logger.info("MOCK_INFERENCE=1 — skipping Depth-Anything weights")
            self.ready = True
            return

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        logger.info("Loading %s on %s (%s)", self.model_id, self.device, self.dtype)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype if self.device.type == "cuda" else torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        self.ready = True
        logger.info("Model ready")

    def infer(self, image: Image.Image) -> DepthResult:
        rgb = image.convert("RGB")
        w, h = rgb.size

        if self.mock:
            return DepthResult(raw_u8=mock_depth_u8(w, h))

        assert self._processor is not None and self._model is not None

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda" and self.dtype == torch.float16:
            inputs = {
                k: v.to(self.dtype) if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        with torch.inference_mode():
            outputs = self._model(**inputs)

        predicted = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(h, w)],
        )[0]["predicted_depth"]

        raw_u8 = _minmax_uint8(predicted)
        return DepthResult(raw_u8=raw_u8)


def depth_png_bytes(raw_u8: np.ndarray) -> bytes:
    img = Image.fromarray(raw_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
