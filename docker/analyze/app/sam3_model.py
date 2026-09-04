"""SAM3 inference: real transformers path or deterministic mock."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get("SAM3_CONFIG_PATH", "/app/configs/sam3-model.yaml")
)


@dataclass
class ModelConfig:
    model_id: str = "facebook/sam3"
    revision: str | None = None
    dtype: str = "float16"
    threshold: float = 0.5
    mask_threshold: float = 0.5
    default_max_masks: int = 8
    max_max_masks: int = 32


def load_config() -> ModelConfig:
    if CONFIG_PATH.is_file():
        raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        return ModelConfig(
            model_id=str(raw.get("model_id", "facebook/sam3")),
            revision=raw.get("revision"),
            dtype=str(raw.get("dtype", "float16")),
            threshold=float(raw.get("threshold", 0.5)),
            mask_threshold=float(raw.get("mask_threshold", 0.5)),
            default_max_masks=int(raw.get("default_max_masks", 8)),
            max_max_masks=int(raw.get("max_max_masks", 32)),
        )
    return ModelConfig()


def mock_enabled() -> bool:
    return os.environ.get("MOCK_INFERENCE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def decode_image_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


async def fetch_image_url(url: str, timeout_s: float = 60.0) -> Image.Image:
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return decode_image_bytes(resp.content)


def decode_base64_image(value: str) -> Image.Image:
    payload = value.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    data = base64.b64decode(payload)
    return decode_image_bytes(data)


def mask_to_data_url(mask: np.ndarray) -> str:
    """Encode boolean/float mask as fal-style grayscale PNG data URL."""
    arr = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    if arr.ndim == 3:
        arr = arr[..., 0]
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def bbox_xywh_normalized(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h]


def _prompt_seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)


def mock_infer(
    image: Image.Image,
    prompt: str,
    *,
    max_masks: int,
    include_scores: bool,
    include_boxes: bool,
) -> dict[str, Any]:
    """Deterministic pseudo-segmentation for local/CI without GPU weights."""
    w, h = image.size
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    seed = _prompt_seed(prompt)
    prompt_l = prompt.strip().lower()

    masks: list[np.ndarray] = []
    if prompt_l == "wall":
        y_cut = int(h * (0.15 + (seed % 7) / 100.0))
        m = np.zeros((h, w), dtype=bool)
        m[y_cut : int(h * 0.92), :] = gray[y_cut : int(h * 0.92), :] > 0.35
        masks.append(m)
    elif prompt_l == "floor":
        y_cut = int(h * (0.55 + (seed % 5) / 100.0))
        m = np.zeros((h, w), dtype=bool)
        m[y_cut:, :] = gray[y_cut:, :] > 0.25
        masks.append(m)
    elif prompt_l == "molding":
        band = max(4, h // 40)
        top = int(h * 0.08)
        m = np.zeros((h, w), dtype=bool)
        m[top : top + band, :] = True
        masks.append(m)
    elif prompt_l == "mullion":
        # Two vertical bands — often empty-ish for benchmark edge cases
        cx = w // 2
        bw = max(8, w // 24)
        m = np.zeros((h, w), dtype=bool)
        if seed % 3 != 0:
            m[:, cx - bw : cx + bw] = gray[:, cx - bw : cx + bw] > 0.4
        masks.append(m)
    else:
        thresh = 0.3 + (seed % 20) / 100.0
        masks.append(gray > thresh)

    # Optional extra masks when return_multiple_masks + max_masks > 1
    while len(masks) < max_masks and len(masks) < 3:
        shifted = np.roll(masks[0], (seed % 11) - 5, axis=0)
        if shifted.any():
            masks.append(shifted)

    masks = masks[:max_masks]
    scores: list[float] = []
    boxes: list[list[float]] = []
    out_masks: list[dict[str, Any]] = []

    for i, m in enumerate(masks):
        score = round(0.82 + ((seed + i * 17) % 13) / 100.0, 4)
        if include_scores:
            scores.append(score)
        box = bbox_xywh_normalized(m) if include_boxes else []
        if include_boxes:
            boxes.append(box)
        entry: dict[str, Any] = {"url": mask_to_data_url(m)}
        if include_scores:
            entry["score"] = score
        out_masks.append(entry)

    return {"masks": out_masks, "scores": scores, "boxes": boxes}


class Sam3Engine:
    """Lazy-load SAM3 on first real inference; mock skips load entirely."""

    def __init__(self) -> None:
        self.config = load_config()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._loaded = False
        self._load_error: str | None = None

    @property
    def mock(self) -> bool:
        return mock_enabled()

    @property
    def model_id(self) -> str:
        return os.environ.get("SAM3_MODEL", self.config.model_id)

    def status(self) -> dict[str, Any]:
        return {
            "mock_inference": self.mock,
            "model_id": self.model_id,
            "model_loaded": self._loaded,
            "device": self._device,
            "load_error": self._load_error,
        }

    def warm_load(self) -> None:
        if self.mock:
            logger.info("MOCK_INFERENCE=1 — skipping SAM3 weight download")
            return
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 — surface at /health
            self._load_error = str(exc)
            logger.exception("SAM3 warm load failed")

    def _ensure_loaded(self) -> None:
        if self._loaded or self.mock:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.float16
            if self._device == "cuda" and self.config.dtype == "float16"
            else torch.float32
        )
        logger.info("Loading %s on %s (dtype=%s)", self.model_id, self._device, dtype)
        kwargs: dict[str, Any] = {}
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        self._processor = Sam3Processor.from_pretrained(self.model_id, **kwargs)
        self._model = Sam3Model.from_pretrained(self.model_id, torch_dtype=dtype, **kwargs)
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True
        self._load_error = None

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_masks: int,
        include_scores: bool,
        include_boxes: bool,
        return_multiple_masks: bool,
    ) -> dict[str, Any]:
        if self.mock:
            cap = max_masks if return_multiple_masks else 1
            return mock_infer(
                image,
                prompt,
                max_masks=cap,
                include_scores=include_scores,
                include_boxes=include_boxes,
            )

        import torch

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        cap = max_masks if return_multiple_masks else 1
        inputs = self._processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        raw_masks = results.get("masks", [])
        raw_scores = results.get("scores", [])
        raw_boxes = results.get("boxes", [])

        out_masks: list[dict[str, Any]] = []
        scores: list[float] = []
        boxes: list[list[float]] = []

        w, h = image.size
        for i in range(min(len(raw_masks), cap)):
            mask_t = raw_masks[i]
            mask_np = mask_t.cpu().numpy().astype(bool)
            score_f = float(raw_scores[i].cpu().item()) if i < len(raw_scores) else 1.0
            if include_scores:
                scores.append(round(score_f, 4))
            if include_boxes and i < len(raw_boxes):
                box = raw_boxes[i].cpu().tolist()
                if len(box) >= 4:
                    # transformers xyxy pixel coords → normalized xywh
                    x0, y0, x1, y1 = box[:4]
                    boxes.append([x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h])
                else:
                    boxes.append(bbox_xywh_normalized(mask_np))
            elif include_boxes:
                boxes.append(bbox_xywh_normalized(mask_np))

            entry: dict[str, Any] = {"url": mask_to_data_url(mask_np)}
            if include_scores:
                entry["score"] = scores[-1]
            out_masks.append(entry)

        return {"masks": out_masks, "scores": scores, "boxes": boxes}


engine = Sam3Engine()
