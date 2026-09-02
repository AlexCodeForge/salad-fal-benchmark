"""Encode fixture images for Salad gateway POST requests."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

MultipartFiles = dict[str, tuple[str, bytes, str]]
FormData = dict[str, str]


def image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Return (width, height) for raw image bytes."""
    with Image.open(BytesIO(image_bytes)) as img:
        return img.size


def encode_image_base64(image_bytes: bytes) -> str:
    """Base64-encode image bytes for JSON POST bodies."""
    return base64.b64encode(image_bytes).decode("ascii")


def build_multipart_image(
    image_bytes: bytes,
    *,
    field_name: str = "image",
    filename: str = "photo.jpg",
    content_type: str = "image/jpeg",
) -> MultipartFiles:
    """Build httpx multipart files dict for a single image field."""
    return {field_name: (filename, image_bytes, content_type)}


def build_sam3_post(
    image_bytes: bytes,
    prompt: str,
    max_masks: int,
    *,
    encoding: str = "multipart",
) -> tuple[MultipartFiles, FormData] | dict[str, Any]:
    """Prepare SAM3 gateway POST payload (multipart default, json optional)."""
    common = {
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    if encoding == "json":
        return {**common, "image": encode_image_base64(image_bytes)}
    form: FormData = {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in common.items()}
    files = build_multipart_image(image_bytes)
    return files, form


def build_depth_post(
    image_bytes: bytes,
    *,
    encoding: str = "multipart",
) -> tuple[MultipartFiles, FormData] | dict[str, Any]:
    """Prepare depth gateway POST payload."""
    if encoding == "json":
        return {"image_base64": encode_image_base64(image_bytes)}
    return build_multipart_image(image_bytes), {}
