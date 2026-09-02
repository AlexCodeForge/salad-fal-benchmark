"""Request/response schemas mirroring fal-ai/sam-3/image."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Sam3InferenceParams(BaseModel):
    """SAM3 prompt/flags — image supplied separately for multipart."""

    prompt: str
    apply_mask: bool = False
    return_multiple_masks: bool = True
    max_masks: int = Field(default=8, ge=1, le=32)
    include_scores: bool = True
    include_boxes: bool = True


class Sam3Request(Sam3InferenceParams):
    image_url: str | None = None
    image: str | None = Field(
        default=None,
        description="Base64-encoded image bytes (optionally data-URL prefixed).",
    )
    image_base64: str | None = Field(
        default=None,
        description="Alias for image (bench JSON encoding).",
    )

    @model_validator(mode="after")
    def require_image_source(self) -> Sam3Request:
        if not self.image_url and not self.image and not self.image_base64:
            raise ValueError("Either image_url, image, or image_base64 is required")
        return self

    @property
    def inline_image_b64(self) -> str | None:
        """Base64 payload from either ``image`` or ``image_base64``."""
        return self.image or self.image_base64


class MaskEntry(BaseModel):
    url: str
    score: float | None = None


class Sam3Response(BaseModel):
    masks: list[MaskEntry | dict[str, Any]]
    scores: list[float] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
