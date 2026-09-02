"""Request/response schemas mirroring fal depth-anything/v2 wire format."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DepthRequest(BaseModel):
    """fal-shaped body: top-level ``image_url`` and/or inline base64."""

    image_url: str | None = None
    image_base64: str | None = Field(default=None, description="Raw or data-URL base64 PNG/JPEG")
    image: str | dict | None = Field(
        default=None,
        description="fal-style image field: URL string or {url|content} object",
    )

    @model_validator(mode="after")
    def require_one_image_source(self) -> DepthRequest:
        has_url = bool(self.image_url)
        has_b64 = bool(self.image_base64)
        has_image = self.image is not None and self.image != ""
        if not (has_url or has_b64 or has_image):
            raise ValueError("provide image_url, image_base64, or image")
        return self


class ImageOutput(BaseModel):
    content_type: str = "image/png"
    width: int
    height: int
    url: str | None = None
    content: str | None = Field(default=None, description="Base64-encoded PNG when url is omitted")


class DepthResponse(BaseModel):
    image: ImageOutput
