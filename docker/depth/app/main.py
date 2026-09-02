"""FastAPI depth service — fal-compatible POST /v1/depth."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.model import PROD_MODEL_ID, DepthModel, depth_png_bytes
from app.schemas import DepthRequest, DepthResponse, ImageOutput

logger = logging.getLogger(__name__)

_depth_model: DepthModel | None = None


def _get_model() -> DepthModel:
    if _depth_model is None or not _depth_model.ready:
        raise HTTPException(status_code=503, detail="model not ready")
    return _depth_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _depth_model
    logging.basicConfig(level=logging.INFO)
    _depth_model = DepthModel()
    await asyncio.to_thread(_depth_model.load)
    yield


app = FastAPI(title="salad-fal-benchmark depth", version="0.1.0", lifespan=lifespan)


def _strip_data_url(raw: str) -> str:
    if raw.startswith("data:"):
        _, _, payload = raw.partition(",")
        return payload
    return raw


def _decode_base64_image(raw: str) -> Image.Image:
    try:
        data = base64.b64decode(_strip_data_url(raw), validate=False)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {exc}") from exc
    try:
        return Image.open(io.BytesIO(data))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image bytes: {exc}") from exc


async def _download_image(url: str) -> Image.Image:
    timeout = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_S", "60"))
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"failed to download image_url: {exc}") from exc
    try:
        return Image.open(io.BytesIO(data))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"downloaded bytes are not an image: {exc}") from exc


def _image_from_request_fields(body: DepthRequest) -> tuple[Image.Image | None, str | None]:
    if body.image_url:
        return None, body.image_url

    if body.image_base64:
        return _decode_base64_image(body.image_base64), None

    if body.image is None:
        return None, None

    if isinstance(body.image, str):
        if body.image.startswith(("http://", "https://")):
            return None, body.image
        return _decode_base64_image(body.image), None

    if isinstance(body.image, dict):
        url = body.image.get("url")
        if url:
            return None, str(url)
        content = body.image.get("content")
        if content:
            return _decode_base64_image(str(content)), None

    raise HTTPException(status_code=400, detail="image field must be URL string or {url|content}")


async def _resolve_input_image(body: DepthRequest) -> Image.Image:
    pil, url = _image_from_request_fields(body)
    if pil is not None:
        return pil
    if url:
        return await _download_image(url)
    raise HTTPException(status_code=400, detail="no image source in request")


def _build_response(raw_u8, *, return_url: bool) -> DepthResponse:
    import numpy as np

    arr = np.asarray(raw_u8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    png = depth_png_bytes(arr)
    b64 = base64.b64encode(png).decode("ascii")

    if return_url:
        # Salad gateway may front a CDN; placeholder for future signed URLs.
        return DepthResponse(
            image=ImageOutput(
                content_type="image/png",
                width=w,
                height=h,
                url=f"data:image/png;base64,{b64}",
            )
        )

    return DepthResponse(
        image=ImageOutput(
            content_type="image/png",
            width=w,
            height=h,
            content=b64,
        )
    )


async def _run_depth(image: Image.Image) -> DepthResponse:
    model = _get_model()
    return_url = os.getenv("RETURN_IMAGE_URL", "").strip().lower() in {"1", "true", "yes"}
    result = await asyncio.to_thread(model.infer, image)
    return _build_response(result.raw_u8, return_url=return_url)


@app.get("/health")
async def health() -> dict[str, Any]:
    model = _depth_model
    return {
        "status": "ok" if model and model.ready else "loading",
        "mock_inference": bool(model and model.mock),
        "model_id": model.model_id if model else None,
        "prod_model_id": PROD_MODEL_ID,
        "device": str(model.device) if model else None,
    }


@app.post("/v1/depth", response_model=DepthResponse)
async def depth_v1(request: Request, image: UploadFile | None = File(default=None)) -> DepthResponse:
    if image is not None and image.filename:
        data = await image.read()
        try:
            pil = Image.open(io.BytesIO(data))
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"invalid multipart image: {exc}") from exc
        return await _run_depth(pil)

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body or multipart image required") from exc

    try:
        body = DepthRequest.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pil = await _resolve_input_image(body)
    return await _run_depth(pil)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
