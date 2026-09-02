"""Unified SAM3 + depth FastAPI gateway for Salad benchmark."""

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

from app.depth_model import PROD_MODEL_ID, DepthModel, depth_png_bytes
from app.depth_schemas import DepthRequest, DepthResponse, ImageOutput
from app.sam3_model import (
    decode_base64_image,
    engine,
    fetch_image_url,
)
from app.sam3_schemas import Sam3InferenceParams, Sam3Request, Sam3Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_depth_model: DepthModel | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.warm_load()
    global _depth_model
    _depth_model = DepthModel()
    await asyncio.to_thread(_depth_model.load)
    yield


app = FastAPI(title="salad-fal-benchmark analyze", version="0.1.0", lifespan=lifespan)


# --- health ---


@app.get("/health")
async def health() -> dict[str, Any]:
    sam3 = engine.status()
    sam3_ok = sam3["mock_inference"] or sam3["model_loaded"]
    depth = _depth_model
    depth_ok = bool(depth and depth.ready)
    ok = sam3_ok and depth_ok
    body: dict[str, Any] = {
        "status": "ok" if ok else "loading",
        "sam3": sam3,
        "depth": {
            "mock_inference": bool(depth and depth.mock),
            "model_id": depth.model_id if depth else None,
            "prod_model_id": PROD_MODEL_ID,
            "device": str(depth.device) if depth else None,
            "ready": depth_ok,
        },
    }
    if not ok and sam3.get("load_error"):
        return JSONResponse(status_code=503, content={**body, "status": "error"})
    return body


# --- SAM3 ---


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sam3_params_from_form(form: dict[str, object]) -> Sam3InferenceParams:
    prompt = form.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    return Sam3InferenceParams(
        prompt=str(prompt),
        apply_mask=_parse_bool(form.get("apply_mask"), False),
        return_multiple_masks=_parse_bool(form.get("return_multiple_masks"), True),
        max_masks=_parse_int(form.get("max_masks"), 8),
        include_scores=_parse_bool(form.get("include_scores"), True),
        include_boxes=_parse_bool(form.get("include_boxes"), True),
    )


async def _resolve_sam3_image(body: Sam3Request) -> Image.Image:
    try:
        if body.image_url:
            return await fetch_image_url(body.image_url)
        inline = body.inline_image_b64
        if inline:
            return decode_base64_image(inline)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image input: {exc}") from exc
    raise HTTPException(status_code=400, detail="Either image_url, image, or image_base64 is required")


async def _run_sam3(image: Image.Image, params: Sam3InferenceParams) -> Sam3Response:
    if not engine.mock and not engine.status()["model_loaded"]:
        err = engine.status().get("load_error")
        raise HTTPException(
            status_code=503,
            detail=f"SAM3 model not loaded{f': {err}' if err else ''}",
        )

    try:
        result = engine.infer(
            image,
            params.prompt,
            max_masks=params.max_masks,
            include_scores=params.include_scores,
            include_boxes=params.include_boxes,
            return_multiple_masks=params.return_multiple_masks,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("SAM3 inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Sam3Response(**result)


@app.post("/v1/sam3", response_model=Sam3Response)
async def sam3_endpoint(
    request: Request,
    image: UploadFile | None = File(default=None),
) -> Sam3Response:
    pil: Image.Image | None = None

    if image is not None and image.filename:
        data = await image.read()
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"invalid multipart image: {exc}") from exc
        form = await request.form()
        params = _sam3_params_from_form(dict(form))
        return await _run_sam3(pil, params)

    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/"):
        form = await request.form()
        file_field = form.get("image")
        if file_field is not None and hasattr(file_field, "read"):
            data = await file_field.read()
            try:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
            except OSError as exc:
                raise HTTPException(status_code=400, detail=f"invalid multipart image: {exc}") from exc
            params = _sam3_params_from_form(dict(form))
            return await _run_sam3(pil, params)

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body or multipart image required") from exc

    try:
        body = Sam3Request.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pil = await _resolve_sam3_image(body)
    return await _run_sam3(pil, body)


# --- depth ---


def _get_depth_model() -> DepthModel:
    if _depth_model is None or not _depth_model.ready:
        raise HTTPException(status_code=503, detail="model not ready")
    return _depth_model


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


async def _resolve_depth_image(body: DepthRequest) -> Image.Image:
    pil, url = _image_from_request_fields(body)
    if pil is not None:
        return pil
    if url:
        return await _download_image(url)
    raise HTTPException(status_code=400, detail="no image source in request")


def _build_depth_response(raw_u8, *, return_url: bool) -> DepthResponse:
    import numpy as np

    arr = np.asarray(raw_u8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    png = depth_png_bytes(arr)
    b64 = base64.b64encode(png).decode("ascii")

    if return_url:
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
    model = _get_depth_model()
    return_url = os.getenv("RETURN_IMAGE_URL", "").strip().lower() in {"1", "true", "yes"}
    result = await asyncio.to_thread(model.infer, image)
    return _build_depth_response(result.raw_u8, return_url=return_url)


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

    pil = await _resolve_depth_image(body)
    return await _run_depth(pil)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
