#!/usr/bin/env python3
"""Capture fal + Salad mask/depth PNGs per pipeline stage for L076 lab."""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.compare_scope import FLOOR_SAM3_STAGE, wall_sam3_stages
from bench.fal.client import FalCallTracker, FalClient, FalClientConfig
from bench.fal.constants import DEPTH_ENDPOINT, SAM3_IMAGE_ENDPOINT
from bench.fal.replay import load_replay_index, resolve_fixture_dir, resolve_photo_path, resolve_replay_dir
from bench.salad.upload import build_depth_post, build_sam3_post

SAM3_STAGES = tuple(wall_sam3_stages()) + (FLOOR_SAM3_STAGE,)

BINARY_KEYS = frozenset(
    {"image", "image_base64", "content", "data", "mask", "masks_b64"}
)


def slug_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.replace(":", "-")).strip("-")


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_json(path: Path, payload: Any) -> None:
    save_bytes(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode())


def _looks_binary(value: str) -> bool:
    if value.startswith("data:"):
        return True
    if len(value) < 120:
        return False
    sample = value[:512]
    if sample.startswith("http://") or sample.startswith("https://"):
        return False
    return sum(ch.isalnum() or ch in "+/=\n" for ch in sample) / max(len(sample), 1) > 0.85


def sanitize_trace(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_trace(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_trace(item) for item in value]
    if isinstance(value, str):
        if key in BINARY_KEYS or _looks_binary(value):
            return f"<omitted {len(value)} chars>"
        if len(value) > 400:
            return value[:400] + f"… <truncated, {len(value)} chars total>"
    return value


def data_url_to_png(data_url: str) -> bytes | None:
    if not data_url:
        return None
    payload = data_url
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload)
    except (ValueError, binascii.Error):
        return None


def _decode_mask_entry(entry: Any, client: FalClient | None = None) -> tuple[bytes | None, float | None]:
    if isinstance(entry, dict):
        url = entry.get("url") or entry.get("mask_url") or entry
        if isinstance(url, dict):
            url = url.get("url") or url.get("mask_url")
        score = entry.get("score")
    elif isinstance(entry, str):
        url = entry
        score = None
    else:
        return None, None
    if not isinstance(url, str):
        return None, None
    if url.startswith("data:"):
        png = data_url_to_png(url)
    elif client is not None:
        try:
            png = client.download_bytes(url)
        except Exception:
            png = None
    else:
        local = Path(url)
        png = local.read_bytes() if local.is_file() else None
    parsed_score = float(score) if isinstance(score, (int, float)) else None
    return png, parsed_score


def masks_from_doc(
    doc: dict,
    client: FalClient | None = None,
    *,
    limit: int | None = None,
) -> list[tuple[bytes, float | None]]:
    masks_raw = doc.get("masks") or []
    decoded: list[tuple[bytes, float | None]] = []
    for i, entry in enumerate(masks_raw):
        if limit is not None and i >= limit:
            break
        png, score = _decode_mask_entry(entry, client)
        if png:
            decoded.append((png, score))
    return decoded


def mask_from_doc(doc: dict, client: FalClient | None = None) -> bytes | None:
    masks = masks_from_doc(doc, client, limit=1)
    return masks[0][0] if masks else None


def depth_from_doc(doc: dict, client: FalClient | None = None) -> bytes | None:
    image = doc.get("image") or {}
    if isinstance(image, dict):
        content = image.get("content")
        if content:
            return base64.b64decode(content)
        url = image.get("url")
        if isinstance(url, str):
            if url.startswith("data:"):
                return data_url_to_png(url)
            if client is not None:
                try:
                    return client.download_bytes(url)
                except Exception:
                    return None
    depth_png = doc.get("depth") or doc.get("depth_map")
    if isinstance(depth_png, str) and depth_png.startswith("data:"):
        return data_url_to_png(depth_png)
    return None


def image_provenance(
    doc: dict,
    *,
    saved_to: str | None,
    saved_paths: list[str] | None = None,
    kind: str = "mask",
) -> dict[str, Any]:
    if kind == "depth":
        image = doc.get("image") or {}
        if isinstance(image, dict) and image.get("url"):
            return {
                "png_from": "response.image.url",
                "saved_to": saved_to,
                "source": "api_response",
            }
        if isinstance(image, dict) and image.get("content"):
            return {
                "png_from": "response.image.content (base64 decoded)",
                "saved_to": saved_to,
                "source": "api_response",
            }
        return {"png_from": "response", "saved_to": saved_to, "source": "api_response"}

    masks = doc.get("masks") or []
    paths = saved_paths if saved_paths is not None else ([saved_to] if saved_to else [])
    if not masks:
        return {
            "png_from": None,
            "saved_to": saved_to,
            "saved_paths": paths,
            "source": "api_response",
            "mask_count": 0,
        }
    first = masks[0] if isinstance(masks[0], dict) else {"url": masks[0]}
    url = first.get("url") or first.get("mask_url")
    preview = None
    if isinstance(url, str):
        if url.startswith("data:"):
            preview = f"data:… ({len(url)} chars)"
        elif len(url) > 120:
            preview = url[:120] + "…"
        else:
            preview = url
    scores = [
        m.get("score")
        for m in masks
        if isinstance(m, dict) and m.get("score") is not None
    ]
    return {
        "png_from": "response.masks[*].url",
        "mask_url_preview": preview,
        "mask_count": len(masks),
        "score": first.get("score"),
        "scores": scores,
        "saved_to": saved_to,
        "saved_paths": paths,
        "source": "api_response",
    }


def save_mask_pngs(
    out_dir: Path,
    slug: str,
    masks: list[tuple[bytes, float | None]],
    backend: str,
) -> list[dict[str, Any]]:
    """Save decoded masks; return manifest entries with relative paths and scores."""
    entries: list[dict[str, Any]] = []
    for i, (png, score) in enumerate(masks):
        save_bytes(out_dir / f"{slug}-{i}.png", png)
        rel = f"{backend}/{slug}-{i}.png"
        entries.append({"path": rel, "score": score})
        if i == 0:
            save_bytes(out_dir / f"{slug}.png", png)
    if entries:
        entries[0]["path"] = f"{backend}/{slug}.png"
    return entries


def fal_mask_from_replay(replay_dir: Path, label: str) -> list[tuple[bytes, float | None]]:
    index = load_replay_index(replay_dir)
    json_path = index.get(label)
    if label == "depth":
        depth_png = replay_dir / "depth.png"
        if depth_png.is_file():
            return [(depth_png.read_bytes(), None)]
        return []
    if not json_path or not json_path.is_file():
        return []
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    return masks_from_doc(doc, client=None, limit=1)


def fal_mask_live(
    client: FalClient,
    image_url: str,
    label: str,
    prompt: str,
    class_name: str,
    max_masks: int,
    tracker: FalCallTracker,
) -> tuple[list[tuple[bytes, float | None]], dict[str, Any], dict[str, Any]]:
    request = {
        "endpoint": SAM3_IMAGE_ENDPOINT,
        "arguments": {
            "image_url": image_url,
            "prompt": prompt,
            "apply_mask": False,
            "return_multiple_masks": True,
            "max_masks": max_masks,
            "include_scores": True,
            "include_boxes": True,
        },
    }
    doc = client.subscribe(
        SAM3_IMAGE_ENDPOINT,
        request["arguments"],
        label,
        tracker,
    )
    return masks_from_doc(doc, client, limit=max_masks), request, doc


def fal_depth_live(
    client: FalClient,
    image_url: str,
    tracker: FalCallTracker,
) -> tuple[bytes | None, dict[str, Any], dict[str, Any]]:
    request = {"endpoint": DEPTH_ENDPOINT, "arguments": {"image_url": image_url}}
    doc = client.subscribe(DEPTH_ENDPOINT, request["arguments"], "depth", tracker)
    return depth_from_doc(doc, client), request, doc


def salad_call(gateway: str, api_key: str, path: str, payload: dict) -> tuple[dict, dict[str, Any]]:
    url = gateway.rstrip("/") + path
    headers = {"Salad-Api-Key": api_key, "Content-Type": "application/json"}
    request_meta = {"method": "POST", "url": url, "headers": {"Salad-Api-Key": "<redacted>"}, "body": payload}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json(), request_meta


def append_trace(
    trace: list[dict[str, Any]],
    *,
    stage: str,
    backend: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    error: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "stage": stage,
        "backend": backend,
        "request": sanitize_trace(copy.deepcopy(request)),
        "response": sanitize_trace(copy.deepcopy(response)) if response is not None else None,
    }
    if error:
        entry["error"] = error
    if provenance:
        entry["image_provenance"] = provenance
    trace.append(entry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="terminados-02")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--live-fal",
        action="store_true",
        help="Call fal.ai live for mask/depth PNGs (default: fal_replay fixtures)",
    )
    parser.add_argument(
        "--backends",
        default="both",
        choices=["both", "fal", "salad"],
        help="Which backends to capture visuals for (default: both)",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=8,
        help="Max SAM3 masks to decode and save per stage (default: 8)",
    )
    args = parser.parse_args()

    if args.max_masks < 1 or args.max_masks > 32:
        print("--max-masks must be 1..32", file=sys.stderr)
        return 2

    run_fal = args.backends in ("both", "fal")
    run_salad = args.backends in ("both", "salad")

    out_root = Path(args.out_dir) / args.run_id
    trace_dir = out_root / "api_trace"
    fal_dir = out_root / "fal"
    salad_dir = out_root / "salad"

    fixture_dir = resolve_fixture_dir(args.fixture)
    photo = resolve_photo_path(fixture_dir)
    image_bytes = photo.read_bytes()
    save_bytes(out_root / "input.jpg", image_bytes)

    replay_dir = resolve_replay_dir(fixture_dir) if not args.live_fal else None

    gateway = os.environ.get("SALAD_ANALYZE_GATEWAY_URL") or os.environ.get("SALAD_GATEWAY_URL", "")
    api_key = os.environ.get("SALAD_API_KEY", "")

    api_trace: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "input": "input.jpg",
        "backends": args.backends,
        "live_fal": args.live_fal,
        "image_source_note": (
            "Las PNG mostradas se decodifican de cada mask en el response API "
            "(fal.ai live o Salad GPU). No se usan máscaras curadas del fixture salvo modo replay."
        ),
        "max_masks": args.max_masks,
        "stages": [],
        "api_trace": api_trace,
    }

    fal_client: FalClient | None = None
    fal_image_url: str | None = None
    fal_tracker = FalCallTracker()
    if run_fal and args.live_fal:
        config = FalClientConfig.from_env(replay_dir=None)
        config.replay = False
        fal_client = FalClient(config)
        try:
            fal_image_url = fal_client.upload_image_bytes(image_bytes, "image/jpeg")
            append_trace(
                api_trace,
                stage="upload",
                backend="fal",
                request={
                    "method": "POST",
                    "endpoint": "fal storage upload (initiate + PUT)",
                    "content_type": "image/jpeg",
                    "bytes": len(image_bytes),
                },
                response={"image_url": fal_image_url},
                provenance={"note": "URL usada en requests SAM3/depth"},
            )
        except Exception as exc:  # noqa: BLE001
            append_trace(
                api_trace,
                stage="upload",
                backend="fal",
                request={"method": "POST", "endpoint": "fal storage upload"},
                response=None,
                error=str(exc),
            )

    try:
        for label, _cls, prompt, _stage_max_masks in SAM3_STAGES:
            slug = slug_label(label)
            max_masks = args.max_masks
            fal_images: list[dict[str, Any]] = []
            if run_fal:
                if args.live_fal and fal_client and fal_image_url:
                    try:
                        fal_masks, fal_req, fal_resp = fal_mask_live(
                            fal_client,
                            fal_image_url,
                            label,
                            prompt,
                            _cls,
                            max_masks,
                            fal_tracker,
                        )
                        if fal_masks:
                            fal_images = save_mask_pngs(fal_dir, slug, fal_masks, "fal")
                        saved = fal_images[0]["path"] if fal_images else None
                        saved_paths = [e["path"] for e in fal_images]
                        append_trace(
                            api_trace,
                            stage=label,
                            backend="fal",
                            request=fal_req,
                            response=fal_resp,
                            provenance=image_provenance(
                                fal_resp,
                                saved_to=saved,
                                saved_paths=saved_paths,
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"fal live {label} error: {exc}", file=sys.stderr)
                        append_trace(
                            api_trace,
                            stage=label,
                            backend="fal",
                            request={"endpoint": SAM3_IMAGE_ENDPOINT, "prompt": prompt},
                            response=None,
                            error=str(exc),
                        )
                elif replay_dir is not None:
                    fal_masks = fal_mask_from_replay(replay_dir, label)
                    if fal_masks:
                        fal_images = save_mask_pngs(fal_dir, slug, fal_masks, "fal")
                    append_trace(
                        api_trace,
                        stage=label,
                        backend="fal",
                        request={"source": "fal_replay fixture", "label": label},
                        response={"source": "fal_replay fixture"},
                        provenance={
                            "source": "fal_replay fixture (no API)",
                            "saved_to": fal_images[0]["path"] if fal_images else None,
                            "saved_paths": [e["path"] for e in fal_images],
                            "mask_count": len(fal_images),
                        },
                    )

            salad_images: list[dict[str, Any]] = []
            if run_salad and gateway and api_key:
                try:
                    payload = build_sam3_post(image_bytes, prompt, max_masks, encoding="json")
                    doc, req_meta = salad_call(gateway, api_key, "/v1/sam3", payload)
                    salad_masks = masks_from_doc(doc, limit=max_masks)
                    if salad_masks:
                        salad_images = save_mask_pngs(salad_dir, slug, salad_masks, "salad")
                    saved = salad_images[0]["path"] if salad_images else None
                    saved_paths = [e["path"] for e in salad_images]
                    append_trace(
                        api_trace,
                        stage=label,
                        backend="salad",
                        request=req_meta,
                        response=doc,
                        provenance=image_provenance(
                            doc,
                            saved_to=saved,
                            saved_paths=saved_paths,
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"salad {label} error: {exc}", file=sys.stderr)
                    append_trace(
                        api_trace,
                        stage=label,
                        backend="salad",
                        request={"url": f"{gateway.rstrip('/')}/v1/sam3", "prompt": prompt},
                        response=None,
                        error=str(exc),
                    )

            stage_entry: dict[str, Any] = {
                "label": label,
                "slug": slug,
                "fal_image": fal_images[0]["path"] if fal_images else None,
                "salad_image": salad_images[0]["path"] if salad_images else None,
            }
            if fal_images:
                stage_entry["fal_images"] = fal_images
            if salad_images:
                stage_entry["salad_images"] = salad_images
            manifest["stages"].append(stage_entry)

        slug = "depth"
        fal_depth = None
        if run_fal:
            if args.live_fal and fal_client and fal_image_url:
                try:
                    fal_depth, fal_req, fal_resp = fal_depth_live(fal_client, fal_image_url, fal_tracker)
                    saved = f"fal/{slug}.png" if fal_depth else None
                    append_trace(
                        api_trace,
                        stage="depth",
                        backend="fal",
                        request=fal_req,
                        response=fal_resp,
                        provenance=image_provenance(fal_resp, saved_to=saved, kind="depth"),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"fal live depth error: {exc}", file=sys.stderr)
                    append_trace(
                        api_trace,
                        stage="depth",
                        backend="fal",
                        request={"endpoint": DEPTH_ENDPOINT},
                        response=None,
                        error=str(exc),
                    )
            elif replay_dir is not None:
                fal_depth_masks = fal_mask_from_replay(replay_dir, "depth")
                fal_depth = fal_depth_masks[0][0] if fal_depth_masks else None
                append_trace(
                    api_trace,
                    stage="depth",
                    backend="fal",
                    request={"source": "fal_replay fixture"},
                    response={"source": "fal_replay fixture"},
                    provenance={"source": "fal_replay fixture", "saved_to": f"fal/{slug}.png" if fal_depth else None},
                )
            if fal_depth:
                save_bytes(fal_dir / f"{slug}.png", fal_depth)

        salad_depth = None
        if run_salad and gateway and api_key:
            try:
                payload = build_depth_post(image_bytes, encoding="json")
                doc, req_meta = salad_call(gateway, api_key, "/v1/depth", payload)
                salad_depth = depth_from_doc(doc)
                saved = f"salad/{slug}.png" if salad_depth else None
                append_trace(
                    api_trace,
                    stage="depth",
                    backend="salad",
                    request=req_meta,
                    response=doc,
                    provenance=image_provenance(doc, saved_to=saved, kind="depth"),
                )
                if salad_depth:
                    save_bytes(salad_dir / f"{slug}.png", salad_depth)
            except Exception as exc:  # noqa: BLE001
                print(f"salad depth error: {exc}", file=sys.stderr)
                append_trace(
                    api_trace,
                    stage="depth",
                    backend="salad",
                    request={"url": f"{gateway.rstrip('/')}/v1/depth"},
                    response=None,
                    error=str(exc),
                )

        manifest["stages"].append(
            {
                "label": "depth",
                "slug": slug,
                "fal_image": f"fal/{slug}.png" if fal_depth else None,
                "salad_image": f"salad/{slug}.png" if salad_depth else None,
            }
        )
    finally:
        if fal_client is not None:
            fal_client.close()

    save_json(trace_dir / "trace.json", api_trace)
    for i, entry in enumerate(api_trace):
        fname = f"{i:02d}_{slug_label(entry['stage'])}_{entry['backend']}.json"
        save_json(trace_dir / fname, entry)

    save_json(out_root / "manifest.json", manifest)
    print(json.dumps({"ok": True, "run_id": args.run_id, "visuals": manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
