#!/usr/bin/env python3
"""Patch running Salad container group with HF_TOKEN and restart replicas."""

from __future__ import annotations

import os
import sys
import time

import httpx

from _api import SaladClient, raise_api_error

GROUP = os.environ.get("SALAD_ANALYZE_GROUP_NAME", "bench-analyze-gpu").strip()


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    ).strip()
    if not token:
        print("HF_TOKEN missing — add to .env then: source .env && python scripts/salad/patch_hf.py")
        sys.exit(1)

    with SaladClient() as client:
        group = client.get_container_group(GROUP)
        if not group:
            print(f"Container group {GROUP!r} not found")
            sys.exit(1)

        payload = {k: v for k, v in group.items() if k not in ("id", "version", "current_state")}
        payload.pop("name", None)
        env = payload.setdefault("container", {}).setdefault("environment_variables", {})
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
        env["MOCK_INFERENCE"] = "0"

        print(f"Patching {GROUP} with HF_TOKEN…")
        updated = client.update_container_group(GROUP, payload)
        print(f"Updated version={updated.get('version', '?')}")

        gateway = os.environ.get("SALAD_ANALYZE_GATEWAY_URL", "").rstrip("/")
        if not gateway:
            print("SALAD_ANALYZE_GATEWAY_URL not set — skip health poll")
            return

        key = os.environ["SALAD_API_KEY"]
        print("Waiting for SAM3 load (max 3 min)…")
        for i in range(36):
            time.sleep(5)
            try:
                r = httpx.get(
                    f"{gateway}/health",
                    headers={"Salad-Api-Key": key},
                    timeout=15,
                )
                if r.status_code != 200:
                    print(f"  [{i+1}] health HTTP {r.status_code}")
                    continue
                sam3 = r.json().get("sam3", {})
                loaded = sam3.get("model_loaded")
                print(f"  [{i+1}] sam3_loaded={loaded} mock={sam3.get('mock_inference')}")
                if loaded:
                    print("SAM3 ready.")
                    return
            except httpx.HTTPError as exc:
                print(f"  [{i+1}] {exc}")

        print("Timeout — check Salad logs. SAM3 may still be downloading weights.")
        sys.exit(2)


if __name__ == "__main__":
    main()
