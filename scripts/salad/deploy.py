#!/usr/bin/env python3
"""Deploy unified analyze SCE container group via Salad API."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from _api import CONFIG_DIR, SaladClient, analyze_group_name, raise_api_error

DEPLOY_LABEL = "analyze"
DEPLOY_TEMPLATE = "analyze-group.yaml"
DEPLOY_IMAGE_ENV = "SALAD_ANALYZE_IMAGE"


def _load_template(filename: str) -> dict[str, Any]:
    path = Path(CONFIG_DIR) / filename
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _apply_overrides(spec: dict[str, Any], *, image: str, group_name: str) -> dict[str, Any]:
    body = copy.deepcopy(spec)
    body["name"] = group_name
    body["display_name"] = group_name
    body.setdefault("container", {})["image"] = image

    priority = os.environ.get("SALAD_GPU_PRIORITY", "").strip()
    if priority:
        body["container"]["priority"] = priority

    registry_user = os.environ.get("DOCKER_REGISTRY_USERNAME", "").strip()
    registry_pass = os.environ.get("DOCKER_REGISTRY_PASSWORD", "").strip()
    if registry_user and registry_pass:
        body["container"]["registry_authentication"] = {
            "docker_hub": {
                "username": registry_user,
                "personal_access_token": registry_pass,
            }
        }

    return body


def _update_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(body)
    payload.pop("name", None)
    return payload


def deploy_group(
    client: SaladClient | None,
    *,
    label: str,
    body: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    name = body["name"]
    image = body["container"]["image"]
    print(f"\n[{label}] group={name} image={image}")

    if dry_run:
        print(json.dumps(body, indent=2))
        return body

    assert client is not None
    existing = client.get_container_group(name)
    if existing:
        print(f"  patching existing group {name}")
        try:
            updated = client.update_container_group(name, _update_payload(body))
        except Exception:
            resp = client.patch(
                client._project_path(f"/containers/{name}"),
                _update_payload(body),
            )
            if resp.status_code >= 400:
                raise_api_error(resp, f"update {name}")
            updated = resp.json()
        print(f"  updated (version={updated.get('version', '?')})")
        return updated

    try:
        created = client.create_container_group(body)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 409:
            raise
        print(f"  conflict — patching existing {name}")
        created = client.update_container_group(name, _update_payload(body))
    print(f"  created (id={created.get('id', '?')})")
    return created


def _build_spec() -> tuple[str, dict[str, Any]]:
    image = os.environ.get(DEPLOY_IMAGE_ENV, "").strip()
    if not image:
        print(f"{DEPLOY_IMAGE_ENV} must be set", file=sys.stderr)
        sys.exit(1)
    group = analyze_group_name()
    template = _load_template(DEPLOY_TEMPLATE)
    return DEPLOY_LABEL, _apply_overrides(template, image=image, group_name=group)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy bench-analyze container group")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print create/update payloads without calling Salad API",
    )
    args = parser.parse_args()

    label, body = _build_spec()

    if args.dry_run:
        deploy_group(client=None, label=label, body=body, dry_run=True)
        return

    with SaladClient() as client:
        deploy_group(client, label=label, body=body, dry_run=False)

    print("\nDeploy complete. Run wait_ready.py then print_gateways.py.")


if __name__ == "__main__":
    main()
