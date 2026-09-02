#!/usr/bin/env python3
"""Deploy SAM3 and depth SCE container groups via Salad API."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from _api import (
    CONFIG_DIR,
    SaladClient,
    depth_group_name,
    raise_api_error,
    sam3_group_name,
)

DEPLOYS: tuple[tuple[str, str, str], ...] = (
    ("sam3", "sam3-group.yaml", "SALAD_SAM3_IMAGE"),
    ("depth", "depth-group.yaml", "SALAD_DEPTH_IMAGE"),
)


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
        print(f"  updating existing group {name}")
        resp = client.patch(
            client._project_path(f"/containers/{name}"),
            _update_payload(body),
        )
        if resp.status_code >= 400:
            raise_api_error(resp, f"update {name}")
        updated = resp.json()
        print(f"  updated (version={updated.get('version', '?')})")
        return updated

    resp = client.post(client._project_path("/containers"), body)
    if resp.status_code == 409:
        print(f"  conflict — updating {name}")
        resp = client.patch(
            client._project_path(f"/containers/{name}"),
            _update_payload(body),
        )
    if resp.status_code >= 400:
        raise_api_error(resp, f"deploy {name}")
    created = resp.json()
    print(f"  created (id={created.get('id', '?')})")
    return created


def _build_specs() -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    for label, template_file, image_env in DEPLOYS:
        image = os.environ.get(image_env, "").strip()
        if not image:
            print(f"{image_env} must be set", file=sys.stderr)
            sys.exit(1)
        group = sam3_group_name() if label == "sam3" else depth_group_name()
        template = _load_template(template_file)
        specs.append((label, _apply_overrides(template, image=image, group_name=group)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy bench SAM3 + depth container groups")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print create/update payloads without calling Salad API",
    )
    args = parser.parse_args()

    specs = _build_specs()

    if args.dry_run:
        for label, body in specs:
            deploy_group(client=None, label=label, body=body, dry_run=True)
        return

    with SaladClient() as client:
        for label, body in specs:
            deploy_group(client, label=label, body=body, dry_run=False)

    print("\nDeploy complete. Run wait_ready.py then print_gateways.py.")


if __name__ == "__main__":
    main()
