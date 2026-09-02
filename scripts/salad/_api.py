"""Shared Salad Cloud API helpers (httpx, Salad-Api-Key header)."""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx

API_BASE = "https://api.salad.com/api/public"

GPU_CLASS_4090 = "ed563892-aacd-40f5-80b7-90c9be6c759b"
GPU_CLASS_3090 = "a5db5c50-cbcb-4596-ae80-6a0c8090d80f"
BENCHMARK_GPU_CLASSES = (GPU_CLASS_4090, GPU_CLASS_3090)

DEFAULT_ORG = "ariseweb"
DEFAULT_PROJECT = "default"
DEFAULT_SAM3_GROUP = "bench-sam3"
DEFAULT_DEPTH_GROUP = "bench-depth"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "salad")


def require_api_key() -> str:
    key = os.environ.get("SALAD_API_KEY", "").strip()
    if not key:
        print("SALAD_API_KEY must be set", file=sys.stderr)
        sys.exit(1)
    return key


def org_name() -> str:
    return os.environ.get("SALAD_ORGANIZATION_NAME", DEFAULT_ORG).strip() or DEFAULT_ORG


def project_name() -> str:
    return os.environ.get("SALAD_PROJECT_NAME", DEFAULT_PROJECT).strip() or DEFAULT_PROJECT


def sam3_group_name() -> str:
    return os.environ.get("SALAD_SAM3_GROUP_NAME", DEFAULT_SAM3_GROUP).strip() or DEFAULT_SAM3_GROUP


def depth_group_name() -> str:
    return os.environ.get("SALAD_DEPTH_GROUP_NAME", DEFAULT_DEPTH_GROUP).strip() or DEFAULT_DEPTH_GROUP


def group_names() -> tuple[str, str]:
    return sam3_group_name(), depth_group_name()


class SaladClient:
    def __init__(self, api_key: str | None = None, *, timeout: float = 60.0) -> None:
        self._api_key = api_key or require_api_key()
        self._client = httpx.Client(
            base_url=API_BASE,
            headers={"Salad-Api-Key": self._api_key},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SaladClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _org_path(self, suffix: str) -> str:
        return f"/organizations/{org_name()}{suffix}"

    def _project_path(self, suffix: str) -> str:
        return f"/organizations/{org_name()}/projects/{project_name()}{suffix}"

    def get(self, path: str) -> httpx.Response:
        return self._client.get(path)

    def post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return self._client.post(path, json=body)

    def patch(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return self._client.patch(
            path,
            json=body,
            headers={"Content-Type": "application/merge-patch+json"},
        )

    def delete(self, path: str) -> httpx.Response:
        return self._client.delete(path)

    def list_gpu_classes(self) -> list[dict[str, Any]]:
        resp = self.get(self._org_path("/gpu-classes"))
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", data if isinstance(data, list) else [])

    def get_quotas(self) -> dict[str, Any]:
        resp = self.get(self._org_path("/quotas"))
        resp.raise_for_status()
        return resp.json()

    def list_container_groups(self) -> list[dict[str, Any]]:
        resp = self.get(self._project_path("/containers"))
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    def get_container_group(self, name: str) -> dict[str, Any] | None:
        resp = self.get(self._project_path(f"/containers/{name}"))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def create_container_group(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.post(self._project_path("/containers"), body)
        resp.raise_for_status()
        return resp.json()

    def update_container_group(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.patch(self._project_path(f"/containers/{name}"), body)
        resp.raise_for_status()
        return resp.json()

    def delete_container_group(self, name: str) -> bool:
        resp = self.delete(self._project_path(f"/containers/{name}"))
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def list_instances(self, group_name: str) -> list[dict[str, Any]]:
        resp = self.get(
            self._project_path(f"/containers/{group_name}/instances")
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("instances", [])


def gateway_url(group: dict[str, Any]) -> str | None:
    networking = group.get("networking") or {}
    dns = networking.get("dns")
    if not dns:
        return None
    if dns.startswith("http://") or dns.startswith("https://"):
        return dns.rstrip("/")
    return f"https://{dns}".rstrip("/")


def raise_api_error(resp: httpx.Response, action: str) -> None:
    detail = resp.text.strip()
    print(f"{action} failed ({resp.status_code}): {detail}", file=sys.stderr)
    sys.exit(1)
