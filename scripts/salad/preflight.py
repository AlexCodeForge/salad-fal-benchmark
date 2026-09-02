#!/usr/bin/env python3
"""Preflight: GPU classes, quotas, and container-group name checks."""

from __future__ import annotations

import argparse
import json
import sys

from _api import (
    BENCHMARK_GPU_CLASSES,
    GPU_CLASS_3090,
    GPU_CLASS_4090,
    SaladClient,
    depth_group_name,
    group_names,
    org_name,
    project_name,
    sam3_group_name,
)

GPU_LABELS = {
    GPU_CLASS_4090: "RTX 4090",
    GPU_CLASS_3090: "RTX 3090",
}


def _print_gpu_classes(client: SaladClient) -> dict[str, dict]:
    items = client.list_gpu_classes()
    by_id = {item["id"]: item for item in items if "id" in item}

    print("GPU classes (organization):")
    for gpu_id in BENCHMARK_GPU_CLASSES:
        label = GPU_LABELS.get(gpu_id, gpu_id)
        item = by_id.get(gpu_id)
        if item:
            name = item.get("name") or item.get("display_name") or "?"
            print(f"  OK  {label}: {gpu_id} ({name})")
        else:
            print(f"  MISSING  {label}: {gpu_id} — not in API list", file=sys.stderr)

    if len(items) > len(BENCHMARK_GPU_CLASSES):
        print("\nOther available GPU classes:")
        for item in items:
            gid = item.get("id", "?")
            if gid in BENCHMARK_GPU_CLASSES:
                continue
            name = item.get("name") or item.get("display_name") or "?"
            print(f"  - {gid} ({name})")

    return by_id


def _print_quotas(client: SaladClient) -> None:
    quotas = client.get_quotas()
    print("\nOrganization quotas:")
    print(json.dumps(quotas, indent=2, sort_keys=True))


def _check_group_names(client: SaladClient) -> None:
    sam3_name, depth_name = group_names()
    existing = {g.get("name"): g for g in client.list_container_groups()}
    print("\nContainer group name check:")
    for label, name in (("sam3", sam3_name), ("depth", depth_name)):
        group = existing.get(name)
        if group:
            state = (group.get("current_state") or {}).get("status", "?")
            print(f"  EXISTS  {label}: {name} (status={state})")
        else:
            print(f"  FREE    {label}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Salad deploy preflight checks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target org/project/groups only; skip API calls",
    )
    args = parser.parse_args()

    sam3_name, depth_name = group_names()
    print(f"Organization: {org_name()}")
    print(f"Project:      {project_name()}")
    print(f"Groups:       sam3={sam3_name}, depth={depth_name}")

    if args.dry_run:
        print("\n(dry-run — no API calls)")
        return

    with SaladClient() as client:
        by_id = _print_gpu_classes(client)
        missing = [gid for gid in BENCHMARK_GPU_CLASSES if gid not in by_id]
        _print_quotas(client)
        _check_group_names(client)

    if missing:
        labels = ", ".join(GPU_LABELS.get(g, g) for g in missing)
        print(f"\nPreflight FAILED: missing GPU classes: {labels}", file=sys.stderr)
        sys.exit(1)

    print("\nPreflight OK")


if __name__ == "__main__":
    main()
