#!/usr/bin/env python3
"""Print Salad gateway URLs for export into .env."""

from __future__ import annotations

import sys

from _api import SaladClient, gateway_url, group_names


def main() -> None:
    sam3_name, depth_name = group_names()

    with SaladClient() as client:
        sam3 = client.get_container_group(sam3_name)
        depth = client.get_container_group(depth_name)

    if not sam3:
        print(f"Container group not found: {sam3_name}", file=sys.stderr)
        sys.exit(1)
    if not depth:
        print(f"Container group not found: {depth_name}", file=sys.stderr)
        sys.exit(1)

    sam3_url = gateway_url(sam3)
    depth_url = gateway_url(depth)

    if not sam3_url:
        print(f"No gateway DNS on {sam3_name}", file=sys.stderr)
        sys.exit(1)
    if not depth_url:
        print(f"No gateway DNS on {depth_name}", file=sys.stderr)
        sys.exit(1)

    print(f"export SALAD_SAM3_GATEWAY_URL={sam3_url}")
    print(f"export SALAD_DEPTH_GATEWAY_URL={depth_url}")
    print()
    print("# Add to .env:")
    print(f"SALAD_SAM3_GATEWAY_URL={sam3_url}")
    print(f"SALAD_DEPTH_GATEWAY_URL={depth_url}")


if __name__ == "__main__":
    main()
