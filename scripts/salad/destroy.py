#!/usr/bin/env python3
"""Teardown SAM3 and depth SCE container groups."""

from __future__ import annotations

import argparse
import sys

from _api import SaladClient, group_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete bench SAM3 + depth container groups")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    sam3_name, depth_name = group_names()
    targets = (sam3_name, depth_name)

    if not args.yes:
        answer = input(f"Delete container groups {targets}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    with SaladClient() as client:
        for name in targets:
            deleted = client.delete_container_group(name)
            if deleted:
                print(f"  deleted {name}")
            else:
                print(f"  not found {name} (skipped)")

    print("\nDestroy complete.")


if __name__ == "__main__":
    main()
