#!/usr/bin/env python3
"""Teardown legacy and unified analyze SCE container groups."""

from __future__ import annotations

import argparse

from _api import SaladClient, all_group_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete bench-sam3, bench-depth, and bench-analyze container groups"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    targets = all_group_names()

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
