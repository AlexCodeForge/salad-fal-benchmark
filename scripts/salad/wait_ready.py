#!/usr/bin/env python3
"""Poll container groups until instances are running and readiness passes."""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from _api import SaladClient, gateway_url, group_names, require_api_key


def _group_ready(group: dict) -> tuple[bool, str]:
    state = group.get("current_state") or {}
    status = state.get("status", "unknown")
    counts = state.get("instance_status_count") or {}
    running = counts.get("running_count", 0)
    replicas = group.get("replicas", 1)

    if status == "running" and running >= replicas:
        return True, f"status=running running_count={running}/{replicas}"
    return False, f"status={status} running_count={running}/{replicas}"


def _probe_gateway(url: str, api_key: str, timeout: float) -> tuple[bool, str]:
    health = f"{url.rstrip('/')}/health"
    try:
        resp = httpx.get(
            health,
            headers={"Salad-Api-Key": api_key},
            timeout=timeout,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            return True, f"GET /health -> {resp.status_code}"
        return False, f"GET /health -> {resp.status_code}"
    except httpx.HTTPError as exc:
        return False, f"GET /health failed: {exc}"


def wait_group(
    client: SaladClient,
    name: str,
    *,
    api_key: str,
    timeout_s: float,
    poll_s: float,
    probe: bool,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_detail = ""

    while time.monotonic() < deadline:
        group = client.get_container_group(name)
        if group is None:
            last_detail = "group not found"
        else:
            ready, last_detail = _group_ready(group)
            if ready:
                gw = gateway_url(group)
                if probe and gw:
                    ok, probe_detail = _probe_gateway(gw, api_key, min(poll_s * 2, 30.0))
                    last_detail = f"{last_detail}; {probe_detail}"
                    if ok:
                        print(f"  READY  {name}: {last_detail}")
                        return
                else:
                    print(f"  READY  {name}: {last_detail}")
                    return

        remaining = int(deadline - time.monotonic())
        print(f"  waiting {name}: {last_detail} ({remaining}s left)")
        time.sleep(poll_s)

    print(f"  TIMEOUT  {name}: {last_detail}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for Salad container groups to run")
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Max wait seconds per group (default: 900)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=15.0,
        help="Poll interval seconds (default: 15)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip gateway /health probe (API state only)",
    )
    args = parser.parse_args()

    api_key = require_api_key()
    sam3_name, depth_name = group_names()
    print(f"Waiting for groups: {sam3_name}, {depth_name}")

    with SaladClient() as client:
        for name in (sam3_name, depth_name):
            wait_group(
                client,
                name,
                api_key=api_key,
                timeout_s=args.timeout,
                poll_s=args.poll,
                probe=not args.no_probe,
            )

    print("\nAll groups ready.")


if __name__ == "__main__":
    main()
