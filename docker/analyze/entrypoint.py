#!/usr/bin/env python3
"""Uvicorn on [::]:8000 with dual-stack for Salad IPv6 + local docker -p."""

from __future__ import annotations

import os
import socket

import uvicorn

HOST = os.environ.get("HOST", "::")
PORT = int(os.environ.get("PORT", "8000"))


def main() -> None:
    if HOST == "::":
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen()
        config = uvicorn.Config(
            "app.main:app",
            fd=sock.fileno(),
            log_level=os.environ.get("LOG_LEVEL", "info"),
        )
    else:
        config = uvicorn.Config(
            "app.main:app",
            host=HOST,
            port=PORT,
            log_level=os.environ.get("LOG_LEVEL", "info"),
        )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
