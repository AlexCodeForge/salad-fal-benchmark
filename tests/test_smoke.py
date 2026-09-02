"""Smoke tests: package imports and CLI entry point."""

import bench
from bench.cli import app


def test_import_bench():
    assert bench.__version__ == "0.1.0"


def test_cli_app_is_typer():
    assert app.info.name == "bench"
