"""Benchmark settings loaded from environment via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchSettings(BaseSettings):
    """Environment-backed configuration for benchmark backends."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # fal.ai
    fal_key: str = ""
    fal_replay_dir: str = ""

    # Salad Cloud
    salad_api_key: str = ""
    salad_organization_name: str = ""
    salad_project_name: str = ""
    salad_sam3_gateway_url: str = ""
    salad_depth_gateway_url: str = ""
    salad_gpu_priority: str = ""

    # Benchmark paths
    bench_fixtures_dir: str = ""
    bench_output_dir: str = ""
    bench_http_timeout_s: float = 120.0


@lru_cache
def get_settings() -> BenchSettings:
    return BenchSettings()
