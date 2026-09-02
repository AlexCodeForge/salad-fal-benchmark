"""Benchmark settings loaded from environment via pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchSettings(BaseSettings):
    """Environment-backed configuration for benchmark backends."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # fal.ai
    fal_key: str = ""
    fal_replay_dir: str = ""

    # Salad Cloud
    salad_api_key: str = ""
    salad_organization_name: str = ""
    salad_project_name: str = ""
    salad_analyze_gateway_url: str = ""
    salad_gateway_url: str = Field(default="", validation_alias="SALAD_GATEWAY_URL")
    salad_sam3_gateway_url: str = ""
    salad_depth_gateway_url: str = ""
    salad_gpu_priority: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coalesce_analyze_gateway(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        merged = dict(data)
        analyze = str(merged.get("salad_analyze_gateway_url") or "").strip()
        gateway = str(merged.get("salad_gateway_url") or merged.get("SALAD_GATEWAY_URL") or "").strip()
        if not analyze and gateway:
            merged["salad_analyze_gateway_url"] = gateway
        return merged

    def resolved_analyze_gateway_url(self) -> str:
        return self.salad_analyze_gateway_url.strip() or self.salad_gateway_url.strip()

    # Benchmark paths
    bench_fixtures_dir: str = ""
    bench_output_dir: str = ""
    bench_http_timeout_s: float = 120.0


@lru_cache
def get_settings() -> BenchSettings:
    return BenchSettings()
