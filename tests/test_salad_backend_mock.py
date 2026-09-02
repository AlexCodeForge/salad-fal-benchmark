"""Mocked Salad gateway integration tests (httpx MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from bench.backends.salad import SaladBackend, segment_salad
from bench.config import BenchSettings
from bench.salad.client import MAX_RETRIES, SaladGatewayClient
from bench.salad.upload import build_sam3_post, encode_image_base64

ANALYZE_URL = "https://analyze.example.test"
SAM3_URL = "https://sam3.example.test"
DEPTH_URL = "https://depth.example.test"
API_KEY = "test-salad-key"

MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7),01444\x1f\x27=9=82<.342\xff\xc0"
    b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00"
    b"\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
    b"\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01"
    b"\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04"
    b"\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15"
    b"R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVW"
    b"XYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
    b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4"
    b"\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3"
    b"\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea"
    b"\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00"
    b"\x00?\x00\xfe\x8a(\xa0\x0f\xff\xd9"
)

SAM3_RESPONSE = {
    "masks": [{"url": "https://cdn.example/mask0.png"}],
    "scores": [0.9],
    "boxes": [[0.1, 0.2, 0.5, 0.6]],
}
DEPTH_RESPONSE = {
    "image": {
        "url": "https://cdn.example/depth.png",
        "content_type": "image/png",
        "width": 1,
        "height": 1,
    }
}


@pytest.fixture
def unified_settings() -> BenchSettings:
    return BenchSettings(
        salad_api_key=API_KEY,
        salad_analyze_gateway_url=ANALYZE_URL,
        salad_sam3_gateway_url="",
        salad_depth_gateway_url="",
        bench_http_timeout_s=120.0,
    )


@pytest.fixture
def dual_settings() -> BenchSettings:
    return BenchSettings(
        salad_api_key=API_KEY,
        salad_sam3_gateway_url=SAM3_URL,
        salad_depth_gateway_url=DEPTH_URL,
        bench_http_timeout_s=120.0,
    )


def test_encode_image_base64_roundtrip() -> None:
    encoded = encode_image_base64(MINIMAL_JPEG)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_build_sam3_post_json_encoding() -> None:
    payload = build_sam3_post(MINIMAL_JPEG, "wall", 8, encoding="json")
    assert payload["prompt"] == "wall"
    assert payload["max_masks"] == 8
    assert "image" in payload


def test_client_retries_502_then_succeeds() -> None:
    attempts = {"count": 0}
    seen_api_key: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        seen_api_key.append(request.headers.get("Salad-Api-Key"))
        if attempts["count"] < MAX_RETRIES:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with SaladGatewayClient(API_KEY, transport=transport) as client:
        result = client.post_json("https://gateway.test/v1/sam3", {"prompt": "wall"})

    assert result == {"ok": True}
    assert attempts["count"] == MAX_RETRIES
    assert seen_api_key[0] == API_KEY


def test_client_retries_503_exhausted_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    transport = httpx.MockTransport(handler)
    with SaladGatewayClient(API_KEY, transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.post_json("https://gateway.test/v1/sam3", {"prompt": "wall"})

    assert exc_info.value.response.status_code == 503


def test_segment_salad_unified_gateway(unified_settings: BenchSettings) -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Salad-Api-Key") == API_KEY
        path = request.url.path
        hosts.append(request.url.host or "")
        if path.endswith("/v1/sam3"):
            return httpx.Response(200, json=SAM3_RESPONSE)
        if path.endswith("/v1/depth"):
            return httpx.Response(200, json=DEPTH_RESPONSE)
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    client = SaladGatewayClient(API_KEY, transport=transport)
    output = segment_salad(MINIMAL_JPEG, settings=unified_settings, client=client)
    client.close()

    assert len(output.stages) == 5
    assert all(stage.success for stage in output.stages)
    assert len(hosts) == 5
    assert all(host == "analyze.example.test" for host in hosts)


def test_segment_salad_dual_gateway_compat(dual_settings: BenchSettings) -> None:
    sam3_calls: list[str] = []
    depth_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Salad-Api-Key") == API_KEY
        path = request.url.path
        if path.endswith("/v1/sam3"):
            sam3_calls.append(request.url.host or "")
            return httpx.Response(200, json=SAM3_RESPONSE)
        if path.endswith("/v1/depth"):
            depth_calls.append(request.url.host or "")
            return httpx.Response(200, json=DEPTH_RESPONSE)
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    client = SaladGatewayClient(API_KEY, transport=transport)
    output = segment_salad(MINIMAL_JPEG, settings=dual_settings, client=client)
    client.close()

    assert len(output.stages) == 5
    assert [stage.label for stage in output.stages] == [
        "wall:wall:sam3",
        "wall:molding:sam3",
        "wall:mullion:sam3",
        "floor:floor:sam3",
        "depth",
    ]
    assert all(stage.success for stage in output.stages)
    assert len(sam3_calls) == 4
    assert len(depth_calls) == 1
    assert sam3_calls[0] == "sam3.example.test"
    assert depth_calls[0] == "depth.example.test"


def test_salad_backend_requires_gateway_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SALAD_ANALYZE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("SALAD_GATEWAY_URL", raising=False)
    monkeypatch.delenv("SALAD_SAM3_GATEWAY_URL", raising=False)
    monkeypatch.delenv("SALAD_DEPTH_GATEWAY_URL", raising=False)
    settings = BenchSettings(salad_api_key=API_KEY)
    backend = SaladBackend(settings=settings)
    with pytest.raises(ValueError, match="SALAD_ANALYZE_GATEWAY_URL"):
        backend.segment(MINIMAL_JPEG)
    backend.close()
