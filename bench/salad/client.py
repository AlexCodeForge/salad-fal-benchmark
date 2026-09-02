"""httpx client for Salad GPU gateway endpoints."""

from __future__ import annotations

import httpx

RETRY_STATUS_CODES = frozenset({502, 503})
MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 120.0
SALAD_API_KEY_HEADER = "Salad-Api-Key"


class SaladGatewayClient:
    """Sync httpx client with Salad-Api-Key auth and 502/503 retries."""

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {SALAD_API_KEY_HEADER: api_key} if api_key else {}
        client_kwargs: dict = {"timeout": timeout, "headers": headers}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)
        self._timeout = timeout

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SaladGatewayClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def post_json(self, url: str, payload: dict) -> dict:
        response = self._request("POST", url, json=payload)
        return response.json()

    def post_multipart(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> dict:
        response = self._request("POST", url, data=data or {}, files=files or {})
        return response.json()

    def get_bytes(self, url: str) -> bytes:
        response = self._request("GET", url)
        return response.content

    def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            response = self._client.request(method, url, **kwargs)
            if response.status_code not in RETRY_STATUS_CODES:
                response.raise_for_status()
                return response
            last_response = response
            if attempt == MAX_RETRIES - 1:
                break
        assert last_response is not None
        last_response.raise_for_status()
        return last_response
