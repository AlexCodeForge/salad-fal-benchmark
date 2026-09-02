"""fal.ai storage upload and queue subscribe client (httpx)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench.fal.constants import fal_price_usd
from bench.fal.replay import load_replay_index

FAL_STORAGE_INITIATE_URL = "https://rest.fal.ai/storage/upload/initiate"
FAL_QUEUE_BASE = "https://queue.fal.run"
MAX_ATTEMPTS = 10
BASE_RETRY_DELAY_MS = 100
MAX_RETRY_DELAY_MS = 30_000
QUEUE_POLL_INTERVAL_S = 0.1
HTTP_CONNECT_TIMEOUT_S = 30.0
HTTP_REQUEST_TIMEOUT_S = 120.0
USER_AGENT = "salad-fal-benchmark/0.1"


class FalApiError(Exception):
    """fal.ai API or replay error."""


@dataclass
class FalClientConfig:
    fal_key: str | None = None
    replay: bool = False
    record: bool = False
    replay_dir: Path | None = None

    @classmethod
    def from_env(cls, replay_dir: Path | None = None) -> FalClientConfig:
        env_replay_dir = os.environ.get("FAL_REPLAY_DIR", "").strip()
        resolved_replay_dir = replay_dir or (Path(env_replay_dir) if env_replay_dir else None)
        replay_flag = os.environ.get("FAL_REPLAY", "").strip() == "1"
        replay = replay_flag and resolved_replay_dir is not None
        return cls(
            fal_key=os.environ.get("FAL_KEY") or None,
            replay=replay,
            record=os.environ.get("FAL_RECORD", "").strip() == "1",
            replay_dir=resolved_replay_dir,
        )


@dataclass
class FalCallRecord:
    endpoint: str
    label: str
    duration_ms: float
    price_usd: float


@dataclass
class FalCallTracker:
    calls: list[FalCallRecord] = field(default_factory=list)

    def record(self, endpoint: str, label: str, duration_ms: float) -> None:
        self.calls.append(
            FalCallRecord(
                endpoint=endpoint,
                label=label,
                duration_ms=round(duration_ms * 10.0) / 10.0,
                price_usd=fal_price_usd(endpoint),
            )
        )

    def cost_usd(self) -> float:
        total = sum(c.price_usd for c in self.calls)
        return round(total * 10_000.0) / 10_000.0


class FalClient:
    def __init__(self, config: FalClientConfig) -> None:
        self.config = config
        self._http = httpx.Client(
            timeout=httpx.Timeout(HTTP_REQUEST_TIMEOUT_S, connect=HTTP_CONNECT_TIMEOUT_S),
            headers={"User-Agent": USER_AGENT},
        )
        self._replay_index: dict[str, Path] = {}
        if config.replay:
            if config.replay_dir is None:
                raise FalApiError("FAL_REPLAY=1 requires replay_dir")
            self._replay_index = load_replay_index(config.replay_dir)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FalClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def upload_image_bytes(self, image_bytes: bytes, content_type: str = "image/jpeg") -> str:
        if self.config.replay:
            replay_dir = self.config.replay_dir
            assert replay_dir is not None
            upload_path = replay_dir / "upload.json"
            if upload_path.is_file():
                payload = json.loads(upload_path.read_text(encoding="utf-8"))
                return _extract_upload_url(payload)
            digest = hashlib.sha256(image_bytes).hexdigest()
            return f"replay://upload/{digest}"

        key = self._require_key()
        ext = content_type.split("/", 1)[-1] if "/" in content_type else "bin"
        ext = ext or "bin"
        auth = f"Key {key}"

        init_resp = self._http_with_retry(
            "upload/initiate",
            lambda: self._http.post(
                FAL_STORAGE_INITIATE_URL,
                headers={"Authorization": auth},
                json={
                    "content_type": content_type,
                    "file_name": f"upload.{ext}",
                },
            ),
        )
        init_body = init_resp.json()
        if not init_resp.is_success:
            raise FalApiError(f"upload initiate status {init_resp.status_code}: {init_body}")

        upload_url = init_body.get("upload_url")
        file_url = init_body.get("file_url")
        if not isinstance(upload_url, str) or not isinstance(file_url, str):
            raise FalApiError("upload initiate missing upload_url or file_url")

        put_resp = self._http_with_retry(
            "upload/put",
            lambda: self._http.put(
                upload_url,
                headers={"Content-Type": content_type},
                content=image_bytes,
            ),
        )
        if not put_resp.is_success:
            detail = put_resp.text
            raise FalApiError(f"upload put status {put_resp.status_code}: {detail}")

        if self.config.record and self.config.replay_dir is not None:
            record_path = self.config.replay_dir / "upload.json"
            record_path.write_text(
                json.dumps({"url": file_url}, indent=2),
                encoding="utf-8",
            )
        return file_url

    def subscribe(
        self,
        endpoint: str,
        arguments: dict[str, Any],
        label: str,
        tracker: FalCallTracker,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.config.replay:
            result = self._replay_subscribe(label)
        else:
            result = self._live_subscribe(endpoint, arguments)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if label:
            tracker.record(endpoint, label, elapsed_ms)
        if self.config.record and self.config.replay_dir is not None and label:
            safe = label.replace(":", "-")
            out_path = self.config.replay_dir / f"{safe}.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def download_bytes(self, url: str) -> bytes:
        if url.startswith("file:"):
            path = Path(url.removeprefix("file:"))
            return path.read_bytes()

        if self.config.replay_dir is not None:
            rel = self.config.replay_dir / url.removeprefix("./")
            if rel.is_file():
                return rel.read_bytes()

        if url.startswith("http://") or url.startswith("https://"):
            resp = self._http_with_retry("download", lambda: self._http.get(url))
            if not resp.is_success:
                raise FalApiError(f"download status {resp.status_code} fetching {url}")
            return resp.content

        raise FalApiError(f"unsupported download url {url}")

    def _replay_subscribe(self, label: str) -> dict[str, Any]:
        path = self._replay_index.get(label)
        if path is None:
            raise FalApiError(f"no replay fixture for label {label}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _live_subscribe(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = self._require_key()
        auth = f"Key {key}"
        submit_url = f"{FAL_QUEUE_BASE}/{endpoint}"

        submit_resp = self._http_with_retry(
            endpoint,
            lambda: self._http.post(
                submit_url,
                headers={"Authorization": auth},
                json=arguments,
            ),
        )
        if not submit_resp.is_success:
            detail = submit_resp.text
            raise FalApiError(f"queue submit status {submit_resp.status_code}: {detail}")

        handle = submit_resp.json()
        request_id = handle.get("request_id", "")
        status_url = handle.get("status_url")
        response_url = handle.get("response_url")
        if not isinstance(status_url, str) or not isinstance(response_url, str):
            raise FalApiError(f"queue submit missing urls (request {request_id})")

        while True:
            status_resp = self._http_with_retry(
                endpoint,
                lambda: self._http.get(status_url, headers={"Authorization": auth}),
            )
            if not status_resp.is_success:
                detail = status_resp.text
                raise FalApiError(
                    f"queue status {status_resp.status_code} (request {request_id}): {detail}"
                )
            status_body = status_resp.json()
            status = status_body.get("status")
            if status == "COMPLETED":
                err = status_body.get("error")
                if isinstance(err, str) and err:
                    raise FalApiError(
                        f"queue completed with error (request {request_id}): {err}"
                    )
                break
            if status in ("IN_QUEUE", "IN_PROGRESS"):
                time.sleep(QUEUE_POLL_INTERVAL_S)
                continue
            raise FalApiError(
                f"unknown queue status {status!r} (request {request_id}): {status_body}"
            )

        result_resp = self._http_with_retry(
            endpoint,
            lambda: self._http.get(response_url, headers={"Authorization": auth}),
        )
        if not result_resp.is_success:
            detail = result_resp.text
            raise FalApiError(
                f"queue result status {result_resp.status_code} (request {request_id}): {detail}"
            )
        payload = result_resp.json()
        if not isinstance(payload, dict):
            raise FalApiError(f"queue result not an object (request {request_id})")
        return payload

    def _http_with_retry(self, label: str, send: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = send()
                if _should_retry_status(resp.status_code) and attempt < MAX_ATTEMPTS:
                    resp.read()
                    time.sleep(_retry_delay_s(attempt))
                    continue
                return resp
            except httpx.RequestError as err:
                last_error = err
                if attempt < MAX_ATTEMPTS and _should_retry_error(err):
                    time.sleep(_retry_delay_s(attempt))
                    continue
                raise FalApiError(f"{label}: {err}") from err
        raise FalApiError(f"{label}: exhausted retries") from last_error

    def _require_key(self) -> str:
        key = (self.config.fal_key or "").strip()
        if not key:
            raise FalApiError("FAL_KEY not set")
        return key


def extract_file_url(result: dict[str, Any], key: str) -> str:
    node = result.get(key)
    if node is None:
        raise FalApiError(f"missing {key} in response")
    if isinstance(node, dict):
        url = node.get("url")
        if isinstance(url, str):
            return url
    if isinstance(node, str) and (node.startswith("http") or node.startswith("file:")):
        return node
    raise FalApiError(f"missing {key}.url")


def _extract_upload_url(payload: dict[str, Any]) -> str:
    url = payload.get("url")
    if isinstance(url, str):
        return url
    return extract_file_url(payload, "file")


def _should_retry_status(status_code: int) -> bool:
    return status_code in (408, 409, 429, 502, 503, 504)


def _should_retry_error(err: httpx.RequestError) -> bool:
    return isinstance(err, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError))


def _retry_delay_s(attempt: int) -> float:
    exp = BASE_RETRY_DELAY_MS * (2 ** max(0, attempt - 1))
    return min(exp, MAX_RETRY_DELAY_MS) / 1000.0
