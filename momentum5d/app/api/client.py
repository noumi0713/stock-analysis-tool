from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from app.config import Settings

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class APIError(RuntimeError):
    """J-Quants APIが回復不能なレスポンスを返した場合の例外。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ApiPage:
    records: list[dict[str, Any]]
    next_key: str | None
    page_number: int


class RateLimiter:
    """プロセス内でリクエスト間隔を保証する単純なレートリミッター。"""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        # 公称上限ちょうどでは、サーバ側のスライディング窓と時刻差で
        # 429になり得るため5%の安全余裕を持たせる。
        self._interval = (60.0 / requests_per_minute) * 1.05
        self._clock = clock
        self._sleeper = sleeper
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_request_at is not None:
                remaining = self._interval - (now - self._last_request_at)
                if remaining > 0:
                    self._sleeper(remaining)
                    now = self._clock()
            self._last_request_at = now


class JQuantsClient:
    """J-Quants API V2の薄いクライアント。

    V2公式仕様に従い ``x-api-key`` ヘッダーと ``pagination_key`` を使用する。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings
        self._api_key = settings.require_api_key()
        self._session = session or requests.Session()
        self._sleeper = sleeper
        self._random = random_fn
        self._rate_limiter = RateLimiter(settings.requests_per_minute, clock=clock, sleeper=sleeper)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "Accept": "application/json",
            "User-Agent": "japan-stock-signal-data/0.1.0",
        }

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> JQuantsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.settings.base_url}/{path.lstrip('/')}"
        query = {key: value for key, value in (params or {}).items() if value != ""}
        attempts = self.settings.max_retries + 1

        for attempt in range(attempts):
            self._rate_limiter.wait()
            try:
                response = self._session.get(
                    url,
                    params=query,
                    headers=self.headers,
                    timeout=(
                        self.settings.connect_timeout_seconds,
                        self.settings.read_timeout_seconds,
                    ),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt + 1 == attempts:
                    raise APIError(
                        f"J-Quants APIへの接続に失敗しました（{attempts}回試行）: {exc}"
                    ) from exc
                delay = self._backoff_seconds(attempt, None)
                LOGGER.warning(
                    "API connection error; retrying attempt=%s delay=%.2fs error=%s",
                    attempt + 1,
                    delay,
                    exc,
                )
                self._sleeper(delay)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                delay = self._backoff_seconds(attempt, response.headers.get("Retry-After"))
                if response.status_code == 429 and not response.headers.get("Retry-After"):
                    delay = max(delay, 60.0 * (attempt + 1))
                LOGGER.warning(
                    "Retryable API response status=%s attempt=%s delay=%.2fs",
                    response.status_code,
                    attempt + 1,
                    delay,
                )
                self._sleeper(delay)
                continue

            if not response.ok:
                raise APIError(self._error_message(response), status_code=response.status_code)

            try:
                payload = response.json()
            except ValueError as exc:
                raise APIError("J-Quants APIからJSON以外のレスポンスを受信しました") from exc
            if not isinstance(payload, dict):
                raise APIError("J-Quants APIレスポンスのルートがオブジェクトではありません")
            return payload

        raise AssertionError("unreachable")

    def iter_pages(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        pagination_key: str | None = None,
        start_page: int = 0,
    ) -> Iterator[ApiPage]:
        query = dict(params or {})
        cursor = pagination_key
        seen: set[str] = set()
        page_number = start_page

        while True:
            if cursor:
                if cursor in seen:
                    raise APIError("pagination_key が循環したため取得を中断しました")
                seen.add(cursor)
                query["pagination_key"] = cursor
            else:
                query.pop("pagination_key", None)

            payload = self.get_json(path, query)
            records = payload.get("data", [])
            if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
                raise APIError("J-Quants APIレスポンスの data がオブジェクト配列ではありません")

            raw_next_key = payload.get("pagination_key")
            next_key = str(raw_next_key) if raw_next_key else None
            yield ApiPage(
                records=records,
                next_key=next_key,
                page_number=page_number,
            )
            if next_key is None:
                return
            cursor = next_key
            page_number += 1

    def _backoff_seconds(self, attempt: int, retry_after: str | None) -> float:
        parsed_retry_after = self._parse_retry_after(retry_after)
        if parsed_retry_after is not None:
            return parsed_retry_after
        exponential = self.settings.backoff_base_seconds * (2**attempt)
        jitter = self.settings.backoff_base_seconds * self._random()
        return exponential + jitter

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return max(
                    0.0,
                    (retry_at - datetime.now(UTC)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        detail: Any
        try:
            body = response.json()
            detail = body.get("message", body) if isinstance(body, dict) else body
        except ValueError:
            detail = response.text[:500]
        return f"J-Quants API error: HTTP {response.status_code}: {detail}"
