from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from app.api.client import APIError, JQuantsClient
from app.config import Settings


def response(status: int, payload: object, *, headers: dict[str, str] | None = None) -> Mock:
    result = Mock(spec=requests.Response)
    result.status_code = status
    result.ok = 200 <= status < 400
    result.headers = headers or {}
    result.text = str(payload)
    result.json.return_value = payload
    return result


def test_uses_v2_api_key_and_timeout(settings: Settings) -> None:
    session = Mock(spec=requests.Session)
    session.get.return_value = response(200, {"data": []})
    client = JQuantsClient(settings, session=session, sleeper=lambda _: None)

    assert client.get_json("/equities/master") == {"data": []}

    _, kwargs = session.get.call_args
    assert kwargs["headers"]["x-api-key"] == "test-api-key"
    assert kwargs["timeout"] == (1, 2)


def test_pagination_preserves_query_and_uses_cursor(settings: Settings) -> None:
    session = Mock(spec=requests.Session)
    session.get.side_effect = [
        response(200, {"data": [{"Code": "13010"}], "pagination_key": "next"}),
        response(200, {"data": [{"Code": "13020"}]}),
    ]
    client = JQuantsClient(settings, session=session, sleeper=lambda _: None)

    pages = list(client.iter_pages("/equities/master", {"date": "2026-01-05"}))

    assert [page.page_number for page in pages] == [0, 1]
    assert pages[1].records == [{"Code": "13020"}]
    first_query = session.get.call_args_list[0].kwargs["params"]
    second_query = session.get.call_args_list[1].kwargs["params"]
    assert first_query == {"date": "2026-01-05"}
    assert second_query == {"date": "2026-01-05", "pagination_key": "next"}


def test_retries_429_using_retry_after(settings: Settings) -> None:
    session = Mock(spec=requests.Session)
    session.get.side_effect = [
        response(429, {"message": "rate limited"}, headers={"Retry-After": "2"}),
        response(200, {"data": []}),
    ]
    sleeps: list[float] = []
    client = JQuantsClient(settings, session=session, sleeper=sleeps.append)

    client.get_json("/equities/master")

    assert 2.0 in sleeps
    assert session.get.call_count == 2


def test_429_without_retry_after_waits_for_rate_window(settings: Settings) -> None:
    session = Mock(spec=requests.Session)
    session.get.side_effect = [
        response(429, {"message": "rate limited"}),
        response(200, {"data": []}),
    ]
    sleeps: list[float] = []
    client = JQuantsClient(settings, session=session, sleeper=sleeps.append)

    client.get_json("/equities/master")

    assert any(delay >= 60.0 for delay in sleeps)


def test_retries_timeout_then_raises(settings: Settings) -> None:
    session = Mock(spec=requests.Session)
    session.get.side_effect = requests.Timeout("slow")
    client = JQuantsClient(settings, session=session, sleeper=lambda _: None)

    with pytest.raises(APIError, match="3回試行"):
        client.get_json("/equities/master")


def test_missing_api_key_is_rejected(settings: Settings) -> None:
    without_key = Settings(
        api_key=None,
        base_url=settings.base_url,
        data_dir=settings.data_dir,
        log_dir=settings.log_dir,
        requests_per_minute=5,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        backoff_base_seconds=0,
        abnormal_return_threshold=0.5,
        split_ratio_tolerance=0.25,
        universe_market_codes=settings.universe_market_codes,
    )
    with pytest.raises(ValueError, match="JQUANTS_API_KEY"):
        JQuantsClient(without_key)
