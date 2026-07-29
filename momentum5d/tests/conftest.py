from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-api-key",
        base_url="https://api.jquants.com/v2",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        requests_per_minute=60000,
        connect_timeout_seconds=1,
        read_timeout_seconds=2,
        max_retries=2,
        backoff_base_seconds=0.01,
        abnormal_return_threshold=0.50,
        split_ratio_tolerance=0.25,
        universe_market_codes=("0111",),
    )
