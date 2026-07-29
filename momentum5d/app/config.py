from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value <= 0:
        raise ValueError(f"{name} は0より大きい値である必要があります")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value < 0:
        raise ValueError(f"{name} は0以上の値である必要があります")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < 0:
        raise ValueError(f"{name} は0以上の値である必要があります")
    return value


def _csv_tuple(name: str, default: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())
    if not values:
        raise ValueError(f"{name} には1つ以上の値を設定してください")
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str | None
    base_url: str
    data_dir: Path
    log_dir: Path
    requests_per_minute: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    abnormal_return_threshold: float
    split_ratio_tolerance: float
    universe_market_codes: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def duckdb_path(self) -> Path:
        return self.metadata_dir / "market.duckdb"

    @property
    def checkpoint_path(self) -> Path:
        return self.metadata_dir / "checkpoints.json"

    @classmethod
    def load(cls, env_file: str | Path = ".env") -> Settings:
        load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            api_key=os.getenv("JQUANTS_API_KEY") or None,
            base_url=os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com/v2").rstrip("/"),
            data_dir=Path(os.getenv("DATA_DIR", "data")).expanduser().resolve(),
            log_dir=Path(os.getenv("LOG_DIR", "logs")).expanduser().resolve(),
            requests_per_minute=_positive_float("JQUANTS_REQUESTS_PER_MINUTE", 5.0),
            connect_timeout_seconds=_positive_float("JQUANTS_CONNECT_TIMEOUT_SECONDS", 10.0),
            read_timeout_seconds=_positive_float("JQUANTS_READ_TIMEOUT_SECONDS", 30.0),
            max_retries=_nonnegative_int("JQUANTS_MAX_RETRIES", 5),
            backoff_base_seconds=_nonnegative_float("JQUANTS_BACKOFF_BASE_SECONDS", 1.0),
            abnormal_return_threshold=_positive_float("ABNORMAL_RETURN_THRESHOLD", 0.50),
            split_ratio_tolerance=_positive_float("SPLIT_RATIO_TOLERANCE", 0.25),
            universe_market_codes=_csv_tuple("UNIVERSE_MARKET_CODES", "0111"),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "JQUANTS_API_KEY が未設定です。.env.example を .env にコピーして設定してください"
            )
        return self.api_key

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.metadata_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
