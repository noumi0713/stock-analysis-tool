from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling.backtest import BacktestResult


class BacktestStore:
    def __init__(self, processed_dir: Path, metadata_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.metadata_dir = metadata_dir

    def save(self, result: BacktestResult) -> dict[str, Path]:
        directory = self.processed_dir / "backtest"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "predictions": directory / "latest_predictions.parquet",
            "trades": directory / "latest_trades.parquet",
            "equity_curve": directory / "latest_equity_curve.parquet",
            "summary": self.metadata_dir / "backtest_latest.json",
        }
        self._atomic_parquet(result.predictions, paths["predictions"])
        self._atomic_parquet(result.trades, paths["trades"])
        self._atomic_parquet(result.equity_curve, paths["equity_curve"])
        self._atomic_json(result.summary, paths["summary"])
        return paths

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json(value: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
