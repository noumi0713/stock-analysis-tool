from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CheckpointStore:
    """原子的に更新されるページ単位の取得チェックポイント。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def unit_key(dataset: str, unit_id: str) -> str:
        return f"{dataset}|{unit_id}"

    def get(self, dataset: str, unit_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._read()
            value = state["units"].get(self.unit_key(dataset, unit_id))
            return deepcopy(value) if value is not None else None

    def begin(
        self,
        dataset: str,
        unit_id: str,
        *,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            key = self.unit_key(dataset, unit_id)
            unit = state["units"].get(key)
            if unit is None:
                unit = {
                    "dataset": dataset,
                    "unit_id": unit_id,
                    "endpoint": endpoint,
                    "params": params,
                    "status": "in_progress",
                    "next_pagination_key": None,
                    "next_page": 0,
                    "row_count": 0,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "error": None,
                }
                state["units"][key] = unit
                self._write(state)
            return deepcopy(unit)

    def page_saved(
        self,
        dataset: str,
        unit_id: str,
        *,
        next_pagination_key: str | None,
        next_page: int,
        added_rows: int,
    ) -> None:
        with self._lock:
            state = self._read()
            unit = state["units"][self.unit_key(dataset, unit_id)]
            unit["next_pagination_key"] = next_pagination_key
            unit["next_page"] = next_page
            unit["row_count"] = int(unit.get("row_count", 0)) + added_rows
            unit["status"] = "downloaded" if next_pagination_key is None else "in_progress"
            unit["updated_at"] = _now()
            unit["error"] = None
            self._write(state)

    def complete(self, dataset: str, unit_id: str, *, processed_rows: int) -> None:
        with self._lock:
            state = self._read()
            unit = state["units"][self.unit_key(dataset, unit_id)]
            unit["status"] = "complete"
            unit["processed_rows"] = processed_rows
            unit["completed_at"] = _now()
            unit["updated_at"] = _now()
            unit["error"] = None
            self._write(state)

    def fail(
        self,
        dataset: str,
        unit_id: str,
        error: str,
        *,
        restart_download: bool = False,
    ) -> None:
        with self._lock:
            state = self._read()
            unit = state["units"][self.unit_key(dataset, unit_id)]
            unit["status"] = "failed"
            unit["error"] = error[:2000]
            unit["updated_at"] = _now()
            if restart_download:
                unit["next_pagination_key"] = None
                unit["next_page"] = 0
                unit["row_count"] = 0
            self._write(state)

    def mark_dataset_unavailable(self, dataset: str, error: str) -> None:
        with self._lock:
            state = self._read()
            state["datasets"][dataset] = {
                "status": "unavailable",
                "error": error[:2000],
                "updated_at": _now(),
            }
            self._write(state)

    def skip_unavailable(self, dataset: str, unit_id: str, error: str) -> None:
        with self._lock:
            state = self._read()
            unit = state["units"][self.unit_key(dataset, unit_id)]
            unit["status"] = "skipped_unavailable"
            unit["error"] = error[:2000]
            unit["updated_at"] = _now()
            self._write(state)

    def mark_dataset_available(self, dataset: str) -> None:
        with self._lock:
            state = self._read()
            state["datasets"][dataset] = {
                "status": "available",
                "error": None,
                "updated_at": _now(),
            }
            self._write(state)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._read())

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "units": {}, "datasets": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"未対応のチェックポイント形式です: {self.path}")
        data.setdefault("units", {})
        data.setdefault("datasets", {})
        return data

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
