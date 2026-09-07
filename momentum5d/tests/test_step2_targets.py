import runpy
from pathlib import Path

import duckdb
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_step2_targets.py"


def test_target_sql_uses_complete_future_windows_and_adjusted_prices() -> None:
    module = runpy.run_path(str(SCRIPT))
    connection = duckdb.connect()
    connection.execute(
        """CREATE TABLE step1 AS SELECT * FROM (VALUES
        (DATE '2026-01-01','A',100.0,110.0,90.0,100.0,1000.0),
        (DATE '2026-01-02','A',102.0,112.0,92.0,102.0,1000.0),
        (DATE '2026-01-03','A',104.0,114.0,94.0,104.0,1000.0),
        (DATE '2026-01-04','A',106.0,116.0,96.0,106.0,1000.0),
        (DATE '2026-01-05','A',108.0,118.0,98.0,108.0,1000.0),
        (DATE '2026-01-06','A',55.0,60.0,50.0,110.0,1000.0)
        ) t(Date,Ticker,Close,High,Low,"Adj Close",Volume)"""
    )
    row = connection.execute(module["target_sql"]() + " WHERE Date=DATE '2026-01-01'").fetchone()
    columns = [item[0] for item in connection.description]
    result = dict(zip(columns, row, strict=True))
    assert result["forward_return_5d"] == pytest.approx(0.10)
    assert result["mfe_5d"] == pytest.approx(0.20)
    assert result["mae_5d"] == pytest.approx(-0.08)
    assert result["forward_return_10d"] is None
    assert result["mfe_10d"] is None
    assert result["mae_10d"] is None
    connection.close()
