import runpy
from pathlib import Path

import duckdb


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_step3_events.py"


def test_event_labels_use_adjusted_close_and_trading_rows() -> None:
    module = runpy.run_path(str(SCRIPT))
    connection = duckdb.connect()
    connection.execute(
        """CREATE TABLE step1 AS SELECT * FROM (VALUES
        (DATE '2026-01-01','A',100.0),(DATE '2026-01-02','A',105.0),
        (DATE '2026-01-05','A',110.0),(DATE '2026-01-06','A',131.0)
        ) t(Date,Ticker,"Adj Close")"""
    )
    connection.execute(
        """CREATE TABLE step2 AS SELECT * FROM (VALUES
        (DATE '2026-01-01','A',1.0,1.0,1.0),(DATE '2026-01-02','A',NULL,NULL,NULL),
        (DATE '2026-01-05','A',NULL,NULL,NULL),(DATE '2026-01-06','A',NULL,NULL,NULL)
        ) t(Date,Ticker,mfe_20d,mfe_40d,mfe_60d)"""
    )
    # The production horizons need 20/40/60 rows, so this compact fixture
    # verifies SQL shape and that insufficient certified horizons stay NULL.
    row = connection.execute(module["event_label_sql"]() + " WHERE Date=DATE '2026-01-02'").fetchone()
    columns = [column[0] for column in connection.description]
    result = dict(zip(columns, row, strict=True))
    assert result["max_forward_close_return_20d"] is None
    assert result["event_40d_ge_30pct"] is None
    connection.close()


def test_overlapping_candidate_intervals_share_event_id() -> None:
    module = runpy.run_path(str(SCRIPT))
    connection = duckdb.connect()
    flag_values = ",".join("TRUE" for _ in range(9))
    connection.execute(
        f"""CREATE TABLE labels AS SELECT * FROM (VALUES
        (DATE '2026-01-01','A',100.0,0.4,0.5,0.6,DATE '2026-01-10',DATE '2026-01-20',DATE '2026-02-01',{flag_values}),
        (DATE '2026-01-15','A',110.0,0.4,0.5,0.6,DATE '2026-01-25',DATE '2026-02-05',DATE '2026-02-10',{flag_values}),
        (DATE '2026-03-01','A',120.0,0.4,0.5,0.6,DATE '2026-03-10',DATE '2026-03-20',DATE '2026-04-01',{flag_values})
        ) t(Date,Ticker,entry_adjusted_close,max_forward_close_return_20d,
        max_forward_close_return_40d,max_forward_close_return_60d,
        peak_date_20d,peak_date_40d,peak_date_60d,
        event_20d_ge_20pct,event_20d_ge_30pct,event_20d_ge_50pct,
        event_40d_ge_20pct,event_40d_ge_30pct,event_40d_ge_50pct,
        event_60d_ge_20pct,event_60d_ge_30pct,event_60d_ge_50pct)"""
    )
    rows = connection.execute(module["candidate_sql"]() + " ORDER BY Date").fetchall()
    assert rows[0][-1] == rows[1][-1]
    assert rows[1][-1] != rows[2][-1]
    connection.close()
