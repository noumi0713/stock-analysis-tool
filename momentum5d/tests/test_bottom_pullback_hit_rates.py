import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/bottom_pullback_hit_rates.py"
spec = importlib.util.spec_from_file_location("bp", SCRIPT)
bp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bp
spec.loader.exec_module(bp)


def bar(i, c=100., h=None, l=None, o=None, half=1, valid=True, v=1000):
    return {"Ticker": "TEST", "Date": str(pd.bdate_range("2021-01-01", periods=i + 1)[-1].date()),
            "session": i, "half": half, "valid": valid, "o": c if o is None else o,
            "h": c + 1 if h is None else h, "l": c - 1 if l is None else l, "c": c, "v": v}


def signal():
    return {"signal_id": "x", "Ticker": "TEST", "kind": "bottom", "signal_date": "2020-12-31",
            "signal_session": -1, "signal_half": 1, "period": "formation_used"}


def detection_path():
    # Sustained decline -> lower low -> genuine five-bar high breakout.
    prices = list(np.linspace(180, 100, 95)) + [98, 97, 96, 95, 97, 98, 100, 103]
    return [bar(i, c=float(c)) for i, c in enumerate(prices)]


def collect(rows):
    d, signals = bp.Detector("TEST"), []
    for b in rows:
        s = d.step(b)
        if s:
            signals.append(s)
    return signals


def test_bottom_confirmation_is_not_backdated():
    found = collect(detection_path())
    assert found and found[0]["kind"] == "bottom"
    assert found[0]["candidate_date"] < found[0]["signal_date"]


def test_future_change_and_prefix_do_not_change_signals():
    path = detection_path()
    original = collect(path)
    for cut in [90, 100, 105, len(path)]:
        prefix = collect(path[:cut])
        assert prefix == [s for s in original if s["signal_session"] < cut]
    changed = path + [bar(len(path)+i, c=1000.) for i in range(12)]
    assert [s for s in collect(changed) if s["signal_session"] < len(path)] == original


def test_pullback_requires_uptrend():
    prices = list(np.linspace(100, 180, 100)) + [177, 174, 172, 173, 174, 176, 179, 183]
    found = collect([bar(i, c=float(c)) for i, c in enumerate(prices)])
    assert found and found[0]["kind"] == "pullback"


def test_entry_bar_counts_and_tenth_bar_counts_eleventh_does_not():
    o = bp.Outcome(signal())
    for i in range(10):
        o.step(bar(i, h=110.0001 if i == 9 else 101., o=100.))
    assert o.row["status"] == "complete"
    assert o.row["hit_5"] and o.row["hit_10"] and o.row["hit_10_session"] == 10
    o2 = bp.Outcome(signal())
    for i in range(11):
        o2.step(bar(i, h=120. if i == 10 else 101.))
    assert not o2.row["hit_5"]
    o3 = bp.Outcome(signal())
    o3.step(bar(0, h=120., o=100.))
    assert o3.row["hit_10"] and o3.row["hit_10_session"] == 1


def test_price_denominator_is_open_not_confirmation_close():
    o = bp.Outcome(signal())
    for i in range(10):
        o.step(bar(i, c=120., o=120., h=125., l=119.))
    assert o.row["entry_price"] == 120.
    assert not o.row["hit_5"]


def test_missing_and_censoring_not_converted_to_losses():
    o = bp.Outcome(signal())
    o.step(bar(0))
    o.step(bar(1, valid=False))
    assert o.row["status"] == "unknown"
    o2 = bp.Outcome(signal())
    o2.step(bar(0))
    o2.close()
    assert o2.row["status"] == "right_censored"


def test_zero_volume_is_not_a_fill():
    o = bp.Outcome(signal())
    o.step(bar(0, v=0))
    assert o.row["status"] == "unfilled"
    assert o.row["entry_price"] is None


def test_absent_session_is_not_compressed():
    o = bp.Outcome(signal())
    o.step(bar(0))
    o.step(bar(2))
    assert o.row["reason"] == "missing_source_session_in_window"


def frame(path):
    return pd.DataFrame([{"Ticker": b["Ticker"], "Date": b["Date"], "Open": b["o"],
                          "High": b["h"], "Low": b["l"], "Close": b["c"], "Adj Close": b["c"],
                          "Volume": b["v"], "session": b["session"], "half": b["half"]} for b in path])


def test_chunk_boundary_keeps_state_and_targets_and_reproducibility(tmp_path):
    rows = detection_path()
    rows += [bar(len(rows)+i, c=110.) for i in range(15)]
    cut = 102  # confirmation in half1, ten-session outcome crosses into half2
    for i, r in enumerate(rows):
        r["half"] = 1 if i < cut else 2
    full, a, b = (tmp_path / s for s in ["full.parquet", "a.parquet", "b.parquet"])
    frame(rows).to_parquet(full, index=False)
    frame(rows[:cut]).to_parquet(a, index=False)
    frame(rows[cut:]).to_parquet(b, index=False)
    bp.run_stream([a, b], tmp_path / "split")
    bp.run_stream([full], tmp_path / "whole")
    bp.run_stream([a, b], tmp_path / "repeat")
    assert bp.manifest(tmp_path / "split") == bp.manifest(tmp_path / "whole")
    assert bp.manifest(tmp_path / "split") == bp.manifest(tmp_path / "repeat")


def test_weighted_pooling_not_mean_of_rates():
    rows = []
    for i in range(10):
        o = bp.Outcome(dict(signal(), signal_id=str(i), signal_half=1 if i == 0 else 2))
        for j in range(10):
            o.step(bar(j, h=120 if i == 0 else 101))
        rows.append(o.row)
    s = bp.summarise(rows)
    allrow = s[(s.scope == "all") & (s.kind == "all")].iloc[0]
    assert allrow.hit_rate_5 == .1


def test_adjustment_is_same_day_factor(tmp_path):
    f = frame([bar(0)])
    f["Adj Close"] = 50.
    p = tmp_path / "adjust.parquet"
    f.to_parquet(p, index=False)
    b = list(bp.bars(p))[0]
    assert b["o"] == 50. and b["h"] == 50.5 and b["l"] == 49.5


def test_atr_seed_and_recursive_update():
    d = bp.Detector("TEST")
    for i in range(15):
        d.step(bar(i, c=100., h=101., l=99.))
    assert d.atr == 2.
    d.step(bar(15, c=100., h=105., l=95.))
    assert d.atr == (26+10)/14


def test_exact_percent_barrier_rounding():
    o = bp.Outcome(signal())
    o.step(bar(0, o=100., h=110.))
    assert o.row["hit_10"]
    p = bp.Outcome(signal())
    p.step(bar(0, o=100., h=109.999))
    assert not p.row["hit_10"]


def test_split_sql_and_source_row_conservation(tmp_path):
    source = tmp_path / "certified_input/features/equity_daily_features"
    source.mkdir(parents=True)
    raw = frame(detection_path()).drop(columns=["session", "half"])
    raw.to_parquet(source / "test_only.parquet", index=False)
    result = bp.split(tmp_path)
    assert sum(h["rows"] for h in result["halves"]) == len(raw)
    first = pd.read_parquet(tmp_path / "halves/half_1.parquet")
    second = pd.read_parquet(tmp_path / "halves/half_2.parquet")
    assert first.Date.max() < second.Date.min()
    assert second.session.min() == first.session.max() + 1
