"""Adapters for licensed IFIS and QUICK consensus feeds.

The public documentation confirms that IFIS Consensus Data is delivered via
FTP/SFTP, email or Snowflake, while QUICK target/rating data is delivered via
QUICK APIs. Their contracted endpoint/schema details are not public and vary by
subscription, so this module intentionally consumes *normalized provider drops*
created by the licensed transport layer instead of guessing private endpoints.

Expected normalized files:

IFIS (one row per stock):
  stock_code,target_low,target_mean,target_high,analyst_count,base_date,source_url

QUICK (one or more rows per stock/broker):
  stock_code,target_price,broker_name,updated_at,source_url

QUICK rows are aggregated to low/mean/high across the currently supplied broker
observations. Missing QUICK coverage does not block a composite as long as two
other providers are available. No missing value is converted to zero.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _code(value) -> str:
    text = str(value).strip().upper().removesuffix(".T")
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _validate_triplet(low, mean, high):
    values = [_num(low), _num(mean), _num(high)]
    if any(v is None or v <= 0 for v in values):
        return None
    if not values[0] <= values[1] <= values[2]:
        return None
    return tuple(values)


def load_ifis_normalized(path: str | Path | None) -> pd.DataFrame:
    """Load a normalized licensed IFIS consensus file.

    IFIS officially provides min/average/max/estimate-count data for Target Price.
    The transport-specific extractor should map those fields into this canonical
    schema before this loader is called.
    """
    columns = [
        "stock_code", "ifis_target_low", "ifis_target_mean", "ifis_target_high",
        "ifis_analyst_count", "ifis_base_date", "ifis_source_url",
    ]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"stock_code": str}).fillna("")
    required = {"stock_code", "target_low", "target_mean", "target_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IFIS normalized feed missing columns: {sorted(missing)}")
    out = []
    for row in frame.to_dict("records"):
        triplet = _validate_triplet(row.get("target_low"), row.get("target_mean"), row.get("target_high"))
        if not triplet:
            continue
        out.append({
            "stock_code": _code(row["stock_code"]),
            "ifis_target_low": triplet[0],
            "ifis_target_mean": triplet[1],
            "ifis_target_high": triplet[2],
            "ifis_analyst_count": _num(row.get("analyst_count")),
            "ifis_base_date": str(row.get("base_date") or ""),
            "ifis_source_url": str(row.get("source_url") or ""),
        })
    result = pd.DataFrame(out, columns=columns)
    if result.empty:
        return result
    return result.drop_duplicates("stock_code", keep="last")


def load_quick_normalized(path: str | Path | None) -> pd.DataFrame:
    """Aggregate normalized QUICK broker target-price rows by stock.

    QUICK's public product documentation exposes current target prices per broker.
    We calculate min/mean/max from the current broker rows supplied by the licensed
    API extractor. This is deliberately separate from QUICK earnings consensus.
    """
    columns = [
        "stock_code", "quick_target_low", "quick_target_mean", "quick_target_high",
        "quick_analyst_count", "quick_base_date", "quick_source_url",
    ]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"stock_code": str}).fillna("")
    required = {"stock_code", "target_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"QUICK normalized feed missing columns: {sorted(missing)}")
    frame["stock_code"] = frame["stock_code"].map(_code)
    frame["target_price"] = pd.to_numeric(frame["target_price"], errors="coerce")
    frame = frame[frame["target_price"].notna() & (frame["target_price"] > 0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for code, group in frame.groupby("stock_code", sort=True):
        values = group["target_price"].astype(float)
        dates = [str(x) for x in group.get("updated_at", pd.Series(dtype=str)).tolist() if str(x)]
        urls = [str(x) for x in group.get("source_url", pd.Series(dtype=str)).tolist() if str(x)]
        rows.append({
            "stock_code": code,
            "quick_target_low": float(values.min()),
            "quick_target_mean": float(values.mean()),
            "quick_target_high": float(values.max()),
            "quick_analyst_count": int(values.count()),
            "quick_base_date": max(dates) if dates else "",
            "quick_source_url": urls[0] if urls else "",
        })
    return pd.DataFrame(rows, columns=columns)


def composite_targets(row: dict, minimum_sources: int = 2) -> dict:
    """Average low/mean/high across available Yahoo, IFIS and QUICK sources."""
    used = []
    lows, means, highs = [], [], []
    for prefix, label in [("yahoo", "Yahoo Finance"), ("ifis", "IFIS"), ("quick", "QUICK")]:
        triplet = _validate_triplet(
            row.get(f"{prefix}_target_low"),
            row.get(f"{prefix}_target_mean"),
            row.get(f"{prefix}_target_high"),
        )
        if not triplet:
            continue
        used.append(label)
        lows.append(triplet[0])
        means.append(triplet[1])
        highs.append(triplet[2])
    if len(used) < minimum_sources:
        return {
            "target_low": None,
            "target_mean": None,
            "target_high": None,
            "composite_source_count": len(used),
            "composite_sources": ";".join(used),
            "composite_quality": "insufficient_sources",
        }
    return {
        "target_low": sum(lows) / len(lows),
        "target_mean": sum(means) / len(means),
        "target_high": sum(highs) / len(highs),
        "composite_source_count": len(used),
        "composite_sources": ";".join(used),
        "composite_quality": "three_source" if len(used) == 3 else "two_source",
    }
