from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.yahoo.theme_context import calculate_theme_context
from scripts.export_close_signals import (
    _condition_results,
    _load_theme_memberships,
    _pullback_condition_results,
    calculate_indicators,
)

HOLDOUT_START = "2025-02-10"
TRANSACTION_COST = 0.002


def load_technical_shards(dataset_dir: Path) -> pd.DataFrame:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = manifest["dates"]
    rows: list[tuple[Any, ...]] = []
    for shard in manifest["shards"]:
        payload = json.loads((dataset_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            rows.extend(
                (
                    ticker,
                    dates[bar[0]],
                    bar[1],
                    bar[2],
                    bar[3],
                    bar[4],
                    bar[4],
                    bar[5],
                )
                for bar in bars
            )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ],
    )


def _membership_frame(
    memberships: dict[str, list[dict[str, str]]],
    field: str = "theme",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": f"{code}.T", "theme": item[field]}
            for code, items in memberships.items()
            for item in items
            if item.get(field)
        ]
    ).drop_duplicates()


def _best_stock_context(
    theme_context: pd.DataFrame,
    memberships: dict[str, list[dict[str, str]]],
    *,
    field: str = "theme",
    prefix: str = "theme",
) -> pd.DataFrame:
    joined = _membership_frame(memberships, field).merge(
        theme_context,
        on="theme",
        how="inner",
        validate="many_to_many",
    )
    best = (
        joined.sort_values(
            ["ticker", "date", "theme_score", "theme"],
            ascending=[True, True, False, True],
        )
        .groupby(["ticker", "date"], sort=False, as_index=False)
        .first()
    )
    return best.rename(
        columns={
            "theme": f"{prefix}_name",
            "theme_score": f"{prefix}_score",
            "theme_flow_confirmed": f"{prefix}_flow_confirmed",
            "theme_return_5d": f"{prefix}_return_5d",
            "theme_breadth_5d": f"{prefix}_breadth_5d",
            "theme_turnover_ratio_1_20": f"{prefix}_turnover_ratio_1_20",
            "theme_member_count": f"{prefix}_member_count",
        }
    )


def _add_forward_outcomes(indicators: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = indicators.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", sort=False)
    frame["entry"] = grouped["_open"].shift(-1)
    shifted_high = grouped["_high"].shift(-1)
    shifted_low = grouped["_low"].shift(-1)
    frame["future_max"] = shifted_high.groupby(frame["ticker"], sort=False).transform(
        lambda values: values.rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    )
    frame["future_min"] = shifted_low.groupby(frame["ticker"], sort=False).transform(
        lambda values: values.rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    )
    frame["future_close"] = grouped["_close"].shift(-horizon)
    frame["target_5pct_hit"] = frame["future_max"].div(frame["entry"]).sub(1).ge(0.05)
    frame["net_return"] = frame["future_close"].div(frame["entry"]).sub(1).sub(TRANSACTION_COST)
    frame["max_adverse_excursion"] = frame["future_min"].div(frame["entry"]).sub(1)
    return frame


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "signals": 0,
            "active_days": 0,
            "target_5pct_hit_rate": None,
            "mean_net_return": None,
            "mean_max_adverse_excursion": None,
            "context_coverage": None,
        }
    return {
        "signals": int(len(frame)),
        "active_days": int(frame["date"].nunique()),
        "target_5pct_hit_rate": float(frame["target_5pct_hit"].mean()),
        "mean_net_return": float(frame["net_return"].mean()),
        "mean_max_adverse_excursion": float(frame["max_adverse_excursion"].mean()),
        "context_coverage": (
            float(frame["_context_covered"].mean()) if "_context_covered" in frame else None
        ),
    }


def _select_comparison(frame: pd.DataFrame, score_column: str) -> dict[str, pd.DataFrame]:
    modes = {
        "1_cluster_only": "cluster_score",
        "2_theme_only": "theme_score",
        "3_hierarchical": "hierarchical_score",
    }
    result: dict[str, pd.DataFrame] = {}
    baseline = frame.assign(_context_covered=False)
    result["0_baseline"] = (
        baseline.sort_values(["date", score_column, "ticker"], ascending=[True, False, True])
        .groupby("date", sort=False)
        .head(4)
    )
    for key, context_score in modes.items():
        ranked = frame.assign(
            _context_covered=frame[context_score].notna(),
            _context_score=frame[context_score].fillna(-1.0),
        )
        result[key] = (
            ranked.sort_values(
                ["date", "_context_score", score_column, "ticker"],
                ascending=[True, False, False, True],
            )
            .groupby("date", sort=False)
            .head(4)
        )
    return result


def _select_variants(frame: pd.DataFrame, score_column: str) -> dict[str, pd.DataFrame]:
    ranked = frame.assign(
        _theme_confirmed=frame["theme_flow_confirmed"].eq(True).astype("int8"),
        _theme_score=frame["theme_score"].fillna(-1.0),
    )
    baseline = (
        ranked.sort_values(
            ["date", score_column, "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(4)
    )
    theme_rank = (
        ranked.sort_values(
            ["date", "_theme_confirmed", "_theme_score", score_column, "ticker"],
            ascending=[True, False, False, False, True],
        )
        .groupby("date", sort=False)
        .head(4)
    )
    theme_filter = (
        ranked.loc[ranked["_theme_confirmed"].eq(1)]
        .sort_values(["date", score_column, "ticker"], ascending=[True, False, True])
        .groupby("date", sort=False)
        .head(4)
    )
    return {
        "baseline": baseline,
        "theme_rank": theme_rank,
        "theme_filter": theme_filter,
    }


def _evaluate_family(
    indicators: pd.DataFrame,
    theme_stock_context: pd.DataFrame,
    cluster_stock_context: pd.DataFrame,
    *,
    family: str,
) -> dict[str, Any]:
    if family == "reversal":
        horizon = 10
        candidates = indicators.loc[_condition_results(indicators).all(axis=1)].copy()
        candidates["_base_score"] = candidates["volume_ratio_1_20"]
    elif family == "pullback":
        horizon = 15
        candidates = indicators.loc[_pullback_condition_results(indicators).all(axis=1)].copy()
        candidates["_base_score"] = (
            candidates["close_position"] * 2
            + candidates["volume_ratio_1_20"]
            - candidates["distance_from_ma25"] * 25
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    outcomes = _add_forward_outcomes(indicators, horizon)[
        [
            "ticker",
            "date",
            "entry",
            "future_max",
            "future_min",
            "future_close",
            "target_5pct_hit",
            "net_return",
            "max_adverse_excursion",
        ]
    ]
    theme_columns = [
        "ticker",
        "date",
        "theme_name",
        "theme_score",
        "theme_flow_confirmed",
        "theme_return_5d",
        "theme_breadth_5d",
        "theme_turnover_ratio_1_20",
        "theme_member_count",
    ]
    cluster_columns = [
        "ticker",
        "date",
        "cluster_name",
        "cluster_score",
        "cluster_flow_confirmed",
        "cluster_return_5d",
        "cluster_breadth_5d",
        "cluster_turnover_ratio_1_20",
        "cluster_member_count",
    ]
    evaluated = (
        candidates.merge(theme_stock_context[theme_columns], on=["ticker", "date"], how="left")
        .merge(cluster_stock_context[cluster_columns], on=["ticker", "date"], how="left")
        .merge(outcomes, on=["ticker", "date"], how="left")
    )
    theme_weight = evaluated["theme_member_count"].div(evaluated["theme_member_count"].add(20.0))
    both = evaluated["theme_score"].notna() & evaluated["cluster_score"].notna()
    evaluated["hierarchical_score"] = evaluated["theme_score"].combine_first(
        evaluated["cluster_score"]
    )
    evaluated.loc[both, "hierarchical_score"] = (
        theme_weight.loc[both] * evaluated.loc[both, "theme_score"]
        + (1.0 - theme_weight.loc[both]) * evaluated.loc[both, "cluster_score"]
    )
    if family == "pullback":
        evaluated = evaluated.loc[
            evaluated["entry"].div(evaluated["_close"]).sub(1).between(-0.04, 0.03)
        ]
    evaluated = evaluated.dropna(subset=["entry", "future_close", "future_max", "future_min"])

    holdout = evaluated.loc[pd.to_datetime(evaluated["date"]).ge(pd.Timestamp(HOLDOUT_START))]
    result: dict[str, Any] = {
        "horizon_sessions": horizon,
        "all_candidates": _summary(evaluated),
        "holdout_candidates": _summary(holdout),
        "all_variants": {
            key: _summary(value)
            for key, value in _select_variants(evaluated, "_base_score").items()
        },
        "holdout_variants": {
            key: _summary(value) for key, value in _select_variants(holdout, "_base_score").items()
        },
        "comparison_all": {
            key: _summary(value)
            for key, value in _select_comparison(evaluated, "_base_score").items()
        },
        "comparison_holdout": {
            key: _summary(value)
            for key, value in _select_comparison(holdout, "_base_score").items()
        },
    }
    if family == "pullback":
        comparison = result["comparison_holdout"]
        result["decision"] = max(
            (key for key in comparison if key != "0_baseline"),
            key=lambda key: (
                comparison[key]["target_5pct_hit_rate"],
                comparison[key]["mean_net_return"],
            ),
        )
    else:
        result["decision"] = "do_not_use_theme_for_selection"
    return result


def run_backtest(prices: pd.DataFrame, theme_path: Path) -> dict[str, Any]:
    indicators = calculate_indicators(prices)
    memberships = _load_theme_memberships(theme_path)
    catalog_themes = {
        item["theme"] for items in memberships.values() for item in items if item.get("theme")
    }
    catalog_clusters = {
        item["cluster"] for items in memberships.values() for item in items if item.get("cluster")
    }
    theme_context = calculate_theme_context(indicators, memberships)
    cluster_memberships = {
        code: [{"theme": item["cluster"]} for item in items if item.get("cluster")]
        for code, items in memberships.items()
    }
    cluster_context = calculate_theme_context(indicators, cluster_memberships)
    theme_stock_context = _best_stock_context(
        theme_context, memberships, field="theme", prefix="theme"
    )
    cluster_stock_context = _best_stock_context(
        cluster_context, memberships, field="cluster", prefix="cluster"
    )
    return {
        "method": "theme_context_incremental_holdout_v1",
        "price_start": str(indicators["date"].min()),
        "price_end": str(indicators["date"].max()),
        "holdout_start": HOLDOUT_START,
        "theme_count": len(catalog_themes),
        "active_theme_count": int(theme_context["theme"].nunique()),
        "cluster_count": len(catalog_clusters),
        "active_cluster_count": int(cluster_context["theme"].nunique()),
        "membership_history": "current_catalog_only",
        "membership_caveat": (
            "株探テーマの過去時点の所属履歴がないため、現在の所属を過去へ適用した研究値"
        ),
        "theme_score": {
            "return_5d_rank_weight": 0.40,
            "breadth_5d_rank_weight": 0.30,
            "turnover_ratio_rank_weight": 0.30,
        },
        "families": {
            family: _evaluate_family(
                indicators,
                theme_stock_context,
                cluster_stock_context,
                family=family,
            )
            for family in ("reversal", "pullback")
        },
        "deployment_decision": (
            "初押しは比較1〜3のホールドアウト最良方式を参考順位にだけ使う。"
            "投げ売り反転には使わず、所属履歴が整うまで強制除外もしない"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_backtest(load_technical_shards(args.dataset_dir), args.themes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["families"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
