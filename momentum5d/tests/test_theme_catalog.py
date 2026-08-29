from __future__ import annotations

import pandas as pd

from scripts.backtest_theme_context import _select_comparison
from scripts.build_theme_catalog import THEME_CLUSTERS, build_catalog


def test_theme_catalog_has_124_unique_themes_in_13_clusters() -> None:
    themes = [theme for values in THEME_CLUSTERS.values() for theme in values]

    assert len(THEME_CLUSTERS) == 13
    assert len(themes) == 124
    assert len(set(themes)) == 124


def test_comparison_modes_rank_the_same_candidates_by_each_context_level() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "ticker": "1001.T",
                "_base_score": 3.0,
                "cluster_score": 0.2,
                "theme_score": 0.9,
                "hierarchical_score": 0.6,
            },
            {
                "date": "2026-01-05",
                "ticker": "1002.T",
                "_base_score": 2.0,
                "cluster_score": 0.9,
                "theme_score": 0.1,
                "hierarchical_score": 0.7,
            },
        ]
    )

    selected = _select_comparison(frame, "_base_score")

    assert selected["0_baseline"].iloc[0]["ticker"] == "1001.T"
    assert selected["1_cluster_only"].iloc[0]["ticker"] == "1002.T"
    assert selected["2_theme_only"].iloc[0]["ticker"] == "1001.T"
    assert selected["3_hierarchical"].iloc[0]["ticker"] == "1002.T"


def test_catalog_normalizes_topix_group_from_sector_master(tmp_path) -> None:
    existing = tmp_path / "themes.csv"
    existing.write_text(
        "theme_name,cluster,topix17_group,stock_code,yahoo_ticker,company_name,"
        "short_name,market,source_url\n"
        "人工知能,old,TOPIX Core30,1001,1001.T,会社,会社,東P,source\n",
        encoding="utf-8",
    )
    sectors = tmp_path / "sectors.csv"
    sectors.write_text(
        "code,sector_17_code,sector_33_code\n1001,10,5250\n",
        encoding="utf-8",
    )

    rows = build_catalog(existing, sectors, fetch=False)

    assert rows[0]["cluster"] == "AI・半導体"
    assert rows[0]["topix17_group"] == "情報通信・サービスその他"
