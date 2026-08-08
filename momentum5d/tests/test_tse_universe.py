from __future__ import annotations

import pandas as pd

from app.yahoo.universe import build_tse_universe


def test_build_tse_universe_keeps_three_domestic_equity_markets() -> None:
    source = pd.DataFrame(
        [
            {
                "コード": "1301",
                "銘柄名": "Prime Co",
                "市場・商品区分": "プライム（内国株式）",
                "17業種コード": "01",
                "33業種コード": "0050",
            },
            {
                "コード": "200A",
                "銘柄名": "Standard Co",
                "市場・商品区分": "スタンダード（内国株式）",
                "17業種コード": "10",
                "33業種コード": "5250",
            },
            {
                "コード": "607A",
                "銘柄名": "Growth Co",
                "市場・商品区分": "グロース（内国株式）",
                "17業種コード": "10",
                "33業種コード": "9050",
            },
            {
                "コード": "1305",
                "銘柄名": "ETF",
                "市場・商品区分": "ETF・ETN",
                "17業種コード": "-",
                "33業種コード": "-",
            },
        ]
    )

    result = build_tse_universe(source)

    assert result["ticker"].tolist() == ["1301.T", "200A.T", "607A.T"]
    assert result["code"].tolist() == ["13010", "200A0", "607A0"]
    assert result.loc[0, "sector_17_code"] == "1"
    assert result.loc[0, "sector_33_code"] == "0050"
