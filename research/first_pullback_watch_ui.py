from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from first_pullback_watch_prototype import (
    STATUS_BUY_NOW,
    STATUS_FORMING,
    STATUS_INVALID,
    STATUS_RAN_AWAY,
    STATUS_SIGNAL,
    STATUS_WAIT,
    process,
)

DEFAULT_STATE = Path("research/state/first_pullback_watch.json")

STATUS_LABELS = {
    STATUS_BUY_NOW: "今買える",
    STATUS_WAIT: "初押し待ち",
    STATUS_FORMING: "押し目形成中",
    STATUS_SIGNAL: "初押しシグナル",
    STATUS_RAN_AWAY: "上昇しすぎ除外",
    STATUS_INVALID: "失格",
}

STATUS_ORDER = {
    STATUS_SIGNAL: 0,
    STATUS_BUY_NOW: 1,
    STATUS_FORMING: 2,
    STATUS_WAIT: 3,
    STATUS_RAN_AWAY: 4,
    STATUS_INVALID: 5,
}


def demo_state() -> dict[str, Any]:
    state: dict[str, Any] = {"watchlist": []}
    days = [
        {
            "as_of": "2026-09-01",
            "candidates": [
                {
                    "ticker": "9999.T",
                    "name": "Prototype AI",
                    "discovery": {
                        "ticker": "9999.T",
                        "name": "Prototype AI",
                        "discovery_price": 1000,
                        "minkabu_relevance": 92,
                        "kabutan_member": True,
                        "ifis_rank": 8,
                        "ifis_access_change_pct": 145,
                        "theme": "AI・半導体",
                        "catalyst_reason": "大型受注を想定したデモ材料",
                    },
                    "technical": {
                        "open": 1030,
                        "high": 1080,
                        "low": 1020,
                        "close": 1070,
                        "volume": 1_000_000,
                        "rsi14": 74,
                        "ma25": 980,
                        "atr14": 38,
                        "overheat_score": 46,
                        "ifis_rank": 8,
                        "ifis_access_change_pct": 145,
                        "minkabu_theme_rank": 3,
                    },
                },
                {
                    "ticker": "8888.T",
                    "name": "Prototype Mobility",
                    "discovery": {
                        "ticker": "8888.T",
                        "name": "Prototype Mobility",
                        "discovery_price": 2000,
                        "minkabu_relevance": 86,
                        "kabutan_member": True,
                        "ifis_rank": 14,
                        "ifis_access_change_pct": 92,
                        "theme": "自動運転車",
                        "catalyst_reason": "新規提携を想定したデモ材料",
                    },
                    "technical": {
                        "open": 2010,
                        "high": 2050,
                        "low": 1995,
                        "close": 2035,
                        "volume": 620_000,
                        "rsi14": 61,
                        "ma25": 1960,
                        "atr14": 55,
                        "overheat_score": 18,
                        "ifis_rank": 14,
                        "ifis_access_change_pct": 92,
                        "minkabu_theme_rank": 8,
                    },
                },
            ],
        },
        {
            "as_of": "2026-09-02",
            "candidates": [
                {
                    "ticker": "9999.T",
                    "technical": {
                        "open": 1050,
                        "high": 1060,
                        "low": 1015,
                        "close": 1025,
                        "volume": 650_000,
                        "rsi14": 61,
                        "ma25": 985,
                        "atr14": 36,
                        "overheat_score": 22,
                        "ifis_rank": 10,
                        "ifis_access_change_pct": 110,
                        "minkabu_theme_rank": 4,
                    },
                },
                {
                    "ticker": "8888.T",
                    "technical": {
                        "open": 2030,
                        "high": 2070,
                        "low": 2020,
                        "close": 2060,
                        "volume": 580_000,
                        "rsi14": 63,
                        "ma25": 1970,
                        "atr14": 54,
                        "overheat_score": 19,
                        "ifis_rank": 11,
                        "ifis_access_change_pct": 101,
                        "minkabu_theme_rank": 7,
                    },
                },
            ],
        },
        {
            "as_of": "2026-09-03",
            "candidates": [
                {
                    "ticker": "9999.T",
                    "technical": {
                        "open": 1030,
                        "high": 1068,
                        "low": 1028,
                        "close": 1065,
                        "volume": 720_000,
                        "rsi14": 64,
                        "ma25": 990,
                        "atr14": 35,
                        "overheat_score": 25,
                        "ifis_rank": 7,
                        "ifis_access_change_pct": 125,
                        "minkabu_theme_rank": 3,
                    },
                },
                {
                    "ticker": "8888.T",
                    "technical": {
                        "open": 2055,
                        "high": 2080,
                        "low": 2040,
                        "close": 2070,
                        "volume": 540_000,
                        "rsi14": 64,
                        "ma25": 1980,
                        "atr14": 52,
                        "overheat_score": 20,
                        "ifis_rank": 12,
                        "ifis_access_change_pct": 96,
                        "minkabu_theme_rank": 6,
                    },
                },
            ],
        },
    ]
    for payload in days:
        state = process(payload, state)
    return state


def load_state(uploaded: Any, use_demo: bool) -> dict[str, Any]:
    if uploaded is not None:
        return json.load(uploaded)
    if DEFAULT_STATE.exists() and not use_demo:
        return json.loads(DEFAULT_STATE.read_text(encoding="utf-8"))
    return demo_state()


def latest_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    history = item.get("history") or []
    return history[-1] if history else {}


def table_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        snap = latest_snapshot(item)
        rows.append(
            {
                "状態": STATUS_LABELS.get(item.get("status"), item.get("status", "")),
                "コード": item.get("ticker"),
                "銘柄": item.get("name"),
                "テーマ": item.get("theme") or "—",
                "終値": snap.get("close"),
                "発見時比%": snap.get("gain_from_discovery_pct"),
                "高値から%": -float(snap.get("pullback_from_reference_high_pct") or 0),
                "RSI14": snap.get("rsi14"),
                "過熱度": snap.get("overheat_score"),
                "IFIS順位": snap.get("ifis_rank"),
                "IFIS変化%": snap.get("ifis_access_change_pct"),
                "みんかぶ関連度": item.get("minkabu_relevance"),
                "更新日": item.get("last_updated"),
            }
        )
    return pd.DataFrame(rows)


def history_chart(item: dict[str, Any]) -> go.Figure:
    hist = pd.DataFrame(item.get("history") or [])
    fig = go.Figure()
    if hist.empty:
        return fig

    fig.add_trace(
        go.Candlestick(
            x=hist["date"],
            open=hist["open"],
            high=hist["high"],
            low=hist["low"],
            close=hist["close"],
            name="株価",
        )
    )
    if "ma25" in hist.columns:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["ma25"], mode="lines", name="25MA"))
    fig.add_hline(y=float(item.get("discovery_price") or 0), line_dash="dot", annotation_text="発見価格")
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    return fig


def status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in STATUS_LABELS}
    for item in items:
        status = item.get("status")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    st.set_page_config(page_title="初押し監視プロトタイプ", page_icon="📈", layout="wide")
    st.title("初押し監視プロトタイプ")
    st.caption("研究用UI｜みんかぶ × 株探 × IFIS候補を、初押し到来まで毎日再判定")

    with st.sidebar:
        st.header("データ")
        uploaded = st.file_uploader("状態JSONを読み込む", type=["json"])
        use_demo = st.toggle("デモデータを表示", value=not DEFAULT_STATE.exists())
        st.caption("本番売買ロジックとは分離されています。閾値はプロトタイプ値です。")

    try:
        state = load_state(uploaded, use_demo)
    except Exception as exc:
        st.error(f"状態JSONを読み込めません: {exc}")
        st.stop()

    items = list(state.get("watchlist") or [])
    items.sort(key=lambda x: (STATUS_ORDER.get(x.get("status"), 99), x.get("ticker", "")))
    counts = status_counts(items)

    cols = st.columns(6)
    metrics = [
        ("初押しシグナル", counts.get(STATUS_SIGNAL, 0)),
        ("今買える", counts.get(STATUS_BUY_NOW, 0)),
        ("押し目形成中", counts.get(STATUS_FORMING, 0)),
        ("初押し待ち", counts.get(STATUS_WAIT, 0)),
        ("上昇しすぎ除外", counts.get(STATUS_RAN_AWAY, 0)),
        ("失格", counts.get(STATUS_INVALID, 0)),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        all_labels = [STATUS_LABELS[s] for s in STATUS_LABELS]
        selected_statuses = st.multiselect("状態で絞り込み", all_labels, default=all_labels)
    with right:
        query = st.text_input("銘柄コード・銘柄名検索")

    filtered = []
    for item in items:
        label = STATUS_LABELS.get(item.get("status"), item.get("status", ""))
        if label not in selected_statuses:
            continue
        haystack = f"{item.get('ticker', '')} {item.get('name', '')}".lower()
        if query and query.lower() not in haystack:
            continue
        filtered.append(item)

    st.subheader("監視一覧")
    if not filtered:
        st.info("条件に一致する銘柄はありません。")
        st.stop()

    df = table_frame(filtered)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "終値": st.column_config.NumberColumn(format="%.1f"),
            "発見時比%": st.column_config.NumberColumn(format="%.2f%%"),
            "高値から%": st.column_config.NumberColumn(format="%.2f%%"),
            "RSI14": st.column_config.NumberColumn(format="%.1f"),
            "過熱度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "IFIS変化%": st.column_config.NumberColumn(format="%.1f%%"),
            "みんかぶ関連度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )

    choices = {f"{item.get('ticker')}｜{item.get('name')}": item for item in filtered}
    selected_name = st.selectbox("詳細を見る銘柄", list(choices))
    item = choices[selected_name]
    snap = latest_snapshot(item)

    st.divider()
    st.subheader(f"{item.get('ticker')} {item.get('name')}")
    status_label = STATUS_LABELS.get(item.get("status"), item.get("status", ""))
    st.markdown(f"### 判定：{status_label}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("発見価格", f"{float(item.get('discovery_price') or 0):,.1f}")
    m2.metric("現在値", f"{float(snap.get('close') or 0):,.1f}", f"{float(snap.get('gain_from_discovery_pct') or 0):+.2f}%")
    m3.metric("高値から", f"{-float(snap.get('pullback_from_reference_high_pct') or 0):.2f}%")
    m4.metric("RSI14", f"{float(snap.get('rsi14') or 0):.1f}")
    m5.metric("過熱度", f"{float(snap.get('overheat_score') or 0):.0f}/100")

    detail_left, detail_right = st.columns([3, 2])
    with detail_left:
        st.plotly_chart(history_chart(item), use_container_width=True)
    with detail_right:
        st.markdown("#### 発見情報")
        st.write(f"**テーマ:** {item.get('theme') or '—'}")
        st.write(f"**みんかぶ関連度:** {float(item.get('minkabu_relevance') or 0):.0f}")
        st.write(f"**株探所属:** {'YES' if item.get('kabutan_member') else 'NO'}")
        st.write(f"**IFIS発見時順位:** {item.get('ifis_rank_at_discovery') or '—'}")
        st.write(f"**材料:** {item.get('catalyst_reason') or '未入力'}")
        st.markdown("#### 今日の判定理由")
        reasons = item.get("last_reasons") or []
        if reasons:
            for reason in reasons:
                st.write(f"・{reason}")
        else:
            st.write("判定理由なし")

    st.markdown("#### 日次履歴")
    history = pd.DataFrame(item.get("history") or [])
    if not history.empty:
        history_view = history.copy()
        if "decision" in history_view.columns:
            history_view["decision"] = history_view["decision"].map(lambda x: STATUS_LABELS.get(x, x))
        keep = [
            c
            for c in [
                "date",
                "close",
                "gain_from_discovery_pct",
                "pullback_from_reference_high_pct",
                "rsi14",
                "overheat_score",
                "volume",
                "ifis_rank",
                "ifis_access_change_pct",
                "minkabu_theme_rank",
                "decision",
            ]
            if c in history_view.columns
        ]
        st.dataframe(history_view[keep].sort_values("date", ascending=False), use_container_width=True, hide_index=True)

    st.caption(f"状態基準日: {state.get('as_of', '—')}｜研究用プロトタイプ。売買判断の正式ルールではありません。")


if __name__ == "__main__":
    main()
