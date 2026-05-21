import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="テーマ別株式スクリーナー", layout="wide")
st.title("📈 戦略的テーマ監視ダッシュボード")

# ==========================================
# 1. 銘柄データ（厳選21銘柄）
# ==========================================
THEME_DICT = {
    "フィジカルAI・ロボット": [
        {"コード": "6506", "銘柄名": "安川電機"}, 
        {"コード": "6954", "銘柄名": "ファナック"},
        {"コード": "6324", "銘柄名": "ハーモニック・ドライブ・システムズ"}
    ],
    "宇宙": [
        {"コード": "9412", "銘柄名": "スカパーJSAT"}, 
        {"コード": "464A", "銘柄名": "QPSHD"},
        {"コード": "186A", "銘柄名": "アストロスケール"}
    ],
    "銀行": [
        {"コード": "8306", "銘柄名": "三菱UFJ"}, 
        {"コード": "8316", "銘柄名": "三井住友FG"},
        {"コード": "8411", "銘柄名": "みずほFG"}
    ],
    "ドローン": [
        {"コード": "278A", "銘柄名": "テラドローン"}, 
        {"コード": "5597", "銘柄名": "ブルーイノベーション"},
        {"コード": "6232", "銘柄名": "ACSL"}
    ],
    "防衛・重工業": [
        {"コード": "7011", "銘柄名": "三菱重工"}, 
        {"コード": "7013", "銘柄名": "IHI"}
    ],
    "テクノロジー・通信": [
        {"コード": "9984", "銘柄名": "ソフトバンクG"}, 
        {"コード": "6752", "銘柄名": "パナソニック"},
        {"コード": "6976", "銘柄名": "太陽誘電"}, 
        {"コード": "6613", "銘柄名": "QDレーザー"},
        {"コード": "7974", "銘柄名": "任天堂"}
    ],
    "次世代エネルギー・素材": [
        {"コード": "5016", "銘柄名": "JX金属"}, 
        {"コード": "485A", "銘柄名": "パワーエックス"}
    ]
}

# --- セッションステート（ポートフォリオ）の初期化 ---
# 新しいリストに変更したため、バージョン管理で強制上書きする
PORTFOLIO_VERSION = "v2_custom21"
if "portfolio_version" not in st.session_state or st.session_state["portfolio_version"] != PORTFOLIO_VERSION:
    default_list = []
    for theme, stocks in THEME_DICT.items():
        for s in stocks:
            default_list.append({"テーマ": theme, "コード": s["コード"], "銘柄名": s["銘柄名"]})
    st.session_state["my_portfolio"] = default_list
    st.session_state["portfolio_version"] = PORTFOLIO_VERSION

# --- ヘルパー関数 ---
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=600)
def fetch_data(tickers, period="6mo"):
    if not tickers: return None
    # yfinanceの仕様変更に対応するための安全なデータ取得
    data = yf.download(tickers, period=period, interval="1d", group_by="ticker", threads=True, progress=False)
    return data

# ==========================================
# UI構築（4つのタブ）
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 強気銘柄スクリーナー", 
    "✨ トレンド自動発掘", 
    "📅 期間データ抽出", 
    "⚙️ リスト管理"
])

current_portfolio = st.session_state["my_portfolio"]
# 監視リストに登録されている有効なティッカー一覧
active_tickers = [f"{str(item['コード']).strip()}.T" for item in current_portfolio if str(item['コード']).strip()]

# --- タブ1: 強気銘柄スクリーナー ---
with tab1:
    st.write("「リスト管理タブ」に登録されている銘柄の中から、テクニカル的に強い銘柄を自動抽出します。")
    if not active_tickers:
        st.warning("監視リストに銘柄がありません。リスト管理タブで追加してください。")
    else:
        if st.button("🔄 スクリーニングを実行"):
            with st.spinner('市場データを読み込み中...'):
                raw_data = fetch_data(active_tickers, period="6mo")
                
                if raw_data is not None:
                    results = []
                    for item in current_portfolio:
                        code = str(item["コード"]).strip()
                        if not code: continue
                        ticker = f"{code}.T"
                        
                        try:
                            # 1銘柄か複数銘柄かでyfinanceの戻り値の構造が違うため分岐
                            if isinstance(raw_data.columns, pd.MultiIndex):
                                df = raw_data[ticker].dropna()
                            else:
                                df = raw_data.dropna() # 1銘柄のみの場合
                                
                            if len(df) < 25: continue
                            
                            df['MA5'] = df['Close'].rolling(5).mean()
                            df['MA25'] = df['Close'].rolling(25).mean()
                            df['MA75'] = df['Close'].rolling(75).mean()
                            df['RSI'] = calc_rsi(df['Close'])
                            
                            c = df.iloc[-1]
                            p = df.iloc[-2]
                            dod = ((float(c['Close']) / float(p['Close'])) - 1) * 100
                            
                            status = "横ばい"
                            if float(c['Close']) > float(df['MA5'].iloc[-1]) > float(df['MA25'].iloc[-1]) > float(df['MA75'].iloc[-1]):
                                if float(df['MA25'].iloc[-1]) > float(df['MA25'].iloc[-2]):
                                    status = "🌟 パーフェクトオーダー"
                            elif float(c['Close']) > float(df['MA5'].iloc[-1]) > float(df['MA5'].iloc[-2]):
                                status = "📈 5日線上向き"

                            results.append({
                                "テーマ": item.get("テーマ", "未分類"),
                                "コード": code,
                                "銘柄名": item.get("銘柄名", ""),
                                "現在値": round(float(c['Close']), 1),
                                "前日比(%)": round(dod, 2),
                                "RSI": round(float(c['RSI']), 1),
                                "判定": status
                            })
                        except Exception as e:
                            continue
                    
                    analysis_df = pd.DataFrame(results)
                    if not analysis_df.empty:
                        col1, col2 = st.columns(2)
                        po_stocks = analysis_df[analysis_df["判定"] == "🌟 パーフェクトオーダー"].drop_duplicates(subset="コード")
                        ma5_stocks = analysis_df[analysis_df["判定"] == "📈 5日線上向き"].drop_duplicates(subset="コード")
                        
                        with col1:
                            st.success(f"🌟 パーフェクトオーダー中 ({len(po_stocks)}銘柄)")
                            st.dataframe(po_stocks[["テーマ", "コード", "銘柄名", "現在値", "前日比(%)", "RSI"]].sort_values("前日比(%)", ascending=False), hide_index=True)
                        with col2:
                            st.info(f"📈 短期上昇傾向 ({len(ma5_stocks)}銘柄)")
                            st.dataframe(ma5_stocks[["テーマ", "コード", "銘柄名", "現在値", "前日比(%)", "RSI"]].sort_values("前日比(%)", ascending=False), hide_index=True)

# --- タブ2: トレンド自動発掘 ---
with tab2:
    st.subheader("✨ テーマ別トレンド自動発掘")
    st.write("事前定義された16テーマ（全160銘柄）の中から、直近5日間で最も資金が集まっている上位テーマと銘柄を抽出します。")
    
    if st.button("🚀 最新のトレンドを自動分析する", type="primary"):
        with st.spinner("各テーマの市場データを分析中... (1分ほどかかる場合があります)"):
            all_theme_tickers = []
            for stocks in THEME_DICT.values():
                for s in stocks:
                    all_theme_tickers.append(f"{s['コード']}.T")
            
            # 直近1ヶ月のデータを取得
            trend_data = fetch_data(all_theme_tickers, period="1mo")
            
            theme_performance = []
            if trend_data is not None:
                for theme, stocks in THEME_DICT.items():
                    theme_returns = []
                    for s in stocks:
                        ticker = f"{s['コード']}.T"
                        try:
                            if isinstance(trend_data.columns, pd.MultiIndex):
                                df = trend_data[ticker].dropna().copy()
                            else:
                                df = trend_data.dropna().copy()
                                
                            if len(df) >= 15:
                                df['RSI'] = calc_rsi(df['Close'])
                                df['Vol_Change'] = df['Volume'].pct_change() * 100
                                
                                start_price = float(df['Close'].iloc[-6]) # 5日前
                                end_price = float(df['Close'].iloc[-1])
                                pct_change = ((end_price / start_price) - 1) * 100
                                current_rsi = float(df['RSI'].iloc[-1])
                                vol_change_5d = float(df['Vol_Change'].iloc[-5:].mean())
                                
                                theme_returns.append({
                                    "コード": s['コード'],
                                    "銘柄名": s['銘柄名'],
                                    "5日リターン(%)": round(pct_change, 2),
                                    "RSI": round(current_rsi, 1),
                                    "5日間出来高変化率(%)": round(vol_change_5d, 1)
                                })
                        except:
                            continue
                    
                    if theme_returns:
                        theme_df = pd.DataFrame(theme_returns)
                        avg_return = theme_df["5日リターン(%)"].mean()
                        theme_performance.append({
                            "テーマ": theme,
                            "平均5日リターン(%)": round(avg_return, 2),
                            "構成銘柄データ": theme_df
                        })
                
                if theme_performance:
                    # テーマ全体のリターンでソート
                    theme_performance.sort(key=lambda x: x["平均5日リターン(%)"], reverse=True)
                    top_3_themes = theme_performance[:3]
                    
                    st.success("✅ 分析が完了しました！現在のトレンド上位3テーマです。")
                    
                    for i, t_info in enumerate(top_3_themes):
                        st.markdown(f"### 🏆 第{i+1}位: {t_info['テーマ']} (平均リターン: {t_info['平均5日リターン(%)']}%)")
                        sorted_stocks = t_info['構成銘柄データ'].sort_values("5日リターン(%)", ascending=False)
                        st.dataframe(sorted_stocks, use_container_width=True, hide_index=True)
                else:
                    st.error("データの算出に失敗しました。")

# --- タブ3: 期間指定データ抽出 ---
with tab3:
    st.subheader("📅 期間指定データ抽出")
    st.write("リスト管理タブに登録されている銘柄の指定期間データをCSV形式でダウンロードできます。")
    
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=30)
    date_range = st.date_input("期間を選択", [default_start, today])

    if len(date_range) == 2:
        start_date, end_date = date_range
        
        if st.button("📊 データを抽出"):
            if not active_tickers:
                st.warning("リスト管理タブに銘柄が登録されていません。")
            else:
                with st.spinner("期間データを集計中..."):
                    raw_data = fetch_data(active_tickers, period="6mo")
                    if raw_data is not None:
                        all_period_data = []
                        start_ts = pd.Timestamp(start_date)
                        end_ts = pd.Timestamp(end_date)
                        
                        for item in current_portfolio:
                            code = str(item["コード"]).strip()
                            ticker = f"{code}.T"
                            try:
                                if isinstance(raw_data.columns, pd.MultiIndex):
                                    df_hist = raw_data[ticker].dropna()
                                else:
                                    df_hist = raw_data.dropna()
                                    
                                df_hist['RSI'] = calc_rsi(df_hist['Close'])
                                mask = (df_hist.index >= start_ts) & (df_hist.index <= end_ts)
                                df_filtered = df_hist.loc[mask]
                                
                                for idx, row in df_filtered.iterrows():
                                    all_period_data.append({
                                        "日付": idx.date(),
                                        "テーマ": item.get("テーマ", "未分類"),
                                        "コード": code,
                                        "銘柄名": item.get("銘柄名", ""),
                                        "引値": round(row['Close'], 1),
                                        "出来高": int(row['Volume']),
                                        "RSI": round(row['RSI'], 1) if not np.isnan(row['RSI']) else "算出中"
                                    })
                            except:
                                continue
                                
                        if all_period_data:
                            df_range_view = pd.DataFrame(all_period_data).sort_values(["日付", "テーマ"], ascending=[False, True])
                            st.success(f"{len(df_range_view)} 件のデータを抽出しました。")
                            st.dataframe(df_range_view, use_container_width=True, hide_index=True)
                            
                            csv = df_range_view.to_csv(index=False).encode('utf-8-sig')
                            st.download_button("💾 抽出結果をCSVで保存", data=csv, file_name=f"stock_data_{start_date}_to_{end_date}.csv", mime="text/csv")
                        else:
                            st.warning("指定期間のデータが見つかりませんでした。")

# --- タブ4: 監視リスト管理 ---
with tab4:
    st.subheader("⚙️ 監視リスト管理")
    st.write("スクリーニングやデータ抽出の対象となる銘柄リストをここで直接編集できます。行の追加・削除やエクセルからのコピペも可能です。")
    
    # st.data_editorを使ってインタラクティブなテーブルを表示
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state["my_portfolio"]),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("💾 リストへの変更を保存して適用", type="primary"):
        st.session_state["my_portfolio"] = edited_df.to_dict('records')
        st.success("変更を保存しました！他のタブに移動すると最新のリストで分析が行われます。")
        st.rerun()
