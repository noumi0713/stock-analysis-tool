import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="国策銘柄・精密スクリーナー", layout="wide")
st.title("🏹 買い時・即戦力スクリーナー")

# --- ベース銘柄データの定義 ---
def get_base_tickers():
   raw_data = """
    1. AI・半導体
    6857/アドバンテスト 8035/東京エレクトロン 6723/ルネサスエレクトロニクス 6920/レーザーテック 7735/ＳＣＲＥＥＮホールディングス 6963/ローム 6707/サンケン電気 7282/豊田合成 9984/ソフトバンクグループ 6501/日立製作所
    
    （中略：2〜23はそのまま）

    24. 蓄電池（次世代電池・材料含む）
    6752/パナソニックホールディングス 6762/ＴＤＫ 6981/村田製作所 4204/積水化学工業 4118/カネカ 4107/伊勢化学工業 3407/旭化成 6502/東芝 6810/マクセル 485A/パワーエックス 6617/東光高岳

    25. 再エネ
    9519/レノバ 1407/ウエストホールディングス
    26. 石炭
    1514/住石ホールディングス 8835/太平洋興発
    27. 天然ガス（追加）
    1663/Ｋ＆Ｏエナジーグループ 1963/日揮ホールディングス
    28. 電力卸
    9513/電源開発 9517/イーレックス
    29. 肥料
    4031/片倉コープアグリ 4979/ＯＡＴアグリオ
    30. バイオ燃料
    9212/ＧｒｅｅｎＥａｒｔｈＩｎｓｔｉｔｕｔｅ 2931/ユーグレナ
    """
    ticker_data = {}
    current_theme = "不明"
    lines = raw_data.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        if '/' not in line:
            current_theme = line 
        else:
            stocks = line.split()
            for s in stocks:
                if '/' in s:
                    code, name = s.split('/')
                    ticker_key = f"{code}.T"
                    if ticker_key in ticker_data:
                        if current_theme not in ticker_data[ticker_key]["theme"]:
                            ticker_data[ticker_key]["theme"] += f", {current_theme}"
                    else:
                        ticker_data[ticker_key] = {"name": name, "theme": current_theme}
    return ticker_data

if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = get_base_tickers()
else:
    _first_val = next(iter(st.session_state.tickers_dict.values()), None)
    if isinstance(_first_val, str):
        st.session_state.tickers_dict = get_base_tickers()

# --- サイドバー管理 ---
st.sidebar.header("⚙️ 監視銘柄の管理")
st.sidebar.info(f"監視対象: {len(st.session_state.tickers_dict)} 銘柄")

st.sidebar.subheader("➕ 銘柄追加")
custom_input = st.sidebar.text_area("形式: コード/銘柄名\n例: 7203/トヨタ自動車", height=100)
if st.sidebar.button("銘柄を追加"):
    if custom_input:
        matches = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', custom_input)
        for c, n in matches:
            st.session_state.tickers_dict[f"{c}.T"] = {"name": n, "theme": "ユーザー追加"}
        st.rerun()

st.sidebar.subheader("🗑️ 銘柄削除")
current_codes = list(st.session_state.tickers_dict.keys())
del_options = {t: f"{t.replace('.T','')}/{st.session_state.tickers_dict[t]['name']}" for t in current_codes}
remove_targets = st.sidebar.multiselect("削除対象を選択", options=current_codes, format_func=lambda x: del_options.get(x, x))
if st.sidebar.button("選択した銘柄を削除"):
    for t in remove_targets: st.session_state.tickers_dict.pop(t, None)
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 初期リスト(24テーマ)にリセット"):
    st.session_state.tickers_dict = get_base_tickers()
    st.rerun()

# --- 出来高補正ロジック ---
def get_volume_multiplier():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    if now.weekday() >= 5: return 1.0
    if now.hour < 9 or now.hour >= 15: return 1.0
        
    if 9 <= now.hour < 11 or (now.hour == 11 and now.minute <= 30):
        elapsed = (now.hour - 9) * 60 + now.minute
    elif 11 <= now.hour <= 12 and not (now.hour == 12 and now.minute >= 30):
        elapsed = 150
    else:
        elapsed = 150 + (now.hour - 12) * 60 + now.minute - 30
        
    if elapsed <= 0: return 1.0
    return 300 / elapsed

# --- データの取得と解析 ---
@st.cache_data(ttl=300)
def fetch_market_data(tickers):
    return yf.download(tickers, period="6mo", interval="1d", group_by="ticker", threads=True)

def analyze_signals(data, tickers, tickers_dict, is_strict_mode):
    buy_now, buy_ready = [], []
    vol_multiplier = get_volume_multiplier()

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            df = df.dropna()
            if len(df) < 40: continue

            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA25'] = df['Close'].rolling(window=25).mean()
            df['STD'] = df['Close'].rolling(window=25).std()
            df['Lower2'] = df['MA25'] - (df['STD'] * 2)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            avg_vol = df['Volume'].iloc[-6:-1].mean()
            projected_vol = df['Volume'].iloc[-1] * vol_multiplier
            vol_ratio = (projected_vol / avg_vol) if avg_vol > 0 else 1.0

            curr, prev = df.iloc[-1], df.iloc[-2]
            
            # --- 新しいトレンド判定条件 ---
            is_ma5_up = curr['MA5'] >= prev['MA5']
            is_ma5_above_ma25 = curr['MA5'] > curr['MA25']
            is_strong_uptrend = is_ma5_up and is_ma5_above_ma25
            
            if is_strong_uptrend:
                trend_status = "🔴 MA5>25＆上昇"
            elif is_ma5_up:
                trend_status = "🟡 MA5上昇(25線下)"
            else:
                trend_status = "🔵 下落(⤵️)"

            res = {
                "コード": ticker.replace(".T",""), 
                "銘柄名": tickers_dict[ticker]["name"],
                "テーマ": tickers_dict[ticker]["theme"],
                "現在値": f"{curr['Close']:.1f}", 
                "RSI": f"{curr['RSI']:.1f}",
                "トレンド": trend_status,
                "Vol変化": f"{(vol_ratio-1)*100:+.1f}%"
            }

            is_5ma_cross = curr['Close'] > curr['MA5'] and prev['Close'] <= prev['MA5']

            if is_strict_mode:
                # 【厳格モード】MA5>MA25かつ上向き の状態で5日線を突破
                if is_strong_uptrend and is_5ma_cross and curr['RSI'] > 30 and curr['Close'] < curr['MA25'] * 1.05:
                    status = "🔥 強い上昇の初動 (厳格)"
                    if vol_ratio > 1.0: 
                        status += " (買い流入)"
                    buy_now.append({**res, "状況": status})

                elif is_ma5_up and curr['Close'] <= curr['Lower2'] and curr['RSI'] <= 30:
                    status = "⏳ 極限底打ち待ち"
                    if vol_ratio < 0.8: 
                        status += " (完全売り枯れ)"
                    buy_ready.append({**res, "状況": status})
            else:
                if is_5ma_cross and curr['RSI'] > 30 and curr['Close'] < curr['MA25'] * 1.05:
                    status = "🔥 5日線突破"
                    if vol_ratio > 1.2: 
                        status += " (買い流入強)"
                    buy_now.append({**res, "状況": status})

                elif curr['Close'] <= curr['Lower2'] * 1.02 or curr['RSI'] <= 35:
                    status = "⏳ 底打ち待ち"
                    if vol_ratio < 0.8: 
                        status += " (売り枯れ兆候)"
                    buy_ready.append({**res, "状況": status})

        except: continue
    return pd.DataFrame(buy_now), pd.DataFrame(buy_ready)

# --- メイン UI ---
tickers_list = list(st.session_state.tickers_dict.keys())

if tickers_list:
    with st.spinner('市場データを取得中...（初回のみ数秒かかります）'):
        market_data = fetch_market_data(tickers_list)

    st.markdown("### ⚙️ スクリーニング条件の設定")
    mode_selection = st.radio(
        "相場の状況に合わせて条件を切り替えてください：",
        ["通常モード（平時・上昇相場向け：多くのチャンスを検出）", "厳格モード（暴落・パニック相場向け：ダマシを極限まで排除）"],
        index=1
    )
    is_strict = "厳格" in mode_selection

    if is_strict:
        st.warning("**【厳格モード作動中】** 「5日線が25日線より上に位置し、かつ上向き（強い上昇トレンド）」の状態で反発した銘柄のみを抽出します。")

    df_now, df_ready = analyze_signals(market_data, tickers_list, st.session_state.tickers_dict, is_strict)

    def style_trend(val):
        if '🔴' in str(val): return 'color: #ff4b4b; font-weight: bold'
        if '🟡' in str(val): return 'color: #faca2b; font-weight: bold'
        if '🔵' in str(val): return 'color: #1c83e1; font-weight: bold'
        return ''

    st.header("🎯 買い向かうべきタイミング（即戦力）")
    if not df_now.empty:
        cols = ["コード", "銘柄名", "テーマ", "現在値", "RSI", "トレンド", "Vol変化", "状況"]
        # ★修正箇所：applymap() を map() に変更
        st.dataframe(df_now[cols].style.map(style_trend, subset=['トレンド']), use_container_width=True, hide_index=True)
    else:
        st.info("現在、シグナル合致銘柄はありません。")

    st.header("⏳ 買い準備・監視候補")
    if not df_ready.empty:
        df_ready['RSI_val'] = df_ready['RSI'].astype(float)
        df_ready = df_ready.sort_values("RSI_val").drop(columns=['RSI_val'])
        cols = ["コード", "銘柄名", "テーマ", "現在値", "RSI", "トレンド", "Vol変化", "状況"]
        # ★修正箇所：applymap() を map() に変更
        st.dataframe(df_ready[cols].style.map(style_trend, subset=['トレンド']), use_container_width=True, hide_index=True)
