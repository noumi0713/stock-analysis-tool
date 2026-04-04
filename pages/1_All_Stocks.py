import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="全銘柄スキャン・押し目スクリーナー", layout="wide")
st.title("🌍 全銘柄一括スキャン（スイング専用）")

st.markdown("""
約4,000銘柄など、大量の銘柄コードを一括で読み込み、**「売り枯れ押し目（大本命・本命）」**の条件に合致するものだけを抽出します。
※銘柄数が多いとデータの取得に数分かかる場合があります。
""")

# --- 入力UI ---
st.sidebar.header("📂 銘柄リストの読み込み")

bulk_input = st.sidebar.text_area(
    "銘柄コードを貼り付け", 
    height=150, 
    placeholder="例1: 7203 トヨタ自動車\n例2: 1332, 1605, 1721\n※名前が一緒に入っていれば自動で抽出します"
)

uploaded_file = st.sidebar.file_uploader("またはCSVをアップロード（JPX公式CSV対応）", type=["csv"])

# --- 銘柄抽出・名前マッピングロジック ---
tickers_info = {}

if bulk_input:
    for line in bulk_input.splitlines():
        # 4桁の数字をすべて検索
        matches = re.findall(r'\b\d{4}\b', line)
        if len(matches) > 1:
            # カンマ区切りなどで1行に複数ある場合は名前が判別できないため「-」にする
            for code in matches:
                tickers_info[f"{code}.T"] = "-"
        elif len(matches) == 1:
            # 1行に1銘柄の場合、コード以外の文字を「銘柄名」として抽出
            code = matches[0]
            name = re.sub(r'\b\d{4}\b', '', line).strip(' ,/:"\'\t　')
            tickers_info[f"{code}.T"] = name if name else "-"

if uploaded_file is not None:
    try:
        # JPXのCSVはShift-JISのことが多いのでエンコーディングを考慮して読み込み
        try:
            df_csv = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_csv = pd.read_csv(uploaded_file, encoding='shift_jis')
            
        # 「コード」と「銘柄名」の列があるか確認
        if 'コード' in df_csv.columns and '銘柄名' in df_csv.columns:
            for _, row in df_csv.iterrows():
                code = str(row['コード'])[:4]
                if code.isdigit():
                    tickers_info[f"{code}.T"] = str(row['銘柄名'])
        else:
            # フォーマットが違う場合は強引にテキストとしてコードのみ抽出
            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            for line in content.splitlines():
                matches = re.findall(r'\b\d{4}\b', line)
                for code in matches:
                    tickers_info[f"{code}.T"] = "-"
    except Exception as e:
        st.sidebar.error(f"CSVの読み込みに失敗しました: {e}")

tickers_list = list(tickers_info.keys())
st.sidebar.info(f"読み込み済みの銘柄数: {len(tickers_list)} 銘柄")

# --- 出来高補正 ---
def get_vol_mul():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    if now.weekday() >= 5 or now.hour < 9 or now.hour >= 15: return 1.0
    if 9 <= now.hour < 11 or (now.hour == 11 and now.minute <= 30):
        elaps = (now.hour - 9) * 60 + now.minute
    elif 11 <= now.hour <= 12 and not (now.hour == 12 and now.minute >= 30):
        elaps = 150
    else:
        elaps = 150 + (now.hour - 12) * 60 + now.minute - 30
    return 300 / elaps if elaps > 0 else 1.0

# --- データ取得・解析 ---
@st.cache_data(ttl=300)
def fetch_data(ts):
    return yf.download(ts, period="6mo", interval="1d", group_by="ticker", threads=True)

def analyze(data, ts, t_info):
    dai_honmei_list, honmei_list = [], []
    v_mul = get_vol_mul()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ts)
    
    for i, t in enumerate(ts):
        if i % 10 == 0 or i == total - 1:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"解析中... {i+1} / {total} 銘柄完了")

        try:
            df = data[t].copy() if len(ts) > 1 else data.copy()
            df = df.dropna()
            if len(df) < 40: continue
            
            df['MA25'] = df['Close'].rolling(25).mean()
            df['RecentHigh'] = df['Close'].rolling(20).max()
            
            d = df['Close'].diff()
            g = (d.where(d > 0, 0)).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + g / l))
            
            c = df.iloc[-1]
            p = df.iloc[-2]
            
            recent_high = df['RecentHigh'].iloc[-2]
            drop_pct = ((c['Close'] / recent_high) - 1) * 100
            
            avg_vol = df['Volume'].iloc[-6:-1].mean()
            curr_vol = df['Volume'].iloc[-1] * v_mul
            vol_change_pct = ((curr_vol / avg_vol) - 1) * 100 if avg_vol > 0 else 0
            
            res = {
                "コード": t.replace(".T",""), 
                "銘柄名": t_info.get(t, "-"),
                "現在値": f"{c['Close']:.1f}", 
                "高値から": f"{drop_pct:.1f}%", 
                "RSI": f"{c['RSI']:.1f}", 
                "Vol変化": f"{vol_change_pct:+.1f}%"
            }
            
            is_dai_honmei = (
                -13.0 <= drop_pct <= -7.0 and
                vol_change_pct <= -40.0 and
                35.0 <= c['RSI'] <= 55.0 and
                c['MA25'] * 0.97 <= c['Close'] <= c['MA25'] * 1.05
            )
            
            is_honmei = (
                -15.0 <= drop_pct <= -6.0 and
                vol_change_pct <= -30.0 and
                30.0 <= c['RSI'] <= 60.0 and
                c['Close'] >= c['MA25'] * 0.97
            )
            
            if is_dai_honmei:
                dai_honmei_list.append({**res, "評価": "👑 大本命"})
            elif is_honmei:
                honmei_list.append({**res, "評価": "🎯 本命"})
                
        except Exception:
            continue
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(dai_honmei_list), pd.DataFrame(honmei_list)

# --- 実行と表示 ---
if st.button("🚀 スキャンを実行する", type="primary"):
    if not tickers_list:
        st.warning("左のサイドバーから銘柄コードを入力またはCSVをアップロードしてください。")
    else:
        with st.spinner('市場データを一括ダウンロード中...（銘柄数に応じて時間がかかります）'):
            m_data = fetch_data(tickers_list)
        
        df_dai, df_hon = analyze(m_data, tickers_list, tickers_info)
        
        def style_eval(v):
            if '👑' in str(v): return 'color: #FFD700; font-weight: bold; background-color: #333333'
            if '🎯' in str(v): return 'color: #ff4b4b; font-weight: bold'
            return ''

        # 列の並び順に「銘柄名」を追加
        cols = ["コード", "銘柄名", "現在値", "高値から", "RSI", "Vol変化", "評価"]

        st.header("👑 大本命（完璧な売り枯れ押し目）")
        if not df_dai.empty:
            st.dataframe(df_dai[cols].style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else: 
            st.info("大本命の条件に合致する銘柄はありませんでした。")

        st.header("🎯 本命（監視すべき優良な押し目）")
        if not df_hon.empty:
            df_hon['SortVal'] = abs(df_hon['高値から'].str.replace('%', '').astype(float) + 10.0)
            df_hon = df_hon.sort_values('SortVal').drop(columns=['SortVal'])
            st.dataframe(df_hon[cols].style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else: 
            st.info("本命の条件に合致する銘柄はありませんでした。")
