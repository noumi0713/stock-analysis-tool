import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="テクニカル指標 詳細分析", layout="wide")
st.title("📊 テクニカル指標 詳細分析ダッシュボード")

# デフォルトの監視リスト
DEFAULT_TICKERS = {
    "5803": "フジクラ",
    "4425": "Ｋｕｄａｎ",
    "5713": "住友金属鉱山", # JX金属代替
    "7011": "三菱重工業",
    "6752": "パナソニックHD",
    "5802": "住友電気工業",
    "9412": "スカパーJSAT",
    "6503": "三菱電機"
}

st.sidebar.header("⚙️ 銘柄の選択・追加")

# 1. 既存リストからの選択（複数選択可）
selected_from_list = st.sidebar.multiselect(
    "リストから選択（複数可）",
    options=list(DEFAULT_TICKERS.keys()),
    default=["5803", "7011", "6752"], # デフォルトでいくつか選択
    format_func=lambda x: f"{x} {DEFAULT_TICKERS[x]}"
)

# 2. 自由入力欄（カンマ区切りで複数銘柄を同時追加可能）
custom_input = st.sidebar.text_input(
    "📝 新規銘柄コードを追加（カンマ区切り）",
    placeholder="例: 7203, 9984, 8035"
)

# 選択された銘柄と、手入力された銘柄を結合
final_tickers = set(selected_from_list)
if custom_input:
    custom_codes = [code.strip() for code in custom_input.split(',')]
    for code in custom_codes:
        if code.isdigit(): # 数字（コード）として正しいか簡易チェック
            final_tickers.add(code)

final_tickers = list(final_tickers)

# RSI計算用の関数
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if not final_tickers:
    st.warning("左側のサイドバーから銘柄を選択、または入力してください。")
    st.stop()

results = []

with st.spinner("市場データを取得・計算中..."):
    # 複数銘柄を1つずつ安全に取得・処理するループ
    for code in final_tickers:
        try:
            ticker_symbol = f"{code}.T"
            # yfinanceで個別取得
            df = yf.download(ticker_symbol, period="3mo", interval="1d", progress=False)
            
            if df.empty or len(df) < 20:
                st.warning(f"コード {code} のデータが十分に取得できませんでした。")
                continue
                
            # yfinanceの仕様変更対策（マルチインデックスの解除）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # --- 指標の計算 ---
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA5_Deviation'] = ((df['Close'] / df['MA5']) - 1) * 100
            df['RSI'] = calc_rsi(df['Close'], period=14)
            df['Vol_Change'] = df['Volume'].pct_change() * 100
            df['Vol_Change_5d_Avg'] = df['Vol_Change'].rolling(window=5).mean()
            
            df['Prev_Close'] = df['Close'].shift(1)
            # 値幅の割合計算（ゼロ割りを防ぐ処理付き）
            df['Daily_Range_Pct'] = np.where(
                df['Prev_Close'] > 0,
                ((df['High'] - df['Low']) / df['Prev_Close']) * 100,
                0
            )
            df['Volatility_5d_Avg'] = df['Daily_Range_Pct'].rolling(window=5).mean()
            
            # 最新日のデータを取得
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            current_price = float(latest['Close'])
            prev_price = float(prev['Close'])
            dod_pct = ((current_price / prev_price) - 1) * 100
            
            company_name = DEFAULT_TICKERS.get(code, "新規追加銘柄")
            
            results.append({
                "コード": code,
                "銘柄名": company_name,
                "現在値": round(current_price, 1),
                "前日比 (%)": round(dod_pct, 2),
                "RSI (14日)": round(float(latest['RSI']), 1),
                "5日線 乖離率 (%)": round(float(latest['MA5_Deviation']), 2),
                "5日間 出来高変化率平均 (%)": round(float(latest['Vol_Change_5d_Avg']), 1),
                "5日間 平均ボラティリティ (%)": round(float(latest['Volatility_5d_Avg']), 2)
            })
            
        except Exception as e:
            st.error(f"{code}のデータ処理中にエラーが発生しました: {e}")
            continue

if results:
    result_df = pd.DataFrame(results)
    
    st.markdown("### 📈 抽出結果（最新データ）")
    
    with st.expander("📖 各指標の読み方・戦略ガイド"):
        st.write("""
        * **RSI (14日)**: 45〜65が安全圏。70以上は利益確定の目安、40以下は底打ちのサインです。
        * **5日線 乖離率 (%)**: 0%に近い（またはマイナス）ほど、5日線にタッチしており「押し目買い」のチャンスです。+5%を超えると高値掴みのリスクが高まります。
        * **5日間 出来高変化率平均 (%)**: プラスの数値が大きいほど、直近1週間で大口の資金が継続して流入している「強いトレンド」を示します。
        * **5日間 平均ボラティリティ (%)**: 数値が高いほど1日の乱高下が激しい状態です。急騰後はボラティリティが跳ね上がります。数値が低く落ち着いている時が安全な仕込み時です。
        """)
    
    def color_rsi(val):
        color = 'red' if val >= 70 else 'blue' if val <= 40 else 'green'
        return f'color: {color}'
        
    def color_deviation(val):
        color = 'red' if val >= 5 else 'blue' if val <= 0 else 'black'
        return f'color: {color}'

    styled_df = result_df.style\
        .map(color_rsi, subset=['RSI (14日)'])\
        .map(color_deviation, subset=['5日線 乖離率 (%)'])\
        .format({
            "前日比 (%)": "{:+.2f}",
            "5日線 乖離率 (%)": "{:+.2f}",
            "5日間 出来高変化率平均 (%)": "{:+.1f}",
        })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.warning("表示できるデータがありません。")
