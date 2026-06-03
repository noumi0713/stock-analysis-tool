import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import re
import io
import requests

# ページ設定
st.set_page_config(page_title="モメンタム投資アナライザー", layout="wide", initial_sidebar_state="expanded")
st.title("📈 モメンタム投資 システムアナライザー")

# ==========================================
# データ取得関数群
# ==========================================
@st.cache_data(ttl=300)
def fetch_stock_data(ticker_symbol, period):
    time.sleep(1) # BAN対策
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period)
    stock_name = f"銘柄コード {ticker_symbol.replace('.T', '')}"
    return df, stock_name

@st.cache_data(ttl=600)
def fetch_trading_value_ranking():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    all_dfs = []
    try:
        for page in range(1, 3):
            url = f"https://minkabu.jp/ranking/trade_value?page={page}"
            res = requests.get(url, headers=headers, timeout=10)
            dfs = pd.read_html(res.content)
            
            if dfs:
                ranking_df = max(dfs, key=len)
                if len(ranking_df) > 10:
                    all_dfs.append(ranking_df)
            time.sleep(1)
            
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df.head(50)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# サイドバー（モード切替と入力フォーム）
# ==========================================
st.sidebar.header("🧭 メニュー")
app_mode = st.sidebar.radio("機能を選択してください", ["📊 個別銘柄アナライザー", "🏆 東証 売買代金ランキング"])

st.sidebar.markdown("---")

# ==========================================
# モード①: 個別銘柄アナライザー
# ==========================================
if app_mode == "📊 個別銘柄アナライザー":
    st.sidebar.header("検索条件")
    st.sidebar.info("複数の銘柄コードをカンマ(,)やスペース区切りで入力できます。")

    ticker_input = st.sidebar.text_input("銘柄コード", value="9984, 4063, 6723, 6506")

    period = st.sidebar.selectbox(
        "取得期間", 
        options=["1mo", "3mo", "6mo", "1y", "max"], 
        index=0, 
        format_func=lambda x: {"1mo": "1ヶ月", "3mo": "3ヶ月", "6mo": "半年", "1y": "1年", "max": "全期間"}[x]
    )

    valid_codes = re.findall(r'\b\d{4}\b', ticker_input)
    valid_codes = list(dict.fromkeys(valid_codes))

    entry_prices = {}
    if valid_codes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("実際の買値（任意）")
        for code in valid_codes:
            entry_prices[code] = st.sidebar.number_input(f"{code} の買値", value=0, step=100, key=f"entry_{code}")

    if valid_codes:
        tabs = st.tabs(valid_codes)
        all_details_list = []
        
        for idx, ticker_code in enumerate(valid_codes):
            with tabs[idx]:
                ticker_symbol = f"{ticker_code}.T"
                entry_price = entry_prices[ticker_code]
                
                try:
                    with st.spinner(f"{ticker_code} のデータを取得・計算中..."):
                        df, stock_name = fetch_stock_data(ticker_symbol, period)

                    if df.empty:
                        st.error(f"{ticker_code} のデータが取得できませんでした。")
                        continue

                    # 指標計算
                    df['SMA5'] = df['Close'].rolling(window=5).mean()
                    df['乖離率(%)'] = ((df['Close'] - df['SMA5']) / df['SMA5']) * 100
                    df['売買代金(億円)'] = (df['Close'] * df['Volume']) / 100000000

                    delta = df['Close'].diff()
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    ema_up = up.ewm(com=13, adjust=False).mean()
                    ema_down = down.ewm(com=13, adjust=False).mean()
                    rs = ema_up / ema_down
                    df['RSI(14)'] = 100 - (100 / (1 + rs))

                    latest = df.iloc[-1]
                    current_price = latest['Close']
                    current_sma5 = latest['SMA5']
                    current_dev = latest['乖離率(%)']
                    current_tv = latest['売買代金(億円)']
                    current_rsi = latest['RSI(14)']
                    
                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100

                    # ジャッジメント
                    if current_price < current_sma5:
                        status = "⚠️ 5日線割れ"
                        status_color = "error"
                        action = "エントリー不可 / 保有中の場合は即時撤退"
                    elif current_dev >= 5.0:
                        status = "🔴 過熱警戒 (+5%超)"
                        status_color = "error"
                        action = f"高値掴みリスク大（新規見送り）。保有中の場合は逆指値を {int(current_sma5)}円付近へ引き上げ推奨。"
                    elif current_dev <= 3.0:
                        status = "🟢 順張り継続 (ストライクゾーン)"
                        status_color = "success"
                        action = "トレンドフォローの最適圏内。売買代金を確認して一点突破を検討。"
                    else:
                        status = "🟡 警戒圏内 (+3%〜5%)"
                        status_color = "warning"
                        action = "新規買いは慎重に。保有中の場合は逆指値を維持。"

                    if entry_price > 0:
                        stop_loss_line = entry_price * 0.92
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        stop_loss_line = current_price * 0.92
                        pnl_pct = 0

                    # UI表示
                    st.subheader(f"■ {stock_name} の現在ステータス")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("現在値", f"{int(current_price):,}円", f"{int(price_change):,}円 ({price_change_pct:.2f}%)")
                    col2.metric("5日線", f"{int(current_sma5):,}円")
                    col3.metric("5日線 乖離率", f"{current_dev:.2f}%")
                    col4.metric("RSI (14日)", f"{current_rsi:.1f}")
                    col5.metric("直近 売買代金", f"{current_tv:,.0f} 億円")

                    st.markdown("---")
                    
                    if status_color == "success":
                        st.success(f"**システム判定:** {status} \n\n **推奨アクション:** {action}")
                    elif status_color == "warning":
                        st.warning(f"**システム判定:** {status} \n\n **推奨アクション:** {action}")
                    else:
                        st.error(f"**システム判定:** {status} \n\n **推奨アクション:** {action}")

                    col_sl, col_pnl = st.columns(2)
                    with col_sl:
                        st.info(f"🛡️ **絶対防衛線 (-8%逆指値):** {int(stop_loss_line):,}円 に設定")
                    with col_pnl:
                        if entry_price > 0:
                            st.info(f"💰 **現在の損益:** {pnl_pct:.2f}%")
                        else:
                            st.info("💡 サイドバーで買値を入力すると現在の損益(%)が計算されます")

                    st.markdown("---")

                    # UI表示: チャート
                    st.subheader("📊 チャート・売買代金・RSI")
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])

                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                                low=df['Low'], close=df['Close'], name="価格"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], mode='lines', 
                                            line=dict(color='magenta', width=2), name="5日線"), row=1, col=1)

                    if entry_price > 0:
                        fig.add_hline(y=entry_price, line_dash="dash", line_color="blue", annotation_text="買値", row=1, col=1)
                        fig.add_hline(y=stop_loss_line, line_dash="dash", line_color="red", annotation_text="損切り(-8%)", row=1, col=1)

                    fig.add_trace(go.Bar(x=df.index, y=df['売買代金(億円)'], marker_color='orange', name="売買代金"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI(14)'], mode='lines',
                                            line=dict(color='purple', width=1.5), name="RSI(14)"), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="blue", line_width=1, row=3, col=1)

                    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified')
                    fig.update_yaxes(range=[0, 100], row=3, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                    # 生データ
                    with st.expander("詳細データ（直近10営業日）"):
                        display_df = df[['Close', 'SMA5', '乖離率(%)', 'RSI(14)', 'Volume', '売買代金(億円)']].tail(10).iloc[::-1]
                        display_df.index = display_df.index.strftime('%Y-%m-%d')
                        display_df = display_df.round({'Close': 0, 'SMA5': 1, '乖離率(%)': 2, 'RSI(14)': 1, '売買代金(億円)': 0})
                        st.dataframe(display_df, use_container_width=True)
                        
                        export_df = display_df.copy()
                        export_df.insert(0, '銘柄コード', ticker_code)
                        export_df.insert(1, '銘柄名', stock_name)
                        all_details_list.append(export_df)

                except Exception as e:
                    st.error(f"エラーが発生しました ({ticker_code}): {e}")

        # ダウンロードセクション
        if all_details_list:
            combined_df = pd.concat(all_details_list)
            st.sidebar.markdown("---")
            st.sidebar.subheader("データのエクスポート")
            
            # 【新規追加】Googleスプレッドシート互換のCSVダウンロード
            # utf-8-sigにすることで、スプレッドシートや日本のExcelで開いても絶対に文字化けしません
            csv_data = combined_df.to_csv(index=True).encode('utf-8-sig')
            st.sidebar.download_button(
                label="🟢 Googleスプレッドシート用 CSVを保存",
                data=csv_data,
                file_name="momentum_analysis_data.csv",
                mime="text/csv",
                key="dl_all_csv"
            )
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='モメンタム分析データ')
            excel_data = excel_buffer.getvalue()

            st.sidebar.download_button(
                label="📊 Excel形式でダウンロード (.xlsx)",
                data=excel_data,
                file_name="momentum_analysis_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_all_excel"
            )
    else:
        st.info("👈 サイドバーに4桁の銘柄コードを入力してください。")

# ==========================================
# モード②: 東証 売買代金ランキング
# ==========================================
elif app_mode == "🏆 東証 売買代金ランキング":
    st.subheader("🔥 東証 売買代金ランキング (トップ50)")
    st.write("「市場の資金が今どこに最も集中しているか」を可視化します。")
    
    with st.spinner("最新のランキングを取得中..."):
        ranking_df = fetch_trading_value_ranking()
        
    if not ranking_df.empty:
        ranking_df.index = range(1, len(ranking_df) + 1)
        st.dataframe(ranking_df, use_container_width=True, height=600)
        
        target_cols = [col for col in ranking_df.columns if any(x in str(col) for x in ['コード', '銘柄', '名称', 'name', 'code'])]
        codes_list = []
        
        if target_cols:
            for col in target_cols:
                for val in ranking_df[col]:
                    match = re.search(r'\b([1-9]\d{3})\b', str(val))
                    if match:
                        code = match.group(1)
                        if code not in codes_list and 1300 <= int(code) <= 9999:
                            codes_list.append(code)
                if codes_list:
                    break
        
        if not codes_list:
            raw_text = ranking_df.to_string()
            matches = re.findall(r'\b([1-9]\d{3})\b', raw_text)
            for match in matches:
                if match not in codes_list and 1300 <= int(match) <= 9999:
                    codes_list.append(match)
                    
        if codes_list:
            st.markdown("### 🎯 アナライザー用 ワンタッチ入力コード")
            
            top10 = ", ".join(codes_list[:10])
            top20 = ", ".join(codes_list[:20])
            top50 = ", ".join(codes_list[:50])
            
            st.text_input("🏆 トップ10銘柄", value=top10)
            st.text_input("🔥 トップ20銘柄", value=top20)
            with st.expander("全50銘柄のコード"):
                st.code(top50)
    else:
        st.error("ランキングデータの自動取得に失敗しました。")
        st.markdown("- [🔗 みんかぶ - 売買代金ランキング](https://minkabu.jp/ranking/trade_value)")
