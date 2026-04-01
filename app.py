import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="テクニカル抽出ツール", layout="wide")
st.title("テクニカル指標 抽出ツール")
st.write("分析したい銘柄コードをカンマ区切りで入力してください。")

# デフォルトでユーザー指定の161銘柄の一部などを入れておくことも可能です
default_tickers = "7011, 8053, 9432, 9501, 1963"
user_input = st.text_area("銘柄コード（例: 7011, 8053）", default_tickers)

if st.button("分析実行（直近5日分）"):
    with st.spinner("データ取得・計算を実行中..."):
        raw_codes = [code.strip() for code in user_input.split(',') if code.strip() != '']
        tickers = [f"{code}.T" if code.isdigit() else code for code in raw_codes]
        
        results = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="60d")
                
                if df.empty or len(df) < 20:
                    continue
                    
                close = df['Close']
                volume = df['Volume']
                
                pct_change_close = close.pct_change()
                pct_change_vol = volume.pct_change()
                
                delta = close.diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
                rsi = 100 - (100 / (1 + rs))
                
                pct_change_rsi = rsi.pct_change()
                
                last_5_dates = df.index[-5:]
                for date in last_5_dates:
                    results.append({
                        "コード": ticker.replace(".T", ""),
                        "日付": date.strftime("%Y-%m-%d"),
                        "株価変化(%)": round(pct_change_close.loc[date] * 100, 2) if pd.notna(pct_change_close.loc[date]) else None,
                        "RSI変化(%)": round(pct_change_rsi.loc[date] * 100, 2) if pd.notna(pct_change_rsi.loc[date]) else None,
                        "出来高変化(%)": round(pct_change_vol.loc[date] * 100, 2) if pd.notna(pct_change_vol.loc[date]) else None
                    })
            except Exception:
                pass

        if results:
            res_df = pd.DataFrame(results).set_index(['コード', '日付'])
            st.success("分析が完了しました！")
            st.dataframe(res_df, use_container_width=True)
            
            # CSVダウンロードボタン
            csv = res_df.to_csv().encode('utf-8-sig')
            st.download_button(
                label="CSVでダウンロード",
                data=csv,
                file_name='technical_data.csv',
                mime='text/csv',
            )
        else:
            st.error("有効なデータが取得できませんでした。")
