import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta

# ページ設定
st.set_page_config(page_title="新・トレンドフォロー型モメンタムスクリーナー", layout="wide", page_icon="📈")

# タイトルとルールの表示
st.title("📈 新・トレンドフォロー型モメンタムスクリーナー")
st.markdown("""
**【ご自身の相場判断に基づく完全版投資ルール】**
* **絶対条件①**: 東証プライム市場の売買代金ランキング上位50位以内（大口資金の流入） ※ご自身でリストをご用意ください。
* **絶対条件②**: 株価がSMA25の上にあり、かつSMA25が上向き（上昇トレンド）
* **ストライクゾーン**: SMA5からの乖離率が「-3% 〜 +3%以内」（高値掴み排除）
""")

st.sidebar.header("⚙️ 対象銘柄の入力")

st.sidebar.markdown("証券会社のツール等で抽出した**「売買代金上位銘柄」**のコードを貼り付けてください。")
custom_tickers = st.sidebar.text_area(
    "銘柄コードを入力（カンマ、スペース、または改行区切り）\n例: 7203, 8306, 9984", 
    "7203\n8306\n9984\n6920\n8035"
)

import re
# 入力されたコードを整形（カンマ、スペース、改行で分割し、.Tを付与）
raw_tickers = re.split(r'[\n, ]+', custom_tickers)
tickers_to_fetch = []
for t in raw_tickers:
    t = t.strip()
    if t:
        if not t.upper().endswith('.T'):
            t += '.T'
        else:
            t = t.upper()
        if t not in tickers_to_fetch:
            tickers_to_fetch.append(t)

if st.sidebar.button("🚀 データ取得＆スクリーニング実行"):
    if not tickers_to_fetch:
        st.warning("対象となる銘柄がありません。")
    else:
        # 進捗状況を視覚化するためのプログレスバーを追加
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        with st.spinner('株価データを取得・計算中です...'):
            try:
                results = []
                all_data_list = []
                total_tickers = len(tickers_to_fetch)
                
                # yfinanceの仕様変更やデータ構造のブレによる読み込みエラーを防ぐため、
                # 1銘柄ずつ独立して安全にデータを取得・パースする方式に変更
                for i, ticker in enumerate(tickers_to_fetch):
                    # プログレスバーの更新
                    progress_text.text(f"データ解析中... {i+1}/{total_tickers} ({ticker})")
                    progress_bar.progress((i + 1) / total_tickers)
                    
                    try:
                        # yfinanceの仕様による当日データ（今日のデータ）の反映遅延を軽減するため、
                        # download() ではなく Ticker().history() メソッドを使用する方式に変更
                        ticker_obj = yf.Ticker(ticker)
                        df = ticker_obj.history(period="6mo")
                        
                        if df is None or df.empty or len(df) < 26:
                            continue
                            
                        acquired_time_str = ""
                        JST = timezone(timedelta(hours=+9), 'JST')
                            
                        # --- 当日（ザラ場中〜15:30大引け）の最新価格を強制反映する確実な補完処理 ---
                        try:
                            # yfinanceの fast_info は不安定なため、
                            # 直近1日分の「1分足データ」を取得して、確実な現在値（直近約定値）を抽出する
                            # 東証の取引時間延長（15:30引け）にも完全対応し、最新時刻のデータを取得
                            df_min = ticker_obj.history(period="1d", interval="1m")
                            
                            if df_min is not None and not df_min.empty:
                                latest_price = float(df_min['Close'].iloc[-1])
                                latest_datetime = df_min.index[-1]
                                
                                # JST(日本時間)で日付と時刻を処理
                                if latest_datetime.tz is None:
                                    latest_dt_jst = pd.Timestamp(latest_datetime).tz_localize(JST)
                                else:
                                    latest_dt_jst = latest_datetime.tz_convert(JST)
                                    
                                acquired_time_str = latest_dt_jst.strftime('%m/%d %H:%M')
                                latest_date = latest_dt_jst.date()
                                    
                                if df.index[-1].tz is not None:
                                    last_df_date = df.index[-1].tz_convert(JST).date()
                                else:
                                    last_df_date = pd.Timestamp(df.index[-1]).tz_localize(JST).date()
                                
                                # 1分足から取得した日付が、日足の最終日より新しい場合（＝今日の日足がまだ無い場合）
                                if latest_date > last_df_date:
                                    # dfのタイムゾーンに合わせてインデックスを作成
                                    if df.index.tz is not None:
                                        new_index = pd.Timestamp(latest_datetime).tz_convert(df.index.tz)
                                    else:
                                        new_index = pd.Timestamp(latest_datetime).tz_localize(None)
                                        
                                    new_row = pd.DataFrame({'Close': [latest_price]}, index=[new_index])
                                    df = pd.concat([df, new_row])
                                    
                                # 日足データに「今日」の行がすでにある場合、最新価格で上書きする
                                elif latest_date == last_df_date:
                                    df.loc[df.index[-1], 'Close'] = latest_price
                                    
                        except Exception:
                            # 補完処理中にエラーが出た場合は無視して既存のデータ(前日までの足)を使用
                            pass
                        # ----------------------------------------------------
                        
                        close_px = df['Close']
                        
                        # 万が一DataFrame形式で返ってきた場合はSeriesに変換
                        if isinstance(close_px, pd.DataFrame):
                            close_px = close_px.iloc[:, 0]
                            
                        # 欠損値(NaN)を前日の値で補完し、データ抜けによるエラーを防止
                        close_px = close_px.ffill().dropna()
                        
                        if len(close_px) < 26:
                            continue
                        
                        # 取得できたデータの「最新日時」を記録（1分足が取れなかった場合は日足の最終日）
                        if not acquired_time_str:
                            if close_px.index[-1].tz is not None:
                                acquired_time_str = close_px.index[-1].tz_convert(JST).strftime('%m/%d')
                            else:
                                acquired_time_str = close_px.index[-1].strftime('%m/%d')
                        
                        # SMAの計算
                        sma5 = close_px.rolling(window=5).mean()
                        sma25 = close_px.rolling(window=25).mean()
                        
                        # 直近の値を取得（float型に明示的に変換し、型の不整合を防止）
                        current_close = float(close_px.iloc[-1])
                        current_sma5 = float(sma5.iloc[-1])
                        current_sma25 = float(sma25.iloc[-1])
                        prev_sma25 = float(sma25.iloc[-2])
                        
                        # 【第1段階：絶対条件②】2つの移動平均線による「波の定義」
                        cond_sma25_up = current_sma25 > prev_sma25
                        cond_above_sma25 = current_close > current_sma25
                        
                        # 【第1段階：絶対条件③】ストライクゾーンの厳守
                        kairi_sma5 = ((current_close - current_sma5) / current_sma5) * 100
                        cond_kairi = -3.0 <= kairi_sma5 <= 3.0
                        
                        # 全銘柄のデータを記録
                        trend_status = "⭕️上昇トレンド" if cond_sma25_up and cond_above_sma25 else "❌条件未達"
                        strike_zone = "🎯圏内" if cond_kairi else "圏外"
                        
                        ticker_code = ticker.replace(".T", "")
                        
                        ticker_info = {
                            "銘柄コード": ticker_code,
                            "取得日時": acquired_time_str,
                            "株価": round(current_close, 1),
                            "SMA5乖離率(%)": round(kairi_sma5, 2),
                            "SMA5": round(current_sma5, 1),
                            "SMA25": round(current_sma25, 1),
                            "トレンド判定": trend_status,
                            "ストライクゾーン": strike_zone
                        }
                        all_data_list.append(ticker_info)
                        
                        # 全条件を満たした場合のみ結果リストに追加
                        if cond_sma25_up and cond_above_sma25 and cond_kairi:
                            results.append(ticker_info)
                            
                    except Exception as e:
                        # 特定の銘柄でエラーが起きても全体を止めずにスキップ
                        continue

                # 結果の表示
                progress_text.empty() # プログレスバーのテキストを消去
                progress_bar.empty()  # プログレスバー本体を消去
                
                st.success("データの取得と計算が完了しました！")
                
                st.subheader(f"🎯 ストライクゾーン到達銘柄 (条件完全クリア): {len(results)}件")
                
                # 乖離率のカラーリング設定（Pandas Styler）
                def color_kairi(val):
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        color = 'red' if val > 0 else 'blue'
                        return f'color: {color}'
                    return ''

                if results:
                    result_df = pd.DataFrame(results)
                    st.dataframe(
                        result_df.style.map(color_kairi, subset=['SMA5乖離率(%)']) if hasattr(result_df.style, 'map') else result_df.style.applymap(color_kairi, subset=['SMA5乖離率(%)']),
                        use_container_width=True
                    )
                else:
                    st.info("現在、入力された銘柄の中で条件（上昇トレンド ＋ 乖離率±3%以内）を満たす銘柄はありませんでした。無理なエントリーは控えましょう。")

                st.markdown("---")
                st.subheader(f"📊 入力銘柄のテクニカルデータ一覧 ({len(all_data_list)}件)")
                if all_data_list:
                    all_df = pd.DataFrame(all_data_list)
                    st.dataframe(
                        all_df.style.map(color_kairi, subset=['SMA5乖離率(%)']) if hasattr(all_df.style, 'map') else all_df.style.applymap(color_kairi, subset=['SMA5乖離率(%)']),
                        use_container_width=True
                    )

                st.markdown("""
                ---
                ### 💡 次のステップ（資金管理とエグジット）
                    * **[9:30-10:00の確認]** 寄り付きのノイズが消えた後、上記銘柄の乖離率と板・歩み値（大口の資金流入）を確認します。
                    * **[打診買い]** ストライクゾーン内で、まずは **資金の半分** をエントリー。
                    * **[ピラミッディング]** その後、浅い押し目からの反発を確認して **残りの資金** を投入します。
                    * **[命綱の設定]** 約定後、必ず **買値の-8%** または直近サポートに逆指値を設定してください。
                    """)
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

st.markdown("---")
st.caption("※本データはYahoo Financeを利用しており、実際の相場データ（特に寄り付き直後）とはタイムラグや差異が生じる場合があります。最終的な執行判断は証券会社のリアルタイムツールにて売買代金と乖離率をご確認ください。")
