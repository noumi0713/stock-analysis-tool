￥# ... existing code ...
                "RSI": round(current_rsi, 1),
                "判定": status
            })
        except Exception as e:
            continue
    return pd.DataFrame(results)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 強気銘柄スクリーナー", 
    "✨ トレンド自動発掘", 
    "📅 期間データ抽出", 
    "⚙️ リスト管理"
])

current_portfolio = st.session_state["my_portfolio"]
active_tickers = [str(item["コード"]).strip() for item in current_portfolio if str(item["コード"]).strip()]

# データの一括ロード（タブ1とタブ3で共通使用）
with st.spinner('監視リストの市場データを読み込み中...'):
# ... existing code ...
```

```python:テーマ別株式スクリーナー:app.py
# ... existing code ...
            components_df = pd.DataFrame(components_list)
            # RSIの計算に14日以上のデータが必要なため、period="1mo"に変更
            hist_data = yf.download(theme_tickers, period="1mo", interval="1d", group_by="ticker", progress=False)
            
            # リターンの計算
            returns = []
            for t in theme_tickers:
                try:
                    df_t = hist_data[t].dropna().copy()
                    if len(df_t) >= 15:
                        # 指標の計算
                        df_t['RSI'] = calc_rsi(df_t['Close'])
                        df_t['Vol_Change'] = df_t['Volume'].pct_change() * 100
                        
                        start_price = float(df_t['Close'].iloc[-6]) # 5日前(実質5日間)
                        end_price = float(df_t['Close'].iloc[-1])
                        pct_change = ((end_price / start_price) - 1) * 100
                        
                        current_rsi = float(df_t['RSI'].iloc[-1])
                        vol_change_5d = float(df_t['Vol_Change'].iloc[-5:].mean())
                        
                        returns.append({
                            "コード": t.replace(".T", ""), 
                            "5日リターン(%)": round(pct_change, 2),
                            "RSI": round(current_rsi, 1),
                            "5日間出来高変化率(%)": round(vol_change_5d, 1)
                        })
                except:
                    continue
                    
            if returns:
# ... existing code ...
```
```python:テーマ別株式スクリーナー:app.py
# ... existing code ...
                st.markdown("### 📈 上位テーマの構成銘柄")
                top_stocks = merged[merged["テーマ"].isin(top_3_sectors)].sort_values("5日リターン(%)", ascending=False)
                st.dataframe(top_stocks, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📅 期間指定データ抽出")
    st.write("監視リスト（リスト管理タブ）に登録されている全銘柄の指定期間のデータをCSV形式で抽出できます。")
    
    today = datetime.date.today()
# ... existing code ...
```

```python:テーマ別株式スクリーナー:app.py
# ... existing code ...
                        csv = df_range_view.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="💾 抽出結果をCSVで保存",
                            data=csv,
                            file_name=f"stock_data_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("指定期間のデータが見つかりませんでした。")

with tab4:
    st.subheader("⚙️ 監視リスト管理")
    st.write("スクリーニング対象となる銘柄リストをここで直接編集できます。（行の追加、削除、エクセルからのコピペも可能です）")
# ... existing code ...
```

不要になったYahoo Financeからのランキングスクレイピング関数（`fetch_market_ranking_from_web`）も上記の置換で削除され、コード全体が軽量化されています。
