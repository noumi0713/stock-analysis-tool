import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from io import StringIO

# === 指定の16テーマ監視対象辞書（各10銘柄） ===
TARGET_THEME_DICT = {
    "半導体": [("8035", "東京エレクトロン"), ("6857", "アドバンテスト"), ("6920", "レーザーテック"), ("7735", "SCREEN"), ("6723", "ルネサス"), ("6963", "ローム"), ("6146", "ディスコ"), ("6890", "フェローテック"), ("4063", "信越化学工業"), ("7731", "ニコン")],
    "AI関連": [("9984", "ソフトバンクG"), ("3993", "PKSHA"), ("6526", "ソシオネクスト"), ("4736", "日本ラッド"), ("5586", "Laboro.AI"), ("4488", "AI inside"), ("4382", "HEROZ"), ("5132", "pluszero"), ("4071", "プラスアルファ"), ("9432", "NTT")],
    "フィジカルAi": [("6954", "ファナック"), ("6506", "安川電機"), ("6324", "ハーモニック"), ("6268", "ナブテスコ"), ("6273", "SMC"), ("6383", "ダイフク"), ("6103", "オークマ"), ("6481", "THK"), ("7779", "CYBERDYNE"), ("6301", "コマツ")],
    "レアアース": [("5713", "住友金属鉱山"), ("5711", "三菱マテリアル"), ("5714", "DOWA"), ("5706", "三井金属"), ("5802", "住友電気工業"), ("5724", "アサカ理研"), ("4004", "レゾナック"), ("4099", "四国化成"), ("5715", "古河機械金属"), ("5726", "大阪チタニウム")],
    "宇宙": [("7011", "三菱重工業"), ("9348", "ispace"), ("5595", "QPS研究所"), ("7013", "IHI"), ("9412", "スカパーJSAT"), ("186A", "アストロスケール"), ("6503", "三菱電機"), ("6701", "NEC"), ("7721", "東京計器"), ("3402", "東レ")],
    "データセンター": [("9432", "NTT"), ("6501", "日立製作所"), ("1951", "エクシオG"), ("6702", "富士通"), ("6504", "富士電機"), ("1417", "ミライト・ワン"), ("1932", "きんでん"), ("8058", "三菱商事"), ("3778", "さくらインターネット"), ("6988", "日東電工")],
    "ドローン": [("6232", "ACSL"), ("278A", "テラドローン"), ("6052", "ブルーイノベーション"), ("7272", "ヤマハ発動機"), ("6594", "ニデック"), ("7732", "トプコン"), ("2303", "ドーン"), ("3687", "フィックスターズ"), ("9433", "KDDI"), ("7012", "川崎重工業")],
    "防衛": [("7012", "川崎重工業"), ("7721", "東京計器"), ("6208", "石川製作所"), ("7011", "三菱重工業"), ("7013", "IHI"), ("6946", "日本アビオニクス"), ("6703", "沖電気工業"), ("5631", "日本製鋼所"), ("6503", "三菱電機"), ("4274", "細谷火工")],
    "銀行": [("8306", "三菱UFJ"), ("8316", "三井住友"), ("8411", "みずほ"), ("8308", "りそなHD"), ("8309", "三井住友トラスト"), ("7182", "ゆうちょ銀行"), ("8331", "千葉銀行"), ("5831", "静岡FG"), ("8354", "ふくおかFG"), ("7167", "めぶきFG")],
    "光デバイス": [("6965", "浜松ホトニクス"), ("6777", "santec"), ("5803", "フジクラ"), ("6618", "大泉製作所"), ("5802", "住友電気工業"), ("5801", "古河電気工業"), ("6971", "京セラ"), ("7731", "ニコン"), ("6701", "NEC"), ("6988", "日東電工")],
    "蓄電池": [("6752", "パナソニックHD"), ("6762", "TDK"), ("6981", "村田製作所"), ("6504", "富士電機"), ("6674", "GSユアサ"), ("6810", "マクセル"), ("4118", "カネカ"), ("4098", "チタン工業"), ("5711", "三菱マテリアル"), ("6955", "FDK")],
    "量子コンピュータ": [("6701", "NEC"), ("6702", "富士通"), ("3687", "フィックスターズ"), ("6501", "日立製作所"), ("9432", "NTT"), ("6503", "三菱電機"), ("6971", "京セラ"), ("4704", "トレンドマイクロ"), ("6758", "ソニーG"), ("4063", "信越化学工業")],
    "ペロブスカイト太陽光電池": [("4204", "積水化学工業"), ("4118", "カネカ"), ("5020", "ENEOS"), ("4369", "トリケミカル"), ("3407", "旭化成"), ("6752", "パナソニックHD"), ("6504", "富士電機"), ("6988", "日東電工"), ("4188", "三菱ケミカルG"), ("7911", "TOPPAN")],
    "商社": [("8058", "三菱商事"), ("8031", "三井物産"), ("8001", "伊藤忠商事"), ("8053", "住友商事"), ("8002", "丸紅"), ("8015", "豊田通商"), ("2768", "双日"), ("8078", "阪和興業"), ("8012", "長瀬産業"), ("8020", "兼松")],
    "保険業": [("8766", "東京海上"), ("8725", "MS&AD"), ("8630", "SOMPO"), ("8750", "第一生命HD"), ("8795", "T&D HD"), ("7181", "かんぽ生命保険"), ("7164", "全国保証"), ("7148", "FPG"), ("8604", "野村HD"), ("8601", "大和証券G")],
    "非鉄金属": [("5713", "住友金属鉱山"), ("5726", "大阪チタニウム"), ("5706", "三井金属"), ("5711", "三菱マテリアル"), ("5802", "住友電気工業"), ("5803", "フジクラ"), ("5801", "古河電気工業"), ("5714", "DOWA"), ("5715", "古河機械金属"), ("5857", "ARE HD")]
}

st.set_page_config(page_title="テーマ別株式スクリーナー", layout="wide", page_icon="📈")
st.title("📈 戦略的テーマ監視ダッシュボード")

# 初期ポートフォリオの一括生成
INITIAL_PORTFOLIO = []
for theme, stocks in TARGET_THEME_DICT.items():
    for code, name in stocks:
        INITIAL_PORTFOLIO.append({"テーマ": theme, "コード": code, "銘柄名": name})

# セッション初期化（リストが破損・不足している場合は初期化して160銘柄をロード）
if "my_portfolio" not in st.session_state or len(st.session_state["my_portfolio"]) < 150:
    st.session_state["my_portfolio"] = INITIAL_PORTFOLIO
    st.session_state["target_themes_loaded"] = True

@st.cache_data(ttl=600)
def fetch_data(tickers):
    if not tickers: return None
    # yfinance用のティッカーに変換 (例: 8035 -> 8035.T)
    yf_tickers = [f"{t}.T" for t in tickers]
    return yf.download(yf_tickers, period="6mo", interval="1d", group_by="ticker", threads=True, progress=False)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stocks(data, portfolio_list):
    results = []
    # DataFrameの列がマルチインデックスの場合は整形処理
    is_multiindex = isinstance(data.columns, pd.MultiIndex)
    
    for item in portfolio_list:
        code = str(item["コード"]).strip()
        t = f"{code}.T"
        
        try:
            # 1銘柄だけ取得した場合は構造が異なるための対策
            if len(portfolio_list) == 1 or not is_multiindex:
                df = data.dropna()
            else:
                if t not in data.columns.levels[0]:
                    continue
                df = data[t].dropna()
                
            if len(df) < 25: continue
            
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA25'] = df['Close'].rolling(25).mean()
            df['MA75'] = df['Close'].rolling(75).mean()
            df['RSI'] = calc_rsi(df['Close'])
            
            c = df.iloc[-1]
            p = df.iloc[-2]
            
            current_close = float(c['Close'].iloc[0] if isinstance(c['Close'], pd.Series) else c['Close'])
            prev_close = float(p['Close'].iloc[0] if isinstance(p['Close'], pd.Series) else p['Close'])
            current_ma5 = float(c['MA5'].iloc[0] if isinstance(c['MA5'], pd.Series) else c['MA5'])
            prev_ma5 = float(p['MA5'].iloc[0] if isinstance(p['MA5'], pd.Series) else p['MA5'])
            current_ma25 = float(c['MA25'].iloc[0] if isinstance(c['MA25'], pd.Series) else c['MA25'])
            prev_ma25 = float(p['MA25'].iloc[0] if isinstance(p['MA25'], pd.Series) else p['MA25'])
            current_ma75 = float(c['MA75'].iloc[0] if isinstance(c['MA75'], pd.Series) else c['MA75'])
            current_rsi = float(c['RSI'].iloc[0] if isinstance(c['RSI'], pd.Series) else c['RSI'])
            
            dod = ((current_close / prev_close) - 1) * 100
            
            status = "横ばい"
            if current_close > current_ma5 > current_ma25 > current_ma75:
                if current_ma25 > prev_ma25:
                    status = "🌟 パーフェクトオーダー"
            elif current_close > current_ma5 > prev_ma5:
                status = "📈 5日線上向き"

            results.append({
                "テーマ": item.get("テーマ", "未分類"),
                "コード": code,
                "銘柄名": item.get("銘柄名", ""),
                "現在値": round(current_close, 1),
                "前日比": dod,
                "RSI": round(current_rsi, 1),
                "判定": status
            })
        except Exception as e:
            continue
    return pd.DataFrame(results)

@st.cache_data(ttl=1800)
def fetch_market_ranking_from_web(ranking_type="gainers"):
    """Yahoo Financeからランキングデータをスクレイピングして取得"""
    if ranking_type == "gainers":
        url = "https://finance.yahoo.co.jp/ranking/up?market=all&term=daily"
    else:
        url = "https://finance.yahoo.co.jp/ranking/volume?market=all&term=daily"
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html_data = StringIO(response.text)
        dfs = pd.read_html(html_data)
        
        if dfs:
            df = dfs[0]
            if len(df.columns) > 1:
                return df.head(50) # 上位50件を返す
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 強気銘柄スクリーナー", 
    "✨ トレンド自動発掘", 
    "🏆 本日の市場ランキング", 
    "📅 期間データ抽出", 
    "⚙️ リスト管理"
])

current_portfolio = st.session_state["my_portfolio"]
active_tickers = [str(item["コード"]).strip() for item in current_portfolio if str(item["コード"]).strip()]

# データの一括ロード（タブ1とタブ4で共通使用）
with st.spinner('監視リストの市場データを読み込み中...'):
    raw_data = fetch_data(active_tickers)
    analysis_df = pd.DataFrame()
    if raw_data is not None and not raw_data.empty:
        analysis_df = analyze_stocks(raw_data, current_portfolio)

with tab1:
    st.subheader("🔥 テクニカル強気シグナル点灯銘柄")
    if analysis_df.empty:
        st.warning("現在リストに銘柄がないか、データの取得に失敗しました。「リスト管理」タブを確認してください。")
    else:
        col1, col2 = st.columns(2)
        po_stocks = analysis_df[analysis_df["判定"] == "🌟 パーフェクトオーダー"].drop_duplicates(subset="コード")
        ma5_stocks = analysis_df[analysis_df["判定"] == "📈 5日線上向き"].drop_duplicates(subset="コード")
        
        with col1:
            st.success(f"🌟 パーフェクトオーダー中 ({len(po_stocks)}銘柄)")
            st.write("短期・中期・長期の移動平均線がすべて上向きの強い上昇トレンド")
            st.dataframe(po_stocks[["コード", "銘柄名", "テーマ", "現在値", "前日比", "RSI"]].sort_values("前日比", ascending=False), hide_index=True)
            
        with col2:
            st.info(f"📈 短期上昇傾向 (5日線上向き) ({len(ma5_stocks)}銘柄)")
            st.write("株価が5日移動平均線を上回り、短期的な反発・上昇が期待できる銘柄")
            st.dataframe(ma5_stocks[["コード", "銘柄名", "テーマ", "現在値", "前日比", "RSI"]].sort_values("前日比", ascending=False), hide_index=True)

with tab2:
    st.subheader("✨ テーマ別 トレンド自動発掘")
    st.write("あらかじめ定義された16テーマ（160銘柄）の中から、過去5日間の株価パフォーマンスを集計し、現在最も勢いのあるテーマ群を自動発掘します。")
    
    if st.button("🚀 最新のテーマ別トレンドを分析する", type="primary"):
        with st.spinner("16テーマの市場データを解析中..."):
            theme_tickers = []
            components_list = []
            for theme, stocks in TARGET_THEME_DICT.items():
                for code, name in stocks:
                    theme_tickers.append(f"{code}.T")
                    components_list.append({"テーマ": theme, "コード": code, "銘柄名": name})
            
            components_df = pd.DataFrame(components_list)
            hist_data = yf.download(theme_tickers, period="7d", interval="1d", group_by="ticker", progress=False)
            
            # リターンの計算
            returns = []
            for t in theme_tickers:
                try:
                    df_t = hist_data[t].dropna()
                    if len(df_t) >= 5:
                        start_price = float(df_t['Close'].iloc[-5])
                        end_price = float(df_t['Close'].iloc[-1])
                        pct_change = ((end_price / start_price) - 1) * 100
                        returns.append({"コード": t.replace(".T", ""), "5日リターン(%)": pct_change})
                except:
                    continue
                    
            if returns:
                ret_df = pd.DataFrame(returns)
                merged = pd.merge(components_df, ret_df, on="コード")
                
                # テーマごとに平均リターンを算出
                sector_perf = merged.groupby("テーマ")["5日リターン(%)"].mean().reset_index()
                sector_perf = sector_perf.sort_values("5日リターン(%)", ascending=False)
                
                st.success("分析完了！直近5日間で最も上昇しているテーマTOP 3はこちらです：")
                top_3_sectors = sector_perf.head(3)["テーマ"].tolist()
                
                col_t1, col_t2, col_t3 = st.columns(3)
                for i, col in enumerate([col_t1, col_t2, col_t3]):
                    if i < len(top_3_sectors):
                        theme_name = top_3_sectors[i]
                        perf = sector_perf.iloc[i]["5日リターン(%)"]
                        with col:
                            st.metric(label=f"第{i+1}位: {theme_name}", value=f"{perf:+.2f}%")
                            
                st.write("---")
                st.markdown("### 📈 上位テーマの構成銘柄")
                top_stocks = merged[merged["テーマ"].isin(top_3_sectors)].sort_values("5日リターン(%)", ascending=False)
                st.dataframe(top_stocks, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🏆 本日の市場ランキング (Yahoo Finance)")
    st.write("全市場を対象とした「値上がり率」と「出来高」のトップ50ランキングです。")
    
    ranking_type = st.radio("ランキングの種類を選択", ["値上がり率ランキング", "出来高ランキング"], horizontal=True)
    
    if st.button("📊 ランキングを取得する"):
        with st.spinner("Webから最新ランキングを取得中..."):
            rtype = "gainers" if ranking_type == "値上がり率ランキング" else "volume"
            ranking_df = fetch_market_ranking_from_web(rtype)
            
            if not ranking_df.empty:
                st.success("データの取得に成功しました。")
                st.dataframe(ranking_df, use_container_width=True)
            else:
                st.error("ランキングの取得に失敗しました。Yahoo Financeの構造が変更されたか、アクセス制限がかかっている可能性があります。")

with tab4:
    st.subheader("📅 期間指定データ抽出")
    st.write("監視リスト（リスト管理タブ）に登録されている全銘柄の指定期間のデータをCSV形式で抽出できます。")
    
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=30)
    date_range = st.date_input("期間を選択", [default_start, today])

    if len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        if st.button("📊 データを抽出"):
            if not active_tickers:
                st.warning("リストに銘柄がありません。")
            else:
                with st.spinner("期間データを集計中..."):
                    all_period_data = []
                    
                    is_multi = isinstance(raw_data.columns, pd.MultiIndex)
                    
                    for item in current_portfolio:
                        code = str(item["コード"]).strip()
                        t = f"{code}.T"
                        
                        try:
                            if is_multi:
                                if t not in raw_data.columns.levels[0]: continue
                                df_hist = raw_data[t].dropna()
                            else:
                                df_hist = raw_data.dropna()
                                
                            df_hist['RSI'] = calc_rsi(df_hist['Close'])
                            
                            mask = (df_hist.index >= start_ts) & (df_hist.index <= end_ts)
                            df_filtered = df_hist.loc[mask]
                            
                            if not df_filtered.empty:
                                for idx, row in df_filtered.iterrows():
                                    close_val = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'])
                                    vol_val = int(row['Volume'].iloc[0] if isinstance(row['Volume'], pd.Series) else row['Volume'])
                                    rsi_val = float(row['RSI'].iloc[0] if isinstance(row['RSI'], pd.Series) else row['RSI'])
                                    
                                    all_period_data.append({
                                        "日付": idx.date(),
                                        "テーマ": item.get("テーマ", ""),
                                        "コード": code,
                                        "銘柄名": item.get("銘柄名", ""),
                                        "引値": round(close_val, 1),
                                        "出来高": vol_val,
                                        "RSI": round(rsi_val, 1) if not np.isnan(rsi_val) else "算出中"
                                    })
                        except Exception as e:
                            continue
                    
                    if all_period_data:
                        df_range_view = pd.DataFrame(all_period_data)
                        df_range_view = df_range_view.sort_values(["日付", "テーマ", "コード"], ascending=[False, True, True])
                        
                        st.success(f"{len(df_range_view)} 件のデータを抽出しました。")
                        st.dataframe(df_range_view, use_container_width=True, hide_index=True)
                        
                        csv = df_range_view.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="💾 抽出結果をCSVで保存",
                            data=csv,
                            file_name=f"stock_data_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("指定期間のデータが見つかりませんでした。")

with tab5:
    st.subheader("⚙️ 監視リスト管理")
    st.write("スクリーニング対象となる銘柄リストをここで直接編集できます。（行の追加、削除、エクセルからのコピペも可能です）")
    
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state["my_portfolio"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "テーマ": st.column_config.TextColumn("テーマ (任意)"),
            "コード": st.column_config.TextColumn("銘柄コード (必須)", required=True),
            "銘柄名": st.column_config.TextColumn("銘柄名 (任意)")
        }
    )
    
    if st.button("💾 変更を保存して適用", type="primary"):
        # 空白行やコードが空の行を除外
        cleaned_list = []
        for _, row in edited_df.iterrows():
            if pd.notna(row["コード"]) and str(row["コード"]).strip() != "":
                cleaned_list.append({
                    "テーマ": str(row["テーマ"]) if pd.notna(row["テーマ"]) else "未分類",
                    "コード": str(row["コード"]).strip(),
                    "銘柄名": str(row["銘柄名"]) if pd.notna(row["銘柄名"]) else ""
                })
        
        st.session_state["my_portfolio"] = cleaned_list
        st.success("リストを更新しました！アプリが新しいリストに基づいて再計算します。")
        st.rerun()

    st.write("---")
    if st.button("⚠️ リストをデフォルト（16テーマ・160銘柄）にリセットする"):
        st.session_state["my_portfolio"] = INITIAL_PORTFOLIO
        st.success("初期化しました。")
        st.rerun()
