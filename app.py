import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="テーマ別株式スクリーナー", layout="wide")
st.title("📈 戦略的テーマ監視ダッシュボード")

# ==========================================
# 1. 銘柄データ（厳選16テーマ × 各10銘柄）
# ==========================================
THEME_DICT = {
    "半導体": [
        {"コード": "8035", "銘柄名": "東京エレクトロン"}, {"コード": "6146", "銘柄名": "ディスコ"},
        {"コード": "6857", "銘柄名": "アドバンテスト"}, {"コード": "6920", "銘柄名": "レーザーテック"},
        {"コード": "7735", "銘柄名": "SCREEN"}, {"コード": "6723", "銘柄名": "ルネサスエレクトロニクス"},
        {"コード": "6963", "銘柄名": "ローム"}, {"コード": "6707", "銘柄名": "サンケン電気"},
        {"コード": "6871", "銘柄名": "日本マイクロニクス"}, {"コード": "3436", "銘柄名": "SUMCO"}
    ],
    "AI関連": [
        {"コード": "9984", "銘柄名": "ソフトバンクG"}, {"コード": "3993", "銘柄名": "PKSHA Technology"},
        {"コード": "4443", "銘柄名": "Sansan"}, {"コード": "6526", "銘柄名": "ソシオネクスト"},
        {"コード": "4056", "銘柄名": "ニューラルG"}, {"コード": "4382", "銘柄名": "HEROZ"},
        {"コード": "4488", "銘柄名": "AI inside"}, {"コード": "5595", "銘柄名": "QPS研究所"},
        {"コード": "4736", "銘柄名": "日本ラッド"}, {"コード": "9432", "銘柄名": "NTT"}
    ],
    "フィジカルAI": [
        {"コード": "6506", "銘柄名": "安川電機"}, {"コード": "6954", "銘柄名": "ファナック"},
        {"コード": "6268", "銘柄名": "ナブテスコ"}, {"コード": "6273", "銘柄名": "SMC"},
        {"コード": "6324", "銘柄名": "ハーモニック・ドライブ"}, {"コード": "7779", "銘柄名": "CYBERDYNE"},
        {"コード": "6367", "銘柄名": "ダイキン工業"}, {"コード": "6481", "銘柄名": "THK"},
        {"コード": "6201", "銘柄名": "豊田自動織機"}, {"コード": "7203", "銘柄名": "トヨタ自動車"}
    ],
    "レアアース": [
        {"コード": "5711", "銘柄名": "三菱マテリアル"}, {"コード": "5713", "銘柄名": "住友金属鉱山"},
        {"コード": "5714", "銘柄名": "DOWA"}, {"コード": "5706", "銘柄名": "三井金属"},
        {"コード": "5715", "銘柄名": "古河機械金属"}, {"コード": "5802", "銘柄名": "住友電工"},
        {"コード": "5803", "銘柄名": "フジクラ"}, {"コード": "3315", "銘柄名": "日本コークス工業"},
        {"コード": "4043", "銘柄名": "トクヤマ"}, {"コード": "4004", "銘柄名": "レゾナックHD"}
    ],
    "宇宙": [
        {"コード": "7011", "銘柄名": "三菱重工業"}, {"コード": "7013", "銘柄名": "IHI"},
        {"コード": "9412", "銘柄名": "スカパーJSAT"}, {"コード": "464A", "銘柄名": "QPS研究所"},
        {"コード": "186A", "銘柄名": "アストロスケール"}, {"コード": "9348", "銘柄名": "ispace"},
        {"コード": "6965", "銘柄名": "浜松ホトニクス"}, {"コード": "7721", "銘柄名": "東京計器"},
        {"コード": "6701", "銘柄名": "NEC"}, {"コード": "6503", "銘柄名": "三菱電機"}
    ],
    "データセンター": [
        {"コード": "3778", "銘柄名": "さくらインターネット"}, {"コード": "3836", "銘柄名": "アバントG"},
        {"コード": "9639", "銘柄名": "三協フロンテア"}, {"コード": "1932", "銘柄名": "きんでん"},
        {"コード": "1951", "銘柄名": "エクシオG"}, {"コード": "6988", "銘柄名": "日東電工"},
        {"コード": "6501", "銘柄名": "日立製作所"}, {"コード": "6702", "銘柄名": "富士通"},
        {"コード": "6504", "銘柄名": "富士電機"}, {"コード": "6971", "銘柄名": "京セラ"}
    ],
    "ドローン": [
        {"コード": "6232", "銘柄名": "ACSL"}, {"コード": "278A", "銘柄名": "テラドローン"},
        {"コード": "6052", "銘柄名": "ブルーイノベーション"}, {"コード": "2303", "銘柄名": "ドーン"},
        {"コード": "7272", "銘柄名": "ヤマハ発動機"}, {"コード": "6594", "銘柄名": "ニデック"},
        {"コード": "7732", "銘柄名": "トプコン"}, {"コード": "3687", "銘柄名": "フィックスターズ"},
        {"コード": "6701", "銘柄名": "NEC"}, {"コード": "9433", "銘柄名": "KDDI"}
    ],
    "防衛": [
        {"コード": "7011", "銘柄名": "三菱重工業"}, {"コード": "7012", "銘柄名": "川崎重工業"},
        {"コード": "7013", "銘柄名": "IHI"}, {"コード": "6946", "銘柄名": "日本アビオニクス"},
        {"コード": "7721", "銘柄名": "東京計器"}, {"コード": "6203", "銘柄名": "豊和工業"},
        {"コード": "6208", "銘柄名": "石川製作所"}, {"コード": "4274", "銘柄名": "細谷火工"},
        {"コード": "3105", "銘柄名": "日清紡HD"}, {"コード": "6703", "銘柄名": "OKI"}
    ],
    "銀行": [
        {"コード": "8306", "銘柄名": "三菱UFJ FG"}, {"コード": "8316", "銘柄名": "三井住友FG"},
        {"コード": "8411", "銘柄名": "みずほFG"}, {"コード": "8308", "銘柄名": "りそなHD"},
        {"コード": "8309", "銘柄名": "三井住友トラスト"}, {"コード": "7182", "銘柄名": "ゆうちょ銀行"},
        {"コード": "8331", "銘柄名": "千葉銀行"}, {"コード": "8354", "銘柄名": "ふくおかFG"},
        {"コード": "8382", "銘柄名": "中国銀行"}, {"コード": "5831", "銘柄名": "しずおかFG"}
    ],
    "光デバイス": [
        {"コード": "6632", "銘柄名": "JVCケンウッド"}, {"コード": "6758", "銘柄名": "ソニーG"},
        {"コード": "6965", "銘柄名": "浜松ホトニクス"}, {"コード": "7753", "銘柄名": "シード"},
        {"コード": "7733", "銘柄名": "オリンパス"}, {"コード": "7731", "銘柄名": "ニコン"},
        {"コード": "6841", "銘柄名": "横河電機"}, {"コード": "6845", "銘柄名": "アズビル"},
        {"コード": "6929", "銘柄名": "日本セラミック"}, {"コード": "6902", "銘柄名": "デンソー"}
    ],
    "蓄電池": [
        {"コード": "6752", "銘柄名": "パナソニック"}, {"コード": "6762", "銘柄名": "TDK"},
        {"コード": "6981", "銘柄名": "村田製作所"}, {"コード": "4118", "銘柄名": "カネカ"},
        {"コード": "4204", "銘柄名": "積水化学"}, {"コード": "6810", "銘柄名": "マクセル"},
        {"コード": "6617", "銘柄名": "東光高岳"}, {"コード": "4098", "銘柄名": "チタン工業"},
        {"コード": "4107", "銘柄名": "伊勢化学工業"}, {"コード": "3407", "銘柄名": "旭化成"}
    ],
    "量子コンピュータ": [
        {"コード": "6702", "銘柄名": "富士通"}, {"コード": "6701", "銘柄名": "NEC"},
        {"コード": "6501", "銘柄名": "日立製作所"}, {"コード": "6503", "銘柄名": "三菱電機"},
        {"コード": "9432", "銘柄名": "NTT"}, {"コード": "4704", "銘柄名": "トレンドマイクロ"},
        {"コード": "6758", "銘柄名": "ソニーG"}, {"コード": "6971", "銘柄名": "京セラ"},
        {"コード": "8053", "銘柄名": "住友商事"}, {"コード": "4063", "銘柄名": "信越化学工業"}
    ],
    "ペロブスカイト太陽光電池": [
        {"コード": "4118", "銘柄名": "カネカ"}, {"コード": "4204", "銘柄名": "積水化学"},
        {"コード": "6752", "銘柄名": "パナソニック"}, {"コード": "3402", "銘柄名": "東レ"},
        {"コード": "3407", "銘柄名": "旭化成"}, {"コード": "4005", "銘柄名": "住友化学"},
        {"コード": "4183", "銘柄名": "三井化学"}, {"コード": "4188", "銘柄名": "三菱ケミカルG"},
        {"コード": "8088", "銘柄名": "岩谷産業"}, {"コード": "5020", "銘柄名": "ENEOS"}
    ],
    "商社": [
        {"コード": "8058", "銘柄名": "三菱商事"}, {"コード": "8031", "銘柄名": "三井物産"},
        {"コード": "8001", "銘柄名": "伊藤忠商事"}, {"コード": "8002", "銘柄名": "丸紅"},
        {"コード": "8053", "銘柄名": "住友商事"}, {"コード": "2768", "銘柄名": "双日"},
        {"コード": "8015", "銘柄名": "豊田通商"}, {"コード": "8078", "銘柄名": "阪和興業"},
        {"コード": "8020", "銘柄名": "兼松"}, {"コード": "8084", "銘柄名": "菱電商事"}
    ],
    "保険業": [
        {"コード": "8766", "銘柄名": "東京海上HD"}, {"コード": "8725", "銘柄名": "MS&AD"},
        {"コード": "8630", "銘柄名": "SOMPO HD"}, {"コード": "8750", "銘柄名": "第一生命HD"},
        {"コード": "8795", "銘柄名": "T&D HD"}, {"コード": "8714", "銘柄名": "池田泉州HD"},
        {"コード": "8732", "銘柄名": "マネパG"}, {"コード": "7166", "銘柄名": "じもとHD"},
        {"コード": "8713", "銘柄名": "フィデアHD"}, {"コード": "8704", "銘柄名": "トレイダーズHD"}
    ],
    "非鉄金属": [
        {"コード": "5713", "銘柄名": "住友金属鉱山"}, {"コード": "5711", "銘柄名": "三菱マテリアル"},
        {"コード": "5714", "銘柄名": "DOWA"}, {"コード": "5706", "銘柄名": "三井金属"},
        {"コード": "5715", "銘柄名": "古河機械金属"}, {"コード": "5802", "銘柄名": "住友電工"},
        {"コード": "5803", "銘柄名": "フジクラ"}, {"コード": "5801", "銘柄名": "古河電工"},
        {"コード": "5726", "銘柄名": "大阪チタニウム"}, {"コード": "5727", "銘柄名": "東邦チタニウム"}
    ]
}

# --- セッションステート（ポートフォリオ）の初期化 ---
# 初回起動時、または古いリストの場合は新しい160銘柄リストで上書き
if "my_portfolio" not in st.session_state or len(st.session_state["my_portfolio"]) < 100:
    default_list = []
    for theme, stocks in THEME_DICT.items():
        for s in stocks:
            default_list.append({"テーマ": theme, "コード": s["コード"], "銘柄名": s["銘柄名"]})
    st.session_state["my_portfolio"] = default_list

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
