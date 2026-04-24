import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="テーマ別株式スクリーナー", layout="wide")
st.title("📈 戦略的テーマ監視ダッシュボード")

# ==========================================
# 1. 銘柄データ（全31テーマ）
# ==========================================
RAW_STOCK_LIST = [
    "1. AI・半導体", "6857/アドバンテスト 8035/東京エレクトロン 6723/ルネサスエレクトロニクス 6920/レーザーテック 7735/ＳＣＲＥＥＮホールディングス 6963/ローム 6707/サンケン電気 7282/豊田合成 9984/ソフトバンクグループ 6501/日立製作所",
    "2. 造船", "7011/三菱重工業 7012/川崎重工業 7014/名村造船所 7003/三井Ｅ＆Ｓ 7018/内海造船 6302/住友重機械工業 9101/日本郵船 9104/商船三井 9107/川崎汽船 7022/サノヤスホールディングス",
    "3. 量子", "6701/日本電気 6702/富士通 9432/日本電信電話 6501/日立製作所 6503/三菱電機 6971/京セラ 8053/住友商事 4704/トレンドマイクロ 6758/ソニーグループ 4063/信越化学工業",
    "4. 合成生物学・バイオ", "4502/武田薬品工業 4568/第一三共 4471/三洋化成工業 8111/ゴールドウイン 4613/関西ペイント 6028/テクノプロ・ホールディングス 2931/ユーグレナ 4523/エーザイ 4519/中外製薬 4901/富士フイルムホールディングス",
    "5. 航空・宇宙", "7011/三菱重工業 7013/ＩＨＩ 9412/スカパーＪＳＡＴホールディングス 464A/ＱＰＳ研究所 186A/アストロスケールホールディングス 9348/ｉｓｐａｃｅ 3402/東レ 7224/新明和工業 3524/日東製網 6965/浜松ホトニクス 290A/シンスペ",
    "6. デジタル・サイバーセキュリティ", "4704/トレンドマイクロ 3692/ＦＦＲＩセキュリティ 4441/トビラシステムズ 3916/デジタル・インフォメーション・テクノロジー 2326/デジタルアーツ 3040/ソリトンシステムズ 4258/網屋 9433/ＫＤＤＩ 4398/ブロードバンドセキュリティ 4722/フューチャー 6701/日本電気",
    "7. コンテンツ", "6758/ソニーグループ 9404/日本テレビホールディングス 7832/バンダイナムコホールディングス 4751/サイバーエージェント 9468/ＫＡＤＯＫＡＷＡ 7974/任天堂 4816/東映アニメーション 9684/スクウェア・エニックス・ホールディングス 9697/カプコン 3765/ガンホー・オンライン・エンターテイメント",
    "8. フードテック", "3182/オイシックス・ラ・大地 1332/ニッスイ 1333/マルハニチロ 2282/日本ハム 2607/不二製油グループ本社 2802/味の素 2811/カゴメ 2193/クックパッド 2296/伊藤ハム米久ホールディングス 2216/カンロ",
    "9. 資源・エネルギー・GX", "9501/東京電力ホールディングス 5020/ＥＮＥＯＳホールディングス 6752/パナソニックホールディングス 4204/積水化学工業 4107/伊勢化学工業 5711/三菱マテリアル 4118/カネカ 9503/関西電力 1605/ＩＮＰＥＸ 1963/日揮ホールディングス",
    "10. 防災・国土強靭化", "1414/ショーボンドホールディングス 1813/不動テトラ 9621/建設技術研究所 1417/ミライト・ワン 208A/構造計画研究所 8088/岩谷産業 6632/ＪＶＣケンウッド 5285/ヤマックス 1848/富士ピー・エス 1888/若築建設",
    "11. 創薬・先端医療", "4543/テルモ 7733/オリンパス 4519/中外製薬 4507/塩野義製薬 4523/エーザイ 4901/富士フイルムホールディングス 7747/朝日インテック 7701/島津製作所 6869/シスメックス 4527/ロート製薬",
    "12. フュージョンエネルギー", "5803/フジクラ 8801/三井不動産 6971/京セラ 6965/浜松ホトニクス 7011/三菱重工業 6501/日立製作所 6503/三菱電機 6504/富士電機 5802/住友電気工業 7013/ＩＨＩ",
    "13. マテリアル", "4063/信越化学工業 3436/ＳＵＭＣＯ 5713/住友金属鉱山 5726/大阪チタニウムテクノロジーズ 5333/日本碍子 5310/東洋炭素 5302/日本カーボン 5406/神戸製鋼所 5401/日本製鉄 5411/ＪＦＥホールディングス",
    "14. 港湾ロジスティクス", "9301/三菱倉庫 9303/住友倉庫 9364/上組 9302/三井倉庫ホールディングス 9147/ＮＸホールディングス 9107/川崎汽船 9101/日本郵船 9104/商船三井 9304/渋沢倉庫 9358/宇徳",
    "15. 防衛産業", "7011/三菱重工業 7012/川崎重工業 7013/ＩＨＩ 7721/東京計器 6946/日本アビオニクス 6703/沖電気工業 3105/日清紡ホールディングス 6486/イーグル工業 5631/日本製鋼所 8093/極東貿易",
    "16. 情報通信", "9432/日本電信電話 9433/ＫＤＤＩ 9434/ソフトバンク 4755/楽天グループ 9412/スカパーＪＳＡＴホールディングス 5801/古河電気工業 5802/住友電気工業 5803/フジクラ 6701/日本電気 6702/富士通",
    "17. 海洋", "6269/三井海洋開発 1963/日揮ホールディングス 7003/三井Ｅ＆Ｓ 7011/三菱重工業 6834/精工技研 6618/大泉製作所 5802/住友電気工業 6777/ｓａｎｔｅｃ 3648/ＡＧＳ 6340/渋谷工業",
    "18. (対米) 次世代原子力", "6501/日立製作所 7011/三菱重工業 1812/鹿島建設 1802/大林組 1803/清水建設 8058/三菱商事 1833/奥村組 7013/ＩＨＩ 8031/三井物産 8001/伊藤忠商事",
    "19. (対米) 天然ガス・AI電源", "6501/日立製作所 7011/三菱重工業 7013/ＩＨＩ 6503/三菱電機 5803/フジクラ 6762/ＴＤＫ 6981/村田製作所 6752/パナソニックホールディングス 9984/ソフトバンクグループ 6701/日本電気",
    "20. (対米) 原油インフラ・備蓄", "5020/ＥＮＥＯＳホールディングス 5019/出光興産 5021/コスモエネルギーホールディングス 1605/ＩＮＰＥＸ 8058/三菱商事 8031/三井物産 8001/伊藤忠商事 8053/住友商事 8002/丸紅 1963/日揮ホールディングス",
    "21. (対米) 先端マテリアル", "8031/三井物産 5711/三菱マテリアル 5802/住友電気工業 3402/東レ 3401/帝人 3407/旭化成 4205/日本ゼオン 4063/信越化学工業 4004/レゾナック・ホールディングス 4208/ＵＢＥ",
    "22. (対米) 重要鉱物資源", "5713/住友金属鉱山 5711/三菱マテリアル 8031/三井物産 8058/三菱商事 8015/豊田通商 5714/ＤＯＷＡホールディングス 5706/三井金属鉱業 5715/古河機械金属 3315/日本コークス工業 8002/丸紅",
    "23. フィジカルAI", "6506/安川電機 6954/ファナック 202A/豆蔵ホールディングス 3132/マクニカホールディングス 6268/ナブテスコ 6273/ＳＭＣ 6324/ハーモニック・ドライブ・システムズ 3741/セック 4425/Ｋｕｄａｎ 7779/サイバーダイン",
    "24. 蓄電池", "6752/パナソニックホールディングス 6762/ＴＤＫ 6981/村田製作所 4204/積水化学工業 4118/カネカ 4107/伊勢化学工業 3407/旭化成 6502/東芝 6810/マクセル 485A/パワーエックス 6617/東光高岳",
    "25. 再エネ", "9519/レノバ 1407/ウエストホールディングス",
    "26. 石炭", "1514/住石ホールディングス 8835/太平洋興発",
    "27. 天然ガス", "1663/Ｋ＆Ｏエナジーグループ 1963/日揮ホールディングス",
    "28. 電力卸", "9513/電源開発 9517/イーレックス",
    "29. 肥料", "4031/片倉コープアグリ 4979/ＯＡＴアグリオ",
    "30. バイオ燃料", "9212/ＧｒｅｅＮＥａｒｔｈＩｎｓｔｉｔｕｔｅ 2931/ユーグレナ",
    "31. ドローン・次世代モビリティ", "278A/テラドローン 6232/ＡＣＳＬ 6052/ブルーイノベーション 7272/ヤマハ発動機 6594/ニデック 7732/トプコン 2303/ドーン 3687/フィックスターズ 6701/日本電気 9433/ＫＤＤＩ"
]

# --- ヘルパー関数 ---
def get_base_tickers():
    t_dict = {}
    cur_theme = "不明"
    for line in RAW_STOCK_LIST:
        if "/" not in line:
            cur_theme = line
        else:
            stocks = line.split()
            for s in stocks:
                if "/" in s:
                    code, name = s.split("/")
                    ticker = f"{code}.T"
                    if ticker not in t_dict:
                        t_dict[ticker] = {"name": name, "themes": [cur_theme]}
                    else:
                        if cur_theme not in t_dict[ticker]["themes"]:
                            t_dict[ticker]["themes"].append(cur_theme)
    return t_dict

# --- データ取得と解析 ---
@st.cache_data(ttl=600)
def fetch_data(tickers):
    if not tickers: return None
    return yf.download(tickers, period="6mo", interval="1d", group_by="ticker", threads=True)

# テクニカル指標計算用の内部関数
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stocks(data, tickers_dict):
    results = []
    ts = list(tickers_dict.keys())
    for t in ts:
        try:
            df = data[t].dropna()
            if len(df) < 25: continue
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA25'] = df['Close'].rolling(25).mean()
            df['MA75'] = df['Close'].rolling(75).mean()
            df['RSI'] = calc_rsi(df['Close'])
            
            c = df.iloc[-1]
            p = df.iloc[-2]
            dod = ((c['Close'] / p['Close']) - 1) * 100
            
            status = "横ばい"
            if c['Close'] > df['MA5'].iloc[-1] > df['MA25'].iloc[-1] > df['MA75'].iloc[-1]:
                if df['MA25'].iloc[-1] > df['MA25'].iloc[-2]:
                    status = "🌟 パーフェクトオーダー"
            elif c['Close'] > df['MA5'].iloc[-1] > df['MA5'].iloc[-2]:
                status = "📈 5日線上向き"

            for theme in tickers_dict[t]["themes"]:
                results.append({
                    "テーマ": theme, "コード": t.replace(".T", ""), "銘柄名": tickers_dict[t]["name"],
                    "現在値": round(c['Close'], 1), "前日比": dod, "RSI": round(c['RSI'], 1), "判定": status
                })
        except: continue
    return pd.DataFrame(results)

# --- UI 構築 ---
tickers_dict = get_base_tickers()
all_tickers = list(tickers_dict.keys())

with st.spinner('データを解析中...'):
    raw_data = fetch_data(all_tickers)
    if raw_data is not None:
        analysis_df = analyze_stocks(raw_data, tickers_dict)
    else:
        st.error("データの取得に失敗しました。")
        st.stop()

tab1, tab2, tab3 = st.tabs(["🔥 強気銘柄スクリーナー", "📂 テーマ別動向", "📅 日付指定検索"])

# --- タブ1: スクリーナー ---
with tab1:
    col1, col2 = st.columns(2)
    po_stocks = analysis_df[analysis_df["判定"] == "🌟 パーフェクトオーダー"].drop_duplicates(subset="コード")
    ma5_stocks = analysis_df[analysis_df["判定"] == "📈 5日線上向き"].drop_duplicates(subset="コード")
    with col1:
        st.success(f"🌟 パーフェクトオーダー中 ({len(po_stocks)}銘柄)")
        st.dataframe(po_stocks[["コード", "銘柄名", "現在値", "前日比", "RSI"]].sort_values("前日比", ascending=False), hide_index=True)
    with col2:
        st.info(f"📈 短期上昇傾向 ({len(ma5_stocks)}銘柄)")
        st.dataframe(ma5_stocks[["コード", "銘柄名", "現在値", "前日比", "RSI"]].sort_values("前日比", ascending=False), hide_index=True)

# --- タブ2: テーマ分析 ---
with tab2:
    theme_perf = analysis_df.groupby("テーマ")["前日比"].mean().sort_values(ascending=False)
    for theme_name in theme_perf.index:
        avg_pct = theme_perf[theme_name]
        icon = "🟢" if avg_pct > 0 else "🔴"
        with st.expander(f"{icon} {theme_name} (平均: {avg_pct:+.2f}%)"):
            theme_df = analysis_df[analysis_df["テーマ"] == theme_name].sort_values("前日比", ascending=False)
            st.dataframe(theme_df[["コード", "銘柄名", "現在値", "前日比", "RSI", "判定"]], use_container_width=True, hide_index=True)

# --- タブ3: 日付指定検索 (NEW!) ---
with tab3:
    st.subheader("🔍 過去データの抽出")
    st.write("カレンダーから日付を選択して、その日の数値を表示します。")
    
    # 日付選択 (初期値は最新の平日)
    target_date = st.date_input("抽出日を選択", value=datetime.date.today())
    target_ts = pd.Timestamp(target_date)

    hist_results = []
    ts = list(tickers_dict.keys())
    
    found_any = False
    for t in ts:
        try:
            df_hist = raw_data[t].dropna()
            # RSIを履歴全体で計算
            df_hist['RSI'] = calc_rsi(df_hist['Close'])
            
            # 指定した日付がインデックスに存在するか確認
            if target_ts in df_hist.index:
                row = df_hist.loc[target_ts]
                for theme in tickers_dict[t]["themes"]:
                    hist_results.append({
                        "テーマ": theme,
                        "コード": t.replace(".T", ""),
                        "銘柄名": tickers_dict[t]["name"],
                        "引値": round(row['Close'], 1),
                        "出来高": int(row['Volume']),
                        "RSI": round(row['RSI'], 1) if not np.isnan(row['RSI']) else "算出中"
                    })
                found_any = True
        except: continue

    if found_any:
        df_hist_view = pd.DataFrame(hist_results)
        st.write(f"### {target_date} の市場データ")
        
        # テーマごとに並び替えて表示
        st.dataframe(
            df_hist_view.sort_values(["テーマ", "コード"]),
            use_container_width=True,
            hide_index=True
        )
        
        # CSVダウンロードボタン
        csv = df_hist_view.to_csv(index=False).encode('utf-8-sig')
        st.download_button("結果をCSVで保存", csv, f"stock_data_{target_date}.csv", "text/csv")
    else:
        st.warning(f"{target_date} のデータは見つかりませんでした。市場の休場日（土日・祝日）であるか、データ取得範囲（過去6ヶ月）外の可能性があります。")

