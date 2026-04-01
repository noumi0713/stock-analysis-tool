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
    # エラー回避のため、変数の定義を先頭に移動しました
    ticker_data = {}
    current_theme = "不明"
    
    raw_data = """
    1. AI・半導体
    6857/アドバンテスト 8035/東京エレクトロン 6723/ルネサスエレクトロニクス 6920/レーザーテック 7735/ＳＣＲＥＥＮホールディングス 6963/ローム 6707/サンケン電気 7282/豊田合成 9984/ソフトバンクグループ 6501/日立製作所
    2. 造船
    7011/三菱重工業 7012/川崎重工業 7014/名村造船所 7003/三井Ｅ＆Ｓ 7018/内海造船 6302/住友重機械工業 9101/日本郵船 9104/商船三井 9107/川崎汽船 7022/サノヤスホールディングス
    3. 量子
    6701/日本電気 6702/富士通 9432/日本電信電話 6501/日立製作所 6503/三菱電機 6971/京セラ 8053/住友商事 4704/トレンドマイクロ 6758/ソニーグループ 4063/信越化学工業
    4. 合成生物学・バイオ
    4502/武田薬品工業 4568/第一三共 4471/三洋化成工業 8111/ゴールドウイン 4613/関西ペイント 6028/テクノプロ・ホールディングス 2931/ユーグレナ 4523/エーザイ 4519/中外製薬 4901/富士フイルムホールディングス
    5. 航空・宇宙
    7011/三菱重工業 7013/ＩＨＩ 9412/スカパーＪＳＡＴホールディングス 464A/ＱＰＳ研究所 186A/アストロスケールホールディングス 9348/ｉｓｐａｃｅ 3402/東レ 7224/新明和工業 3524/日東製網 6965/浜松ホトニクス 290A/シンスペ
    6. デジタル・サイバーセキュリティ
    4704/トレンドマイクロ 3692/ＦＦＲＩセキュリティ 4441/トビラシステムズ 3916/デジタル・インフォメーション・テクノロジー 2326/デジタルアーツ 3040/ソリトンシステムズ 4258/網屋 9433/ＫＤＤＩ 4398/ブロードバンドセキュリティ 4722/フューチャー 6701/日本電気
    7. コンテンツ
    6758/ソニーグループ 9404/日本テレビホールディングス 7832/バンダイナムコホールディングス 4751/サイバーエージェント 9468/ＫＡＤＯＫＡＷＡ 7974/任天堂 4816/東映アニメーション 9684/スクウェア・エニックス・ホールディングス 9697/カプコン 3765/ガンホー・オンライン・エンターテイメント
    8. フードテック
    3182/オイシックス・ラ・大地 1332/ニッスイ 1333/マルハニチロ 2282/日本ハム 2607/不二製油グループ本社 2802/味の素 2811/カゴメ 2193/クックパッド 2296/伊藤ハム米久ホールディングス 2216/カンロ
    9. 資源・エネルギー・GX
    9501/東京電力ホールディングス 5020/ＥＮＥＯＳホールディングス 6752/パナソニックホールディングス 4204/積水化学工業 4107/伊勢化学工業 5711/三菱マテリアル 4118/カネカ 9503/関西電力 1605/ＩＮＰＥＸ 1963/日揮ホールディングス
    10. 防災・国土強靭化
    1414/ショーボンドホールディングス 1813/不動テトラ 9621/建設技術研究所 1417/ミライト・ワン 208A/構造計画研究所 8088/岩谷産業 6632/ＪＶＣケンウッド 5285/ヤマックス 1848/富士ピー・エス 1888/若築建設
    11. 創薬・先端医療
    4543/テルモ 7733/オリンパス 4519/中外製薬 4507/塩野義製薬 4523/エーザイ 4901/富士フイルムホールディングス 7747/朝日インテック 7701/島津製作所 6869/シスメックス 4527/ロート製薬
    12. フュージョンエネルギー
    5803/フジクラ 8801/三井不動産 6971/京セラ 6965/浜松ホトニクス 7011/三菱重工業 6501/日立製作所 6503/三菱電機 6504/富士電機 5802/住友電気工業 7013/ＩＨＩ
    13. マテリアル
    4063/信越化学工業 3436/ＳＵＭＣＯ 5713/住友金属鉱山 5726/大阪チタニウムテクノロジーズ 5333/日本碍子 5310/東洋炭素 5302/日本カーボン 5406/神戸製鋼所 5401/日本製鉄 5411/ＪＦＥホールディングス
    14. 港湾ロジスティクス
    9301/三菱倉庫 9303/住友倉庫 9364/上組 9302/三井倉庫ホールディングス 9147/ＮＸホールディングス 9107/川崎汽船 9101/日本郵船 9104/商船三井 9304/渋沢倉庫 9358/宇徳
    15. 防衛産業
    7011/三菱重工業 7012/川崎重工業 7013/ＩＨＩ 7721/東京計器 6946/日本アビオニクス 6703/沖電気工業 3105/日清紡ホールディングス 6486/イーグル工業 5631/日本製鋼所 8093/極東貿易
    16. 情報通信
    9432/日本電信電話 9433/ＫＤＤＩ 9434/ソフトバンク 4755/楽天グループ 9412/スカパーＪＳＡＴホールディングス 5801/古河電気工業 5802/住友電気工業 5803/フジクラ 6701/日本電気 6702/富士通
    17. 海洋
    6269/三井海洋開発 1963/日揮ホールディングス 7003/三井Ｅ＆Ｓ 7011/三菱重工業 6834/精工技研 6618/大泉製作所 5802/住友電気工業 6777/ｓａｎｔｅｃ 3648/ＡＧＳ 6340/渋谷工業
    18. (対米) 次世代原子力
    6501/日立製作所 7011/三菱重工業 1812/鹿島建設 1802/大林組 1803/清水建設 8058/三菱商事 1833/奥村組 7013/ＩＨＩ 8031/三井物産 8001/伊藤忠商事
    19. (対米) 天然ガス・AI電源
    6501/日立製作所 7011/三菱重工業 7013/ＩＨＩ 6503/三菱電機 5803/フジクラ 6762/ＴＤＫ 6981/村田製作所 6752/パナソニックホールディングス 9984/ソフトバンクグループ 6701/日本電気
    20. (対米) 原油インフラ・備蓄
    5020/ＥＮＥＯＳホールディングス 5019/出光興産 5021/コスモエネルギーホールディングス 1605/ＩＮＰＥＸ 8058/三菱商事 8031/三井物産 8001/伊藤忠商事 8053/住友商事 8002/丸紅 1963/日揮ホールディングス
    21. (対米) 先端マテリアル
    8031/三井物産 5711/三菱マテリアル 5802/住友電気工業 3402/東レ 3401/帝人 3407/旭化成 4205/日本ゼオン 4063/信越化学工業 4004/レゾナック・ホールディングス 4208/ＵＢＥ
    22. (対米) 重要鉱物資源
    5713/住友金属鉱山 5711/三菱マテリアル 8031/三井物産 8058/三菱商事 8015/豊田通商 5714/ＤＯＷＡホールディングス 5706/三井金属鉱業 5715/古河機械金属 3315/日本コークス工業 8002/丸紅
    23. フィジカルAI
    6506/安川電機 6954/ファナック 202A/豆蔵ホールディングス 3132/マクニカホールディングス 6268/ナブテスコ 6273/ＳＭＣ 6324/ハーモニック・ドライブ・システムズ 3741/セック 4425/Ｋｕｄａｎ 7779/サイバーダイン
    24. 蓄電池（次世代電池・材料含む）
    6752/パナソニックホールディングス 6762/ＴＤＫ 6981/村田製作所 4204/積水化学工業 4118/カネカ 4107/伊勢化学工業 3407/旭化成 6502/東芝 6810/マクセル 485A/パワーエックス 6617/東光高岳
    25. 再エネ
    9519/レノバ 1407/ウエストホールディングス
    26. 石炭
    1514/住石ホールディングス 8835/太平洋興発
    27. 天然ガス
    1663/Ｋ＆Ｏエナジーグループ 1963/日揮ホールディングス
    28. 電力卸
    9513/電源開発 9517/イーレックス
    29. 肥料
    4031/片倉コープアグリ 4979/ＯＡＴアグリオ
    30. バイオ燃料
    9212/ＧｒｅｅｎＥａｒｔｈＩｎｓｔｉｔｕｔｅ 2931/ユーグレナ
    """
    
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
if st.sidebar.button("🔄 初期リスト(30テーマ)にリセット"):
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
        st.dataframe(df_now[cols].style.map(style_trend, subset=['トレンド']), use_container_width=True, hide_index=True)
    else:
        st.info("現在、シグナル合致銘柄はありません。")

    st.header("⏳ 買い準備・監視候補")
    if not df_ready.empty:
        df_ready['RSI_val'] = df_ready['RSI'].astype(float)
        df_ready = df_ready.sort_values("RSI_val").drop(columns=['RSI_val'])
        cols = ["コード", "銘柄名", "テーマ", "現在値", "RSI", "トレンド", "Vol変化", "状況"]
        st.dataframe(df_ready[cols].style.map(style_trend, subset=['トレンド']), use_container_width=True, hide_index=True)
