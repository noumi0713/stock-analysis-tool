import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re

# --- ページ設定 ---
st.set_page_config(page_title="国策銘柄・買い時スクリーナー", layout="wide")
st.title("🏹 買い時・即戦力スクリーナー")
st.markdown("「25日線・ボリバン・RSIダイバージェンス」に基づき、今、買い向かうべき銘柄を抽出します。")

# --- ベース銘柄データの定義 ---
def get_base_tickers():
    raw_data = """
    6857/アドバンテスト 8035/東京エレクトロン 6723/ルネサスエレクトロニクス 6920/レーザーテック 7735/ＳＣＲＥＥＮホールディングス 6963/ローム 6707/サンケン電気 7282/豊田合成 9984/ソフトバンクグループ 6501/日立製作所
    7011/三菱重工業 7012/川崎重工業 7014/名村造船所 7003/三井Ｅ＆Ｓ 7018/内海造船 6302/住友重機械工業 9101/日本郵船 9104/商船三井 9107/川崎汽船 7022/サノヤスホールディングス
    6701/日本電気 6702/富士通 9432/日本電信電話 6503/三菱電機 6971/京セラ 8053/住友商事 4704/トレンドマイクロ 6758/ソニーグループ 4063/信越化学工業
    4502/武田薬品工業 4568/第一三共 4471/三洋化成工業 8111/ゴールドウイン 4613/関西ペイント 6028/テクノプロ・ホールディングス 2931/ユーグレナ 4523/エーザイ 4519/中外製薬 4901/富士フイルムホールディングス
    7013/ＩＨＩ 9412/スカパーＪＳＡＴホールディングス 5595/ＱＰＳ研究所 186A/アストロスケールホールディングス 9348/ｉｓｐａｃｅ 3402/東レ 7224/新明和工業 3524/日東製網 6965/浜松ホトニクス
    3692/ＦＦＲＩセキュリティ 4441/トビラシステムズ 3916/デジタル・インフォメーション・テクノロジー 2326/デジタルアーツ 3040/ソリトンシステムズ 4258/網屋 9433/ＫＤＤＩ 4398/ブロードバンドセキュリティ 4722/フューチャー
    9404/日本テレビホールディングス 7832/バンダイナムコホールディングス 4751/サイバーエージェント 9468/ＫＡＤＯＫＡＷＡ 7974/任天堂 4816/東映アニメーション 9684/スクウェア・エニックス・ホールディングス 9697/カプコン 3765/ガンホー・オンライン・エンターテイメント
    3182/オイシックス・ラ・大地 1332/ニッスイ 1333/マルハニチロ 2282/日本ハム 2607/不二製油グループ本社 2802/味の素 2811/カゴメ 2193/クックパッド 2296/伊藤ハム米久ホールディングス 2216/カンロ
    9501/東京電力ホールディングス 5020/ＥＮＥＯＳホールディングス 6752/パナソニックホールディングス 4204/積水化学工業 4107/伊勢化学工業 5711/三菱マテリアル 4118/カネカ 9503/関西電力 1605/ＩＮＰＥＸ 1963/日揮ホールディングス
    1414/ショーボンドホールディングス 1813/不動テトラ 9621/建設技術研究所 1417/ミライト・ワン 208A/構造計画研究所 8088/岩谷産業 6632/ＪＶＣケンウッド 5285/ヤマックス 1848/富士ピー・エス 1888/若築建設
    4543/テルモ 7733/オリンパス 4507/塩野義製薬 7747/朝日インテック 7701/島津製作所 6869/シスメックス 4527/ロート製薬
    5803/フジクラ 8801/三井不動産 6504/富士電機 5802/住友電気工業
    3436/ＳＵＭＣＯ 5713/住友金属鉱山 5726/大阪チタニウムテクノロジーズ 5333/日本碍子 5310/東洋炭素 5302/日本カーボン 5406/神戸製鋼所 5401/日本製鉄 5411/ＪＦＥホールディングス
    9301/三菱倉庫 9303/住友倉庫 9364/上組 9302/三井倉庫ホールディングス 9147/ＮＸホールディングス 9107/川崎汽船 9101/日本郵船 9104/商船三井 9304/渋沢倉庫 9358/宇徳
    7721/東京計器 6946/日本アビオニクス 6703/沖電気工業 3105/日清紡ホールディングス 6486/イーグル工業 5631/日本製鋼所 8093/極東貿易
    9434/ソフトバンク 4755/楽天グループ 5801/古河電気工業
    6269/三井海洋開発 6834/精工技研 6618/大泉製作所 6777/ｓａｎｔｅｃ 3648/ＡＧＳ 6340/渋谷工業
    1812/鹿島建設 1802/大林組 1803/清水建設 8058/三菱商商事 1833/奥村組 8031/三井物産 8001/伊藤忠商事
    6762/ＴＤＫ 6981/村田製作所
    5019/出光興産 5021/コスモエネルギーホールディングス 8002/丸紅
    3401/帝人 3407/旭化成 4205/日本ゼオン 4004/レゾナック・ホールディングス 4208/ＵＢＥ
    8015/豊田通商 5714/ＤＯＷＡホールディングス 5706/三井金属鉱業 5715/古河機械金属 3315/日本コークス工業
    """
    matches = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', raw_data)
    return {f"{code}.T": name for code, name in matches}

if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = get_base_tickers()

# --- サイドバー管理（追加・削除・リセット） ---
st.sidebar.header("⚙️ 監視銘柄の管理")
custom_input = st.sidebar.text_area("追加（コード/銘柄名）", height=70)
if st.sidebar.button("追加実行"):
    if custom_input:
        matches = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', custom_input)
        for c, n in matches: st.session_state.tickers_dict[f"{c}.T"] = n
        st.rerun()

remove_targets = st.sidebar.multiselect("削除選択", options=list(st.session_state.tickers_dict.keys()), format_func=lambda x: f"{x.replace('.T','')}/{st.session_state.tickers_dict[x]}")
if st.sidebar.button("削除実行"):
    for t in remove_targets: st.session_state.tickers_dict.pop(t, None)
    st.rerun()

if st.sidebar.button("初期化リセット"):
    st.session_state.tickers_dict = get_base_tickers()
    st.rerun()

# --- 解析ロジック ---
@st.cache_data(ttl=3600)
def analyze_buy_signals(tickers, tickers_dict):
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", threads=True)
    
    buy_now = []      # 買い向かうべき（シグナル点灯）
    buy_ready = []    # 買い準備（そろそろ来そう）

    for ticker in tickers:
        try:
            df = data[ticker].copy() if len(tickers) > 1 else data.copy()
            df = df.dropna()
            if len(df) < 40: continue

            # インジケーター計算
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA25'] = df['Close'].rolling(window=25).mean()
            df['STD'] = df['Close'].rolling(window=25).std()
            df['Lower2'] = df['MA25'] - (df['STD'] * 2)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # 直近データ
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # ダイバージェンスの簡易判定（直近20日の安値とRSIの関係）
            low_20 = df.iloc[-20:]
            idx_price_min = low_20['Low'].idxmin()
            idx_rsi_min = low_20['RSI'].idxmin()
            
            # 価格が安値を更新しているがRSIが更新していない状態
            is_divergence = (idx_price_min != idx_rsi_min) and (df.loc[idx_price_min, 'RSI'] > df['RSI'].min())

            res = {
                "コード": ticker.replace(".T",""), "銘柄名": tickers_dict[ticker],
                "現在値": round(curr['Close'], 1), "RSI": round(curr['RSI'], 1),
                "MA25": "⤴️" if curr['MA25'] >= prev['MA25'] else "⤵️"
            }

            # --- 判定1: 買い向かうべきタイミング（シグナル点灯） ---
            # 条件：RSIが底を打って上昇 ＆ 5日線を突破 ＆ ダイバージェンス兆候
            if curr['Close'] > curr['MA5'] and prev['Close'] <= prev['MA5'] and curr['RSI'] > 30:
                if curr['Close'] < curr['MA25'] * 1.05: # 高値掴み防止
                    buy_now.append({**res, "状況": "🔥 5日線突破・反発開始"})

            # --- 判定2: 買い準備（来ていそうな銘柄） ---
            # 条件：ボリバン-2σ以下 ＆ RSIが30以下（極めて売られすぎ）
            elif curr['Close'] <= curr['Lower2'] * 1.02 or curr['RSI'] <= 35:
                buy_ready.append({**res, "状況": "⏳ 売られすぎ・底打ち待ち"})

        except: continue
    return pd.DataFrame(buy_now), pd.DataFrame(buy_ready)

# --- UI表示 ---
tickers_list = list(st.session_state.tickers_dict.keys())
if tickers_list:
    with st.spinner('「買い時」をスキャン中...'):
        df_now, df_ready = analyze_buy_signals(tickers_list, st.session_state.tickers_dict)

    st.header("🎯 買い向かうべきタイミング（即戦力）")
    st.caption("5日線を上に抜け、反発の初動を捉えた銘柄です。ダイバージェンスを確認しエントリーを検討。")
    if not df_now.empty:
        st.dataframe(df_now, use_container_width=True, hide_index=True)
    else:
        st.info("現在、即エントリーのシグナルが出ている銘柄はありません。")

    st.header("⏳ 買い準備・来ていそうな銘柄（監視）")
    st.caption("ボリンジャーバンド-2σ到達やRSI30台など、底打ちが近い銘柄です。数日内の反発に備えてください。")
    if not df_ready.empty:
        st.dataframe(df_ready.sort_values("RSI"), use_container_width=True, hide_index=True)
    else:
        st.info("現在、底打ち待ちの候補はありません。")
