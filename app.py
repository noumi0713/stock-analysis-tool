import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re

# --- ページ設定 ---
st.set_page_config(page_title="スイングトレード・スクリーナー", layout="wide")
st.title("📈 ダイバージェンス・スクリーナー")
st.markdown("移動平均線・ボリンジャーバンド・RSIを用いた戦略に基づき、220銘柄+αをリアルタイム解析します。")

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
    1812/鹿島建設 1802/大林組 1803/清水建設 8058/三菱商事 1833/奥村組 8031/三井物産 8001/伊藤忠商事
    6762/ＴＤＫ 6981/村田製作所
    5019/出光興産 5021/コスモエネルギーホールディングス 8002/丸紅
    3401/帝人 3407/旭化成 4205/日本ゼオン 4004/レゾナック・ホールディングス 4208/ＵＢＥ
    8015/豊田通商 5714/ＤＯＷＡホールディングス 5706/三井金属鉱業 5715/古河機械金属 3315/日本コークス工業
    """
    matches = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', raw_data)
    return {f"{code}.T": name for code, name in matches}

# --- セッションステート（状態保持）の初期化 ---
if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = get_base_tickers()

# --- サイドバー：銘柄管理 UI ---
st.sidebar.header("⚙️ 監視銘柄の管理")
st.sidebar.info(f"現在の監視対象: 合計 {len(st.session_state.tickers_dict)} 銘柄")

# 1. 銘柄の追加
st.sidebar.subheader("➕ 追加")
custom_tickers_input = st.sidebar.text_area("追加する銘柄（形式: コード/銘柄名）\n例: 7203/トヨタ自動車", height=100)
if st.sidebar.button("銘柄を追加"):
    if custom_tickers_input:
        custom_matches = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', custom_tickers_input)
        added_count = 0
        for code, name in custom_matches:
            ticker_code = f"{code}.T"
            if ticker_code not in st.session_state.tickers_dict:
                st.session_state.tickers_dict[ticker_code] = name
                added_count += 1
        if added_count > 0:
            st.sidebar.success(f"{added_count}銘柄を追加しました！")
            st.rerun()
        else:
            st.sidebar.warning("正しい形式（コード/名称）で入力してください。")

# 2. 銘柄の削除
st.sidebar.subheader("🗑️ 削除")
current_tickers = list(st.session_state.tickers_dict.keys())
display_options = {t: f"{t.replace('.T', '')}/{st.session_state.tickers_dict[t]}" for t in current_tickers}

remove_targets = st.sidebar.multiselect(
    "削除する銘柄を選択",
    options=current_tickers,
    format_func=lambda x: display_options.get(x, x)
)
if st.sidebar.button("選択した銘柄を削除"):
    if remove_targets:
        for t in remove_targets:
            st.session_state.tickers_dict.pop(t, None)
        st.sidebar.success(f"{len(remove_targets)}銘柄を削除しました！")
        st.rerun()

# 3. リセット
st.sidebar.subheader("🔄 リセット")
if st.sidebar.button("初期リスト(220銘柄)に戻す"):
    st.session_state.tickers_dict = get_base_tickers()
    st.sidebar.success("初期リストにリセットしました！")
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("📉 データを再取得・解析"):
    st.cache_data.clear()
    st.rerun()

# --- データ取得・解析処理 ---
@st.cache_data(ttl=3600)
def fetch_and_analyze(tickers, tickers_dict):
    # show_errors=Falseを削除してTypeErrorを回避
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", threads=True)
    
    perfect_matches = []
    near_matches = []
    
    for ticker in tickers:
        try:
            # 1銘柄のみの場合と複数銘柄の場合でデータフレームの構造が異なるため対応
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
            
            df = df.dropna()
            if len(df) < 30: continue
            
            # テクニカル指標計算
            df['MA25'] = df['Close'].rolling(window=25).mean()
            df['STD'] = df['Close'].rolling(window=25).std()
            df['Upper2'] = df['MA25'] + (df['STD'] * 2)
            df['Lower2'] = df['MA25'] - (df['STD'] * 2)
            
            # RSI(14)
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/14, min_periods=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            is_ma25_up = latest['MA25'] > prev['MA25']
            
            close = latest['Close']
            lower2 = latest['Lower2']
            upper2 = latest['Upper2']
            rsi = latest['RSI']
            
            result_data = {
                "コード": ticker.replace(".T", ""),
                "銘柄名": tickers_dict[ticker],
                "現在値": round(close, 1),
                "MA25向き": "⤴️ 上" if is_ma25_up else "⤵️ 下",
                "RSI": round(rsi, 1),
                "-2σ": round(lower2, 1),
                "+2σ": round(upper2, 1)
            }
            
            # 1. 完全合致の判定
            perfect_buy = is_ma25_up and (close <= lower2 * 1.02) and (rsi < 40)
            perfect_sell = (close >= upper2 * 0.98) and (rsi > 70)
            
            # 2. 監視候補の判定
            near_buy = is_ma25_up and (close <= lower2 * 1.05) and (rsi < 50) and not perfect_buy
            near_sell = (close >= upper2 * 0.95) and (rsi > 60) and not perfect_sell
            
            if perfect_buy:
                perfect_matches.append({**result_data, "シグナル": "🟢 完全合致: 押し目"})
            elif perfect_sell:
                perfect_matches.append({**result_data, "シグナル": "🔴 完全合致: 天井警戒"})
            elif near_buy:
                near_matches.append({**result_data, "シグナル": "🟡 監視候補: 押し目接近"})
            elif near_sell:
                near_matches.append({**result_data, "シグナル": "🟠 監視候補: 天井接近"})

        except Exception:
            continue
            
    return pd.DataFrame(perfect_matches), pd.DataFrame(near_matches)

# --- UI描画 ---
tickers_list = list(st.session_state.tickers_dict.keys())

if not tickers_list:
    st.warning("監視対象がありません。サイドバーから追加するかリセットしてください。")
else:
    with st.spinner('全銘柄のデータを解析中...'):
        df_perfect, df_near = fetch_and_analyze(tickers_list, st.session_state.tickers_dict)

    st.header("🎯 条件クリア銘柄（完全合致）")
    if not df_perfect.empty:
        st.dataframe(df_perfect.sort_values(['シグナル', 'RSI']), use_container_width=True, hide_index=True)
    else:
        st.info("現在、完全に条件をクリアしている銘柄はありません。")

    st.header("👀 監視候補（わずかにクリアできなかった銘柄）")
    if not df_near.empty:
        st.dataframe(df_near.sort_values(['シグナル', 'RSI']), use_container_width=True, hide_index=True)
    else:
        st.info("現在、監視候補に該当する銘柄はありません。")
