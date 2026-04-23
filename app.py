import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="総合スクリーナー＆テーマ分析", layout="wide")
st.title("📈 買い時キャッチ ＆ テーマ別動向")

# ==========================================
# 1. 銘柄データ（30の国策テーマ）
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
    "30. バイオ燃料", "9212/ＧｒｅｅｎＥａｒｔｈＩｎｓｔｉｔｕｔｅ 2931/ユーグレナ"
]

def get_base_tickers():
    t_dict = {}
    cur = "不明"
    for line in RAW_STOCK_LIST:
        if "/" not in line: cur = line
        else:
            for s in line.split():
                if "/" in s:
                    c, n = s.split("/")
                    tk = f"{c}.T"
                    if tk in t_dict:
                        if cur not in t_dict[tk]["theme"]: t_dict[tk]["theme"] += f", {cur}"
                    else: t_dict[tk] = {"name": n, "theme": cur}
    return t_dict

if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = get_base_tickers()

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 監視銘柄の管理")
st.sidebar.info(f"監視対象: {len(st.session_state.tickers_dict)} 銘柄")

st.sidebar.subheader("➕ 銘柄追加")
c_in = st.sidebar.text_area("形式: コード/銘柄名\n例: 4425/Kudan", height=100)
if st.sidebar.button("銘柄を追加"):
    if c_in:
        ms = re.findall(r'([A-Za-z0-9]{4,5})/([^\s]+)', c_in)
        for c, n in ms:
            st.session_state.tickers_dict[f"{c}.T"] = {"name": n, "theme": "追加"}
        st.rerun()

st.sidebar.subheader("🗑️ 銘柄削除")
del_target = st.sidebar.multiselect("削除を選択", options=list(st.session_state.tickers_dict.keys()), 
                                    format_func=lambda x: f"{x.replace('.T','')}/{st.session_state.tickers_dict[x]['name']}")
if st.sidebar.button("削除実行"):
    for t in del_target: st.session_state.tickers_dict.pop(t, None)
    st.rerun()

if st.sidebar.button("🔄 初期リストにリセット"):
    st.session_state.tickers_dict = get_base_tickers()
    st.rerun()

# --- データ取得・解析関数 ---
def get_vol_mul():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    if now.weekday() >= 5 or now.hour < 9 or now.hour >= 15: return 1.0
    if 9 <= now.hour < 11 or (now.hour == 11 and now.minute <= 30):
        elaps = (now.hour - 9) * 60 + now.minute
    elif 11 <= now.hour <= 12 and not (now.hour == 12 and now.minute >= 30):
        elaps = 150
    else:
        elaps = 150 + (now.hour - 12) * 60 + now.minute - 30
    return 300 / elaps if elaps > 0 else 1.0

@st.cache_data(ttl=300)
def fetch_data(ts):
    return yf.download(ts, period="1y", interval="1d", group_by="ticker", threads=True)

def analyze(data, ts, t_dict):
    po_list, ma5_list, theme_data = [], [], []
    v_mul = get_vol_mul()
    
    for t in ts:
        try:
            df = data[t].copy() if len(ts) > 1 else data.copy()
            df = df.dropna()
            if len(df) < 80: continue
            
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA25'] = df['Close'].rolling(25).mean()
            df['MA75'] = df['Close'].rolling(75).mean()
            d = df['Close'].diff()
            g = (d.where(d > 0, 0)).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + g / l))
            
            c, p = df.iloc[-1], df.iloc[-2]
            dod_pct = ((c['Close'] / p['Close']) - 1) * 100
            
            theme_data.append({
                "テーマ": t_dict[t]["theme"].split(", ")[0], 
                "銘柄名": t_dict[t]["name"], 
                "コード": t.replace(".T",""), 
                "前日比": dod_pct, 
                "現在値": c['Close'], 
                "RSI": c['RSI']
            })
            
            res = {
                "コード": t.replace(".T",""), 
                "銘柄名": t_dict[t]["name"], 
                "現在値": f"{c['Close']:.1f}", 
                "RSI": f"{c['RSI']:.1f}", 
                "前日比": f"{dod_pct:+.1f}%"
            }
            
            if c['Close'] > df['MA5'].iloc[-1] > df['MA25'].iloc[-1] > df['MA75'].iloc[-1] and df['MA25'].iloc[-1] > df['MA25'].iloc[-2]:
                po_list.append({**res, "評価": "🌟 上昇PO"})
            elif c['Close'] > df['MA5'].iloc[-1] > df['MA5'].iloc[-2]:
                ma5_list.append({**res, "評価": "📈 5日線上向き"})
        except Exception as e:
            continue
            
    return pd.DataFrame(po_list), pd.DataFrame(ma5_list), pd.DataFrame(theme_data)

# --- UI表示 ---
ts_list = list(st.session_state.tickers_dict.keys())
if ts_list:
    tab1, tab2 = st.tabs(["📊 スクリーナー", "🗺️ テーマ別動向 (プルダウン)"])
    
    with st.spinner('市場データを取得・解析中...'):
        df_po, df_ma5, df_theme = analyze(fetch_data(ts_list), ts_list, st.session_state.tickers_dict)
    
    def style_eval(v):
        if '🌟' in str(v): return 'color: #00FF00; font-weight: bold; background-color: #1E3A1E'
        if '📈' in str(v): return 'color: #00BFFF; font-weight: bold'
        return ''

    # --- タブ1: スクリーナー ---
    with tab1:
        st.header("🌟 パーフェクトオーダー（強い上昇トレンド）")
        if not df_po.empty:
            df_po['SortRSI'] = df_po['RSI'].astype(float)
            st.dataframe(df_po.sort_values('SortRSI').drop(columns=['SortRSI']).style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else:
            st.info("現在、パーフェクトオーダーを形成している銘柄はありません。")
            
        st.header("📈 5日線上向き（短期モメンタム継続・初動）")
        if not df_ma5.empty:
            df_ma5['SortDoD'] = df_ma5['前日比'].str.replace('%', '').astype(float)
            st.dataframe(df_ma5.sort_values('SortDoD', ascending=False).drop(columns=['SortDoD']).style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else:
            st.info("現在、該当する銘柄はありません。")

    # --- タブ2: テーマ別動向（プルダウン形式） ---
    with tab2:
        st.markdown("各テーマの**平均前日比**を算出し、強いセクター順に並べています。クリックして展開すると個別銘柄を確認できます。")
        
        if not df_theme.empty:
            # 1. テーマごとの平均前日比を計算して、強い順に並び替え
            theme_avg = df_theme.groupby('テーマ')['前日比'].mean().reset_index()
            theme_avg = theme_avg.sort_values('前日比', ascending=False)
            
            # プラスは緑、マイナスは赤に色付けする関数（データフレーム用）
            def color_dod(val):
                if '+' in str(val): return 'color: #00FF00'
                elif '-' in str(val): return 'color: #ff4b4b'
                return 'color: gray'

            # 2. 強いテーマから順番にプルダウン（expander）を作成
            for _, t_row in theme_avg.iterrows():
                theme_name = t_row['テーマ']
                avg_val = t_row['前日比']
                
                # アイコンの設定
                if avg_val > 0:
                    icon = "🟢"
                elif avg_val < 0:
                    icon = "🔴"
                else:
                    icon = "⚪"
                    
                expander_title = f"{icon} {theme_name} 【 平均: {avg_val:+.2f}% 】"
                
                with st.expander(expander_title):
                    # そのテーマに属する銘柄だけを抽出し、前日比が強い順に並べる
                    df_theme_stocks = df_theme[df_theme['テーマ'] == theme_name].sort_values('前日比', ascending=False)
                    
                    # 表示用にデータを綺麗に整形
                    display_df = df_theme_stocks[['コード', '銘柄名', '現在値', '前日比', 'RSI']].copy()
                    display_df['現在値'] = display_df['現在値'].apply(lambda x: f"{x:.1f} 円")
                    display_df['前日比'] = display_df['前日比'].apply(lambda x: f"{x:+.2f} %")
                    display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
                    
                    # テーブルとして綺麗に表示
                    st.dataframe(display_df.style.map(color_dod, subset=['前日比']), use_container_width=True, hide_index=True)
        else:
            st.warning("データが取得できませんでした。")
