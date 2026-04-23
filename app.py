import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="プロフェッショナル・セクター・スクリーナー", layout="wide")
st.title("📈 戦略的テーマ監視ダッシュボード")

# ==========================================
# 1. 銘柄データ（ご提示の全33テーマ・保有銘柄優先）
# ==========================================
RAW_STOCK_LIST = [
    "1. AI・半導体", "6963/ローム 6920/レーザーテック 7735/SCREENホールディングス 8035/東京エレクトロン 6857/アドバンテスト 6146/ディスコ 9984/ソフトバンクグループ 6503/三菱電機 6758/ソニーグループ 285A/キオクシアホールディングス",
    "2. 造船", "7012/川崎重工業 7011/三菱重工業 7014/名村造船所 7003/三井E&S 6023/ダイハツディーゼル 6018/阪神内燃機工業 6022/赤阪鐵工所 6302/住友重機械工業 6814/古野電気 6016/ジャパンエンジンコーポレーション",
    "3. 量子", "6501/日立製作所 9432/NTT 6702/富士通 6701/日本電気 6728/アルバック 6503/三菱電機 3915/テラスカイ 6088/シグマクシス・ホールディングス 2693/YKT 3687/フィックスターズ",
    "4. 合成生物学・バイオ", "2931/ユーグレナ",
    "5. 航空・宇宙", "9412/スカパーJSATホールディングス 186A/アストロスケールホールディングス 464A/QPS研究所 4091/日本酸素ホールディングス 4188/三菱ケミカルグループ 4202/ダイセル 4026/神島化学工業 3402/東レ 3524/日東製網 3569/セーレン",
    "6. デジタル・サイバーセキュリティ", "4704/トレンドマイクロ 4493/サイバーセキュリティクラウド 4417/グローバルセキュリティエキスパート",
    "7. コンテンツ", "4816/東映アニメーション 5032/ANYCOLOR 5253/カバー 3659/ネクソン 9605/東映",
    "8. フードテック", "2897/日清食品ホールディングス 1333/マルハニチロ 2282/日本ハム 2296/伊藤ハム米久ホールディングス 2607/不二製油グループ本社 4188/三菱ケミカルグループ 1332/ニッスイ 6088/シグマクシス・ホールディングス 7701/島津製作所 7911/TOPPANホールディングス",
    "9. 資源・エネルギー・GX", "5711/三菱マテリアル 6752/パナソニックホールディングス 1605/INPEX 5019/出光興産 5020/ENEOSホールディングス 5016/JX金属 5021/コスモエネルギーホールディングス 6330/東洋エンジニアリング 6366/千代田化工建設 6378/木村化工機",
    "10. 防災・国土強靭化", "",
    "11. 創薬・先端医療", "4568/第一三共 4588/オンコリスバイオファーマ 4593/ヘリオス 4597/ソレイジア・ファーマ 4586/メドレックス",
    "12. フュージョンエネルギー", "5803/フジクラ 6503/三菱電機",
    "13. マテリアル", "5726/大阪チタニウムテクノロジーズ 5713/住友金属鉱山 5333/日本碍子 3436/SUMCO",
    "14. 港湾ロジスティクス", "9301/三菱倉庫 9302/三井倉庫ホールディングス 9303/住友倉庫 9304/渋沢倉庫 9357/名港海運 9359/伊勢湾海運 9360/鈴与シンワ 9361/伏木海陸運送 9362/兵機海運 9310/トランシティ",
    "15. 防衛産業", "7011/三菱重工業 7012/川崎重工業 7721/東京計器 3105/日清紡ホールディングス 4274/細谷火工 4275/カーリット 4284/ソルクシーズ 4316/ビーマップ 5631/日本製鋼所 5189/桜ゴム",
    "16. 情報通信", "5803/フジクラ 9434/ソフトバンク 9412/スカパーJSATホールディングス 5801/古河電気工業 9432/NTT 9433/KDDI 3626/TIS 4307/野村総合研究所 4684/オービック 6701/日本電気",
    "17. 海洋", "1885/東亜建設工業 1893/五洋建設 3176/三洋貿易 4403/日油 5401/日本製鉄 5541/大平洋金属 5715/古河機械金属 6269/三井海洋開発 4673/川崎地質 6361/荏原製作所",
    "18. (対米) 次世代原子力", "6501/日立製作所 7011/三菱重工業 7013/IHI",
    "19. (対米) 天然ガス・AI電源", "5803/フジクラ 6503/三菱電機 6752/パナソニックホールディングス 6981/村田製作所 6762/TDK",
    "20. (対米) 原油インフラ・備蓄", "1605/INPEX 5020/ENEOSホールディングス 5021/コスモエネルギーホールディングス 5019/出光興産",
    "21. (対米) 先端マテリアル", "5711/三菱マテリアル 4063/信越化学工業 4004/レゾナック・ホールディングス",
    "22. (対米) 重要鉱物資源", "5711/三菱マテリアル 5713/住友金属鉱山 5714/DOWAホールディングス 5706/三井金属鉱業",
    "23. フィジカルAI", "4425/Kudan 5885/ジーデップ・アドバンス 6268/ナブテスコ 6273/SMC 6302/住友重機械工業 6324/ハーモニック・ドライブ・システムズ 3444/菊池製作所 3652/ディジタルメディアプロフェッショナル 3741/セック 3443/川田テクノロジーズ",
    "24. 蓄電池", "6752/パナソニックホールディングス 6981/村田製作所 6617/東光高岳 485A/パワーエックス",
    "25. 再エネ", "3150/グリムス 3156/レスター 3232/三重交通グループホールディングス 3266/ファンドクリエーショングループ 1798/守谷商会 1832/北海電気工事 1925/大和ハウス工業 1945/東京エネシス 1407/ウエストホールディングス 1434/JESCOホールディングス",
    "26. 石炭", "",
    "27. 天然ガス", "1605/INPEX",
    "28. 電力卸", "9517/イーレックス",
    "29. 肥料", "2931/ユーグレナ",
    "30. バイオ燃料", "2931/ユーグレナ 2613/J-オイルミルズ 4631/DIC 5020/ENEOSホールディングス 5021/コスモエネルギーホールディングス 6330/東洋エンジニアリング 6366/千代田化工建設 6378/木村化工機 6902/デンソー 7011/三菱重工業",
    "31. 金・貴金属・リサイクル", "",
    "32. 純内需・ディフェンシブ", "2897/日清食品ホールディングス 2914/JT 3038/神戸物産 9434/ソフトバンク 9843/ニトリホールディングス 9989/サンドラッグ 8267/イオン",
    "33. インド・グローバルサウス開拓", ""
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

# --- サイドバー管理 ---
st.sidebar.header("⚙️ システム設定")
st.sidebar.info(f"監視銘柄数: {len(st.session_state.tickers_dict)}")

if st.sidebar.button("🔄 初期銘柄リストにリセット"):
    st.session_state.tickers_dict = get_base_tickers()
    st.rerun()

# --- 解析ロジック ---
# サーバー負荷対策: periodを6ヶ月(6mo)に短縮
@st.cache_data(ttl=600)
def fetch_data(ts):
    return yf.download(ts, period="6mo", interval="1d", group_by="ticker", threads=True)

def analyze(data, ts, t_dict):
    po_list, ma5_list, theme_data = [], [], []
    for t in ts:
        try:
            df = data[t].dropna()
            if len(df) < 50: continue
            
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
                "銘柄名": t_dict[t]["name"], "コード": t.replace(".T",""), 
                "前日比": dod_pct, "現在値": c['Close'], "RSI": c['RSI']
            })
            
            res = {"コード": t.replace(".T",""), "銘柄名": t_dict[t]["name"], "現在値": f"{c['Close']:.1f}", "RSI": f"{c['RSI']:.1f}", "前日比": f"{dod_pct:+.1f}%"}
            
            # パーフェクトオーダー判定
            if c['Close'] > df['MA5'].iloc[-1] > df['MA25'].iloc[-1] > df['MA75'].iloc[-1] and df['MA25'].iloc[-1] > df['MA25'].iloc[-2]:
                po_list.append({**res, "評価": "🌟 上昇PO"})
            # 5日線上向き判定
            elif c['Close'] > df['MA5'].iloc[-1] > df['MA5'].iloc[-2]:
                ma5_list.append({**res, "評価": "📈 5日線上向き"})
        except: continue
    return pd.DataFrame(po_list), pd.DataFrame(ma5_list), pd.DataFrame(theme_data)

# --- UI表示 ---
ts_list = list(st.session_state.tickers_dict.keys())
if ts_list:
    tab1, tab2 = st.tabs(["📊 スクリーナー", "📂 テーマ別分析 (プルダウン)"])
    
    with st.spinner('データを取得中...'):
        df_po, df_ma5, df_theme = analyze(fetch_data(ts_list), ts_list, st.session_state.tickers_dict)
    
    def style_eval(v):
        if '🌟' in str(v): return 'color: #00FF00; font-weight: bold; background-color: #1E3A1E'
        if '📈' in str(v): return 'color: #00BFFF; font-weight: bold'
        return ''

    with tab1:
        st.header("🌟 強気トレンド銘柄 (PO)")
        if not df_po.empty:
            st.dataframe(df_po.style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else: st.info("該当なし")
            
        st.header("📈 短期リバウンド (5日線上向き)")
        if not df_ma5.empty:
            st.dataframe(df_ma5.style.map(style_eval, subset=['評価']), use_container_width=True, hide_index=True)
        else: st.info("該当なし")

    with tab2:
        if not df_theme.empty:
            # テーマ平均でソート
            theme_avg = df_theme.groupby('テーマ')['前日比'].mean().sort_values(ascending=False)
            for theme, avg in theme_avg.items():
                icon = "🟢" if avg > 0 else "🔴"
                with st.expander(f"{icon} {theme} 【 平均 {avg:+.2f}% 】"):
                    stocks = df_theme[df_theme['テーマ'] == theme].sort_values('前日比', ascending=False)
                    stocks['前日比'] = stocks['前日比'].apply(lambda x: f"{x:+.2f}%")
                    st.dataframe(stocks[['コード', '銘柄名', '現在値', '前日比', 'RSI']], use_container_width=True, hide_index=True)
