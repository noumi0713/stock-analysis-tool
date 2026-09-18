"""Small, transport-independent entry point for analysis data."""
PUBLIC = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest"
BLOB = "https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest"
SWIPE = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swipe-decisions/swipe_review"


def access_document():
    return """# 分析データの取得入口

## 取得経路
Web閲覧で取得できない場合、設定済みのGitHub連携で同じraw URLまたはGitHubファイルURLを取得する。
Webツールの失敗だけでデータ全体を取得不能と判断しない。権限エラーは停止し、制限を回避しない。
大容量CSVやZIPを扱えない取得経路では、stock_status.csvとuniverse.csvで対象を確定し、
stocks/銘柄コード.csvを順次取得して数値処理する。要約・一覧・保存済み指標では代用しない。
全銘柄を処理できなければ未処理数を明示し、全東証を分析済みと表現しない。

## 最初に確認
1. [更新状態](PUBLIC/update_status.json)
2. [保存済みスナップショットの品質・対象日](PUBLIC/manifest.json)
3. [対象ユニバース](PUBLIC/universe.csv)
4. [銘柄別取得状態](PUBLIC/stock_status.csv)
5. [市場系列の状態](PUBLIC/market_status.json)

最新更新の失敗と保存済みスナップショットの品質は別に確認する。
manifestのrun_id・対象日・品質、実際の日足の最終日・行数を照合する。
分析に必要な終値が古い、品質が不明、取得できない場合は停止する。
PARTIALは全銘柄取得済みを意味しない。欠損・履歴不足・未処理を分離して報告する。

## 分析母集団
ユーザー指定が全東証ならuniverse.csvを基準にし、掲示板ランキングで母集団を制限しない。
ランキング内限定の指定の場合のみ、bbs_ranking_status.jsonの当日成功確認と
bbs_ranking_latest.csv・bbs_ranking_trends.csvを必須とする。
現在の収録数と実際の処理数を区別する。IPOは取得できた日足を使い、不足指標は算定不能とする。

## スワイプ選別
- [当日100銘柄のスワイプ母集団](PUBLIC/swipe_review_universe.json)
- [スワイプ母集団の状態](PUBLIC/swipe_review_status.json)
- ユーザー仕分け: SWIPE/data/YYYY-MM-DD.json
- チャッピー推奨: SWIPE/recommendations/YYYY-MM-DD.json

母集団は当日のYahoo掲示板投稿ランキング1〜100位のみ。5営業日騰落率は保存せず、
母集団JSON内の直近6営業日の調整後終値から読み込み時に計算して降順表示する。
興味あり/なしはユーザーの裁量ラベルであり、買いシグナルや収益優位性として扱わない。
チャッピー推奨も候補フラグであり、最終判断では120日価格・需給・材料・地合いを再確認する。

## 生データ
- [全銘柄CSV](PUBLIC/equities_120d.csv)
- [ZIP](PUBLIC/chatgpt_120d.zip)
- 個別CSV: PUBLIC/stocks/銘柄コード.csv
- 個別CSVのGitHub表示: BLOB/stocks/銘柄コード.csv
- [市場データ](PUBLIC/markets_120d.csv)
- [テーマ所属](PUBLIC/theme_members.csv)
- [計算式](PUBLIC/indicators.py)

終値×出来高は売買代金の推計であり純資金流入ではない。
テーマ所属は出典・基準日を確認する。掲示板本文と最新IRは別途取得・検証する。
目標株価・理論株価・短期利確目標を区別する。
""".replace("PUBLIC", PUBLIC).replace("BLOB", BLOB).replace("SWIPE", SWIPE)
