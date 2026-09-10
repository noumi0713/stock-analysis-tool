# 日本株120営業日データ

5〜10営業日の裁量スイング分析に渡すデータ収集基盤。売買シグナルや候補フィルターは生成しません。

## 取得対象

- JPXの最新一覧に載る東証プライム・スタンダード・グロースの内国普通株式。ETF・REIT・ETNは対象外。
- XTKSの終了済みセッションから厳密に120営業日。引け後30分までは当日を使用しません。
- 以前の124テーマ・13クラスター（3,298銘柄、10,870所属）と関連度評価を原本のまま継承。テーマ未所属株も取得します。
- 120営業日未満、欠損、古い株価、異常OHLC、ゼロ出来高のある株は状態別に区別し、適格データに混ぜません。
- 指数・ドル円・日米10年金利は設定ファイルで管理し、取得元・日付・単位・使用フィールドを系列ごとに記録します。

## 毎日更新と手動更新

`.github/workflows/swing-data-120d.yml`をデフォルトブランチに登録すると、平日17:30・19:30 JSTに実行されます。祝日は最新の終了済み営業日が対象です。後の回は取得遅延への再取得も兼ねます。

サイトの「手動更新」からGitHubの専用実行画面を開き、サインインして「Run workflow」を押します。認証トークンをサイトに保存する必要はありません。単なる画面再読込とデータの再取得を区別します。

収集コードは専用ブランチ `feature/swing-120d-data`。出力はデータ専用ブランチ `swing-data-120d-latest` へ1コミットで公開します。既存の売買戦略、研究STEP、定期テーマ更新は変更しません。

初回は対象全件を取得します。毎回120営業日を再取得し、株式分割などの訂正を反映します。最大3試行、同時ダウンロード数4、40銘柄単位。全滅時は前回のデータを残し、失敗状態のみ更新します。一部取得時はPARTIALとし、件数・個別理由・欠損市場を開示します。

長期検証データとは別の120営業日スナップショットです。各実行の成果物はActionsに30日保存され、最新データは専用ブランチで保持されます。以前の7年データは変更しません。

## チャッピーから読む

入口:
https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest/index.md

ここから更新状態、manifest、全銘柄CSV/Parquet/ZIP、各銘柄CSV、テーマと市場データへ辿れます。URLを毎日の分析プロンプトに付ければ、ファイルを毎回手で添付する必要はありません。ただし利用するチャッピーにWeb/ファイル取得・数値処理の能力が必要です。データ量による切り詰めを全件分析とみなさないでください。

## ローカル実行

```bash
python -m pip install -r swing_data/requirements.txt
python -m swing_data.collector
python -m swing_data.publish --source swing_data/runtime --target public-data
```

少数銘柄で接続確認する場合は `python -m swing_data.collector --tickers 7203 9984 --output /tmp/swing-smoke`。一部銘柄実行は全件取得と明確に区別されます。

検証: `python -m pytest tests/swing_data -q`

## データ上の制約

個別株はYahoo Financeの研究用データです。配信遅延や取得制限があり、公式取引所フィードの保証はありません。`auto_adjust=False`のOHLCと配当等調整後の`adj_*`を別列で保持します。売買代金は終値×出来高の概算です。

現在のテーマ所属を過去に適用する集計には所属の先読みがあり、現在上場している銘柄には生存者バイアスがあります。この出力をそのまま戦略バックテストには使わないでください。既存関連度には低信頼度や裁量評価も含まれ、最新のみんかぶ順位を再取得した値ではありません。自動的に関連度を再推定したり所属を削除したりしません。

### 市場系列と品質検査

- TOPIXはYahoo!ファイナンス日本版の`998405.T`の指数時系列を取得します。指数名と列名を確認し、ページ送りで120営業日の範囲を取得します。
- VIXはCboe公式のVIX日次履歴CSVからOHLCを取得します。Yahoo側で始値・高値・安値がゼロになった日を架空の値で補正しません。
- グロース250はTradingView公開チャートの`TSE:MOS`を使います。20分遅延の現物指数で、ETF・先物ではありません。指数名・種別・取引所・時間帯を検証し、東証の終了済み120営業日だけを採用します。内部の系列キー`^TSEMOTHERS`は既存ファイルとの互換性のため維持し、取得には使いません。公開チャートの通信仕様は公式API契約ではないため、変更・停止を検知した場合はFAILとし、代替銘柄で穴埋めしません。認証・有料フィードは使用しません。
- 日本10年金利は財務省の国債金利情報CSVを使います。10年コンスタントマチュリティー金利（単位%）で、新発10年国債の取引利回りとは定義が異なります。観測日の翌営業日9:30頃という公表予定に合わせて120観測日を選びます。利回りの負値・ゼロは有効です。
- ドル円はYahooの日次終値を使用します。終値が有効でもOHLCの大小関係が異常な日を一律に削除していたため、終値の検証とOHLCの検証を分離しました。異常OHLCの日付・値は`market_status.json`、取得時の全フィールドは`market_*.raw.csv`に残します。終値に異常や欠測があれば適格にはしません。
- `field_mode=close_only`は日次値のみです。始値・高値・安値は空欄とし、ローソク足やATRには使えません。日次金利を架空のOHLCに変換しません。利回りの差分の単位はpercentage point（100倍でbp）です。
- 海外指数はそれぞれ終了済み取引日まで、FXは提供元の日付で進行中のUTC当日を除外します。市場をまたぐ前方補完や先読み結合はしません。
- `market_status.json`に系列ごとの120日検査・欠損日・不正行・取得元と取得証跡を保存します。必須系列に不足がある場合、`manifest.json`の`market_quality=FAIL`と`required_market_failures`で明示し、更新ジョブも失敗扱いにします。取得済みデータと個別株の品質は別に保持します。

公式仕様参照:
- https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
- https://finance.yahoo.co.jp/quote/998405.T/history
- https://www.mof.go.jp/jgbs/reference/interest_rate/index.htm
- https://www.mof.go.jp/faq/jgbs/04hf.htm
- https://www.cboe.com/tradable-products/vix/vix-historical-data
- https://www.tradingview.com/symbols/TSE-MOS/
- https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule
