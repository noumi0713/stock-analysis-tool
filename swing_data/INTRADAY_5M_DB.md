# 5分足・日中安値DB（第1段階）

指定した東証銘柄について、Yahoo Finance の5分足を取得し、SQLite に日付単位で保存します。同じ日を再取得するとその日の行を置き換えます。`bars` は5分足OHLCV、`daily_lows` は日中の安値・最初の安値時刻・高値・始値・終値です。時刻は日本時間、価格は未調整です。

対象は `swing_data/config/intraday_5m_tickers.json` に列挙します。最初は監視銘柄と日中分析対象の10銘柄です。ETFの1570も既存の日中分析対象として含めています。勝負でETFを除外する場合は、選定側で除外してください。

## 初期取得と自動更新

GitHub Actions の **Intraday 5-minute database** を手動で実行すると、初回は直近59暦日を取り込みます。平日16:45 JSTの自動実行は直近7暦日を再取得し、遅延・訂正を反映します。正常終了後に `intraday-5m-data` ブランチへ最新版の `intraday_5m.sqlite` と取得レポートを保存します。データの履歴はSQLite内に保持し、配布ブランチ自体は最新版のスナップショットへ更新します。一部銘柄の取得失敗はレポートに記録して成功分を保存します。全銘柄で有効な取引日を取得できなければ発行を停止します。

手元での実行例:

```bash
pip install -r swing_data/requirements.txt
python -m swing_data.intraday_5m_db --lookback-days 59 --db intraday_5m.sqlite --report intraday_5m_report.json
```

実行時刻が引け後で、当日のデータも保存する場合に限り `--include-today` を付けます。15:40 JST以前の実行では指定しても当日は除外します。部分的な日（始値9:00や15:20以降の足がない、50本未満など）は取り込みません。売買停止日や短縮取引日も対象外になる可能性があります。

```bash
sqlite3 intraday_5m.sqlite 'SELECT ticker,trading_date,low,first_low_time,bar_count FROM daily_lows ORDER BY trading_date DESC,ticker LIMIT 20;'
```

Yahoo/yfinance の過去の日中足は約60暦日に制限されます。過去1～3年を今すぐ復元する仕組みではなく、これから毎日蓄積します。Yahooの取得失敗、仕様変更、無償データの欠落は起こり得ます。品質監視と予測モデルは次の段階で作ります。

データブランチは現在のリポジトリ内です。対象銘柄を大幅に増やす前に、容量・利用条件を確認して専用ストレージに移してください。
