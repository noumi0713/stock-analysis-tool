# 指定銘柄 1分足取得

指定した銘柄だけを対象に、Yahoo Finance (`yfinance`) から1分足を取得します。
全東証の一括取得は行いません。

## GitHub Actions から実行

1. GitHub の **Actions** を開く
2. **Selected tickers 1-minute data** を選ぶ
3. **Run workflow** を押す
4. `tickers` に取得したい銘柄だけを入力する
   - 例: `7203,6758,9984`
   - 英字入り東証コードも可: `130A`
5. `days` を 1〜7 から選ぶ
6. `session` を選ぶ
   - `morning`: 09:00〜11:30 JST
   - `full`: 09:00〜11:30 + 12:30〜15:30 JST
7. 実行後、Artifact `intraday-1m-<run_id>` を取得する

`7203` のような4桁コードや `130A` のような東証コードには `.T` を自動付与します。
`7203.T` のように入力しても構いません。

## ローカル実行

```bash
pip install -r swing_data/requirements.txt
python -m swing_data.intraday_1m --tickers 7203 6758 9984 --days 7 --session morning
```

カンマ区切りでも指定できます。

```bash
python -m swing_data.intraday_1m --tickers "7203,6758,9984" --days 7 --session morning
```

## 出力

`intraday_1m_output/` に次を出力します。

- `<ticker>_1m_<days>cald.csv`: 銘柄別1分足
- `selected_tickers_1m_combined.csv`: 指定銘柄をまとめた1分足
- `morning_low_summary.csv`: 各銘柄・各日の前場安値時刻
- `manifest.json`: 取得成功/失敗、行数、実際に含まれた取引日数

`morning_low_summary.csv` には以下を含みます。

- `Ticker`
- `Date`
- `MorningLow`
- `MorningLowTime`
- `MinutesFromOpen`
- `Open`
- `MorningClose`
- `OpenToLowPct`
- `LowToMorningClosePct`

これにより「寄り付きから何分後に前場安値を付けたか」を銘柄・日ごとに集計できます。

## 期間について

1分足は Yahoo Finance 側の短期データ制限があります。この実装では `interval="1m"` と `period="max"` を使い、yfinance が返す直近の取得可能範囲から、指定した 1〜7 **暦日**に絞ります。

そのため `days=7` は「7営業日」を保証しません。土日・祝日を含む場合、実際に含まれる取引日は少なくなります。実際の取引日数は `manifest.json` の `trading_dates` で確認してください。

7営業日以上を確実に統計に使う場合は、毎日実行して履歴を自前で蓄積する方式に拡張してください。
