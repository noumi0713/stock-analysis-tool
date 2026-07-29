# 日本株・短期上昇シグナル検知基盤 — Phase 1 + ベースライン検証

各営業日の引け後に利用可能な情報だけを使い、将来「5営業日以内に5%以上上昇する可能性」を推定するための基盤です。Phase 1の取得・保存・品質検査に加え、リーク検査済みの特徴量、ベースラインモデル、ウォークフォワード・バックテスト、日次ランキングを実装しています。本番運用モデルの選定・チューニングは今後のPhaseです。

現在の価格・出来高の取得源は `yfinance` です。個人研究・非公開利用だけを想定し、直近365日をローリング保存します。Prime銘柄コード一覧には保存済みJ-Quants銘柄マスタを使いますが、日足価格はYahoo Finance由来です。従来のJ-Quants取得コマンドとバックテスト実装は互換性のため残しています。

`yfinance` はYahoo公式ライブラリではなく、仕様変更・レート制限・データ訂正の可能性があります。取得データの公開・再配布は行わず、利用前にYahooの最新利用条件を確認してください。

## 実装済みの機能

- 上場銘柄一覧の日次スナップショット
- 調整前／調整済みOHLC、出来高、売買代金、調整係数
- 各営業日時点の市場区分に基づくPrime銘柄限定ユニバース
- TOPIX日足および契約プランで利用可能な指数日足
- 公式取引カレンダー
- 初回期間取得と、最終株価保存日の翌日からの差分更新
- `pagination_key`、レート制限、接続／読取タイムアウト、再試行、`Retry-After`
- APIページ単位のチェックポイントと途中再開
- raw／processedのParquet保存とDuckDBビュー
- 主キーupsertと重複検査
- OHLC、欠損、出来高、異常リターン、株式分割、過去時点ユニバースの品質検査
- ファイルログとpytest
- 過去20営業日までの価格・出来高・売買代金特徴量と日次横断順位
- 未来5営業日の最大高値が当日終値比+5%以上となる教師ラベル
- 5営業日のラベル期間をパージしたウォークフォワード学習
- 上位N銘柄ランキングと、翌営業日始値エントリーの取引評価
- 取引コスト、5%利確、5営業日目決済、段階投入を考慮した資産曲線

## 必要環境

- Python **3.12**
- J-Quants API V2のアカウントとAPIキー
- 指数データも取得する場合は対応する契約プラン

現在の公式プランでは、TOPIXはLight以上、TOPIX以外の指数はStandard以上が必要です。権限のない任意データセットがHTTP 403を返した場合、株価と銘柄一覧の取得は継続し、`status` に利用不可として記録します。

## セットアップ

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

`.env` にJ-Quantsダッシュボードで発行したキーを設定します。

```dotenv
JQUANTS_API_KEY=your-api-key
UNIVERSE_MARKET_CODES=0111
```

`.env`、`data/`、`logs/` は `.gitignore` 対象です。APIキーをコード、設定サンプル、ログへ書かないでください。

`0111` はPrime市場です。日次株価APIには市場区分パラメータがないため、APIから返ったページを同日の銘柄マスタで直ちに絞り込み、raw／processedにはPrime銘柄だけを保存します。`equities_daily` だけを指定した場合も、判定に必要な `listed_master` は自動取得されます。

## 実行方法

### yfinanceによる日次運用

初回取得と毎日の差分更新は同じコマンドです。既存データの最終日から10日重ねて取得し、訂正値を上書きしたあと、基準日から365日より古い行を削除します。

```bash
python -m app yahoo-ingest
python -m app yahoo-validate
python -m app yahoo-analyze
python -m app yahoo-export-dashboard --output work/latest.json
python -m app yahoo-status
```

`yahoo-ingest` は保存済みの最新Prime銘柄スナップショットをYahooティッカー（例: `7203.T`）へ変換します。任意の銘柄一覧を使う場合は、1行1ティッカーのUTF-8ファイルを指定します。

```bash
python -m app yahoo-ingest --tickers-file tickers.txt
python -m app yahoo-ingest --as-of 2026-07-29
python -m app yahoo-ingest --full-refresh
python -m app yahoo-analyze --top-n 20
```

`yahoo-validate` は主キー重複、OHLC矛盾、欠損、出来高ゼロ、異常な調整済みリターン、調整済み／未調整価格比率から推定した株式分割前後、銘柄一覧にあるのに価格を取得できないティッカーを検査します。

分析は予測モデルを学習しません。出来高0または価格が無効な行を取引不能として除外し、過去5営業日以内に高値が+5%へ到達した場面について、直前の1日・5日・20日騰落率、前日比出来高、5日対20日出来高比、20日高値接近度、日中値幅を集計します。最新候補はこれらの同日横断順位を組み合わせた説明可能なルールスコアであり、上昇確率ではありません。

保存先:

```text
data/yahoo/
├─ raw/batch=.../                 # 取得バッチ
├─ processed/equities_daily.parquet
├─ processed/analysis/latest_candidates.parquet
├─ processed/analysis/historical_patterns.parquet
├─ processed/quality/latest.parquet
└─ metadata/
   ├─ prime_universe.parquet
   ├─ ingestion_status.json
   ├─ analysis_latest.json
   ├─ quality_latest.json
   └─ market.duckdb
```

Windowsの日次実行用スクリプトは `scripts/run_yahoo_daily.ps1` です。取得、品質検査、分析を順番に実行し、`logs/yahoo-daily.log` に記録します。

Yahoo側のDuckDBには `equities_daily`、`prime_universe`、`latest_candidates`、`historical_patterns`、`quality_issues` ビューが作られます。

### GitHub ActionsによるPC不要の更新

`.github/workflows/momentum5d-daily.yml` は平日18:10（日本時間）にPython 3.12で次を実行します。

1. 単体テストと静的検査
2. Prime銘柄の直近365日をyfinanceから取得
3. 品質検査と+5%局面分析
4. 生の日足をGitHubへ保存せず、集計済みランキングJSONだけを非公開サイトへ送信

リポジトリのActions secretsには、サイト側で更新要求を検証する共有シークレット `DASHBOARD_UPDATE_TOKEN` が必要です。サイトの通常画面はChatGPTサインインとメールアドレス許可リストで保護し、更新APIはこの長いランダム値で保護します。

ActionsのcronはUTCで評価され、スケジュール実行は既定ブランチの最新版を使用します。祝日判定は曜日から推測せず、Yahooから有効な日足が返らない日は保存済みの最新日が維持されます。

### 従来のJ-Quantsコマンド

初回取得では、契約プランで参照可能な開始日を明示します。J-Quants APIには「このキーで参照可能な最古日」を返す仕様がないため、開始日は推測しません。

```bash
python -m app ingest --start 2016-01-01 --end 2026-07-31
```

`end` を省略すると日本時間の当日です。未来日は取得できません。営業日は曜日から推測せず、J-Quantsの公式取引カレンダーで判定します。公開前などで必須データが空の場合は成功扱いにせず、次回同じコマンドで再取得できます。

2回目以降:

```bash
python -m app update
python -m app update --end 2026-07-31
```

`update` は `equities_daily` の最終保存日の翌暦日から開始し、その範囲内の公式営業日だけを取得します。

品質検査:

```bash
python -m app validate
```

品質検査に `error` がある場合は終了コード2、実行自体の失敗は1、合格は0です。検査結果は `data/processed/quality/latest.parquet`、要約は `data/metadata/quality_latest.json` に保存されます。

状態確認:

```bash
python -m app status
```

各データセットの件数、最小／最大日、主キー重複グループ数、チェックポイント状態、契約プランで利用できなかったデータセットを表示します。`status` と `validate` はAPIキーなしでも実行できます。

バックテスト:

```bash
python -m app backtest
```

既定値は、過去252営業日以上を学習、20営業日ごとに再学習、上位20銘柄、20bpsの往復取引コスト、過去20日平均売買代金1,000万円以上です。

```bash
python -m app backtest \
  --start 2025-06-01 \
  --end 2026-04-30 \
  --min-train-days 252 \
  --retrain-every-days 20 \
  --top-n 20 \
  --min-turnover 10000000 \
  --transaction-cost-bps 20
```

モデルは標準化ロジスティック回帰をベースラインとして使用します。各テスト期間の学習データは、その時点で5営業日の結果が確定している日までに制限します。

共通オプションはサブコマンドより前に指定します。

```bash
python -m app --env-file .env.staging --verbose status
```

## ディレクトリ構造

```text
app/
├─ api/                 # V2クライアント、ページング、レート制限、再試行
├─ ingestion/           # 全量／差分取得と再開制御
├─ processing/          # V2短縮項目名から分析用項目名への正規化
├─ quality/             # 品質検査
├─ modeling/            # 特徴量、ラベル、ウォークフォワード検証
├─ storage/             # Parquet、チェックポイント、DuckDB
├─ config.py
└─ __main__.py          # CLI
tests/
data/
├─ raw/                 # APIレスポンスをページ単位で保持
├─ processed/           # 正規化・主キー一意化済みParquet
└─ metadata/            # DuckDB、チェックポイント、品質要約
logs/                   # 日別実行ログ
```

### raw

```text
data/raw/{dataset}/unit={日付または期間}/page-00000.parquet
```

APIのV2短縮項目名をそのまま残し、`__ingested_at`、`__endpoint`、`__page_number` を付加します。銘柄マスタと株価は同日時点のPrime銘柄だけを保存します。ページ保存後にチェックポイントを原子的に更新するため、停止後は次の `pagination_key` から再開します。保存とチェックポイント更新の間で停止した場合も、同じページを原子的に上書きするだけです。

### processed

```text
data/processed/{dataset}/year=YYYY/data.parquet
```

| データセット | 主キー | 内容 |
|---|---|---|
| `equities_daily` | `code, date` | 調整前／調整済みOHLC、出来高、売買代金、調整係数 |
| `listed_master` | `code, date` | 各営業日時点の銘柄、会社、業種、市場、信用区分 |
| `indices_daily` | `code, date` | 利用可能な指数OHLC |
| `topix_daily` | `code, date` | TOPIX OHLC（`code='TOPIX'` を付与） |
| `trading_calendar` | `date` | 公式営業日区分 |

既存年パーティションと新規データを結合し、主キー単位で後着データを採用してから原子的に置換します。保存後に重複が残る場合は失敗します。

### metadata

- `market.duckdb`: processed Parquetを読む同名ビューと `quality_issues`
- `checkpoints.json`: unitごとの状態、次ページ、次カーソル、行数、失敗内容
- `quality_latest.json`: 最新品質検査の件数と閾値
- `backtest_latest.json`: 最新バックテストの設定と評価指標

バックテスト出力:

```text
data/processed/backtest/latest_predictions.parquet
data/processed/backtest/latest_trades.parquet
data/processed/backtest/latest_equity_curve.parquet
```

DuckDBでは `backtest_predictions`、`backtest_trades`、`backtest_equity_curve` ビューとして参照できます。

DuckDB利用例:

```python
import duckdb

con = duckdb.connect("data/metadata/market.duckdb", read_only=True)
prices = con.sql("""
    SELECT date, code, adjusted_close, volume, turnover_value
    FROM equities_daily
    WHERE date >= DATE '2026-01-01'
    ORDER BY date, code
""").df()
```

過去時点のPrime投資対象集合は次のように取得できます。各日の市場区分スナップショットを使い、最新銘柄だけで過去を絞らないため、上場廃止や市場区分変更による生存者バイアスを避けられます。

```sql
SELECT code, company_name, market_code
FROM listed_master
WHERE date = DATE '2020-06-30';
```

## バックテスト定義

- シグナル時点: 営業日 `t` の引け後
- ラベル: `t+1`〜`t+5` の調整済み高値の最大値が `t` の調整済み終値比+5%以上
- エントリー: `t+1` の調整済み始値
- 利確: `t+1`〜`t+5` の高値がエントリー比+5%に到達した最初の日
- 未達時: `t+5` の調整済み終値で決済
- 特徴量: 当日までのリターン、ボラティリティ、移動平均乖離、日中レンジ、出来高・売買代金比、同日横断順位
- 学習: 過去データだけを使うウォークフォワード方式
- パージ: テスト開始前5営業日にラベル期間が重なる学習行を除外
- ポートフォリオ近似: 毎営業日に資金の1/5を新規シグナル群へ等配分

## 品質検査

`validate` は次を検査します。

- 全データセットの主キー欠損・重複
- 調整前／調整済みOHLCの部分欠損と `low <= open, close <= high`
- 全OHLC欠損（売買不成立、停止、欠損候補）
- 出来高0、出来高／売買代金の負値、価格あり・出来高欠損
- 調整済み終値の日次リターン閾値超過
- 調整係数0以下
- 調整係数が1以外の分割・併合イベント
- 分割日の未調整価格ギャップと調整係数の整合性
- 分割前後の調整済み終値の不連続
- 株価銘柄が同日の `listed_master` に存在するか
- 株価日が公式カレンダー上の東証営業日か

取引不成立日のOHLC・出来高・売買代金がNullになること、および2020-10-01が東証システム障害で終日停止だったことは公式仕様に従って扱います。閾値は `.env` の `ABNORMAL_RETURN_THRESHOLD` と `SPLIT_RATIO_TOLERANCE` で変更できます。

## テスト

```bash
python -m pytest
python -m ruff check app tests
```

テストは外部APIへ接続せず、認証ヘッダー、タイムアウト、再試行、ページング、再開、Parquet upsert、DuckDB、営業日選択、契約権限エラー、品質検査をモック／一時ディレクトリで検証します。

## 既知の制約

- `yfinance` は非公式クライアントで、YahooによるSLAや後方互換性はありません。失敗バッチはメタデータへ保存され、次回実行で再取得します。
- YahooのCSV・日足データには銘柄や地域によるライセンス制約があります。本システムは個人研究・非公開利用に限定します。
- Yahooの日足には売買代金がないため、分析用 `turnover_value` は `close × volume` の近似値です。
- 直近候補の `signal_score` は説明可能なルール順位であり、予測確率ではありません。売買推奨でもありません。
- 取得可能期間とデータ種別は契約プランに依存します。開始日は利用者が指定してください。
- バックテスト対象は各営業日時点のPrime銘柄に限定されます。流動性と標本数は短期シグナルの基礎検証に十分ですが、Standard／Growth銘柄へ結果を一般化することはできません。また、Primeでは5営業日以内+5%の陽性率が全市場より低くなる可能性があります。
- 日次の株価、銘柄スナップショット、TOPIX、指数を順に取得するため、長期間の初回取得は多数のAPIコールになります。`.env` のレート値を契約プラン上限より高くしないでください。
- `update` は要件どおり最終保存日の翌日から取得します。J-Quantsの調整済み過去値が将来の分割により遡及改定された場合、既存パーティションは自動再取得しません。一方、未調整値と各日の調整係数は保持するため、Phase 2でpoint-in-time調整系列を再構成できます。
- J-Quantsの調整は分割・併合が中心で、一部コーポレートアクションは反映されない場合があります。
- APIキーはログへ出しませんが、ログにはエンドポイント、日付、HTTP状態、失敗内容が残ります。
- 本データの利用・再配布はJ-Quantsの最新利用規約に従ってください。
- 日足OHLCだけでは利確価格への到達順序やギャップ約定を完全には再現できません。利確は指値価格で約定できたという近似です。
- ベースライン結果は将来収益を保証しません。ハイパーパラメータ探索、銘柄種別の厳密な除外、スリッページ・値幅制限・売買停止の詳細処理は今後の課題です。

## 仕様の根拠

実装時点（2026-07-27）に、次の公式情報を確認しています。

- [JPX: J-Quants API V2提供開始とAPIキー認証](https://www.jpx.co.jp/corporate/news/news-releases/6020/20260119.html)
- [J-Quants公式Pythonクライアント](https://github.com/J-Quants/jquants-api-client-python)
- [J-Quants公式サイト](https://jpx-jquants.com/)
- [yfinance公式ドキュメント: download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [yfinance公式GitHub](https://github.com/ranaroussi/yfinance)
- [Yahoo Finance: 履歴データのダウンロード](https://help.yahoo.com/kb/finance/historical-prices-sln2311.html)
- [Yahoo利用規約](https://legal.yahoo.com/xw/en/yahoo/terms/otos/index.html)

公式クライアントのV2実装で、ベースURL、`x-api-key`、`data`／`pagination_key`、`/equities/master`、`/equities/bars/daily`、`/markets/calendar`、`/indices/bars/daily`、`/indices/bars/daily/topix` および短縮項目名を照合しています。
