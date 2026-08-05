# 日本株・短期上昇シグナル検知基盤 — Phase 1 + ベースライン検証

各営業日の引け後に利用可能な情報だけを使い、将来「5営業日以内に5%以上上昇する可能性」を推定するための基盤です。Phase 1の取得・保存・品質検査に加え、リーク検査済みの特徴量、ベースラインモデル、ウォークフォワード・バックテスト、日次ランキングを実装しています。本番運用モデルの選定・チューニングは今後のPhaseです。

現在の価格・出来高の取得源は `yfinance` です。個人研究・非公開利用だけを想定し、直近365日をローリング保存します。Prime銘柄コード一覧には保存済みJ-Quants銘柄マスタを使いますが、日足価格はYahoo Finance由来です。従来のJ-Quants取得コマンドとバックテスト実装は互換性のため残しています。

`yfinance` はYahoo公式ライブラリではなく、仕様変更・レート制限・データ訂正の可能性があります。全銘柄の日足は公開・再配布せず、利用前にYahooの最新利用条件を確認してください。スマホ画面用に、候補20銘柄のランキングと直近60営業日の日足だけを公開リポジトリへ保存します。

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
python -m app yahoo-ingest --full-refresh --intraday-session morning
python -m app yahoo-ingest --full-refresh --intraday-session close
python -m app yahoo-analyze --top-n 20
python -m app yahoo-backtest-sakata --start 2026-04-01 --initial-capital 1000000
python -m app yahoo-backtest-retail --start 2026-04-01 --initial-capital 1000000
```

`--intraday-session morning` は当日の5分足を9:00〜11:30で集計し、`close` は9:00〜15:30で集計します。取得済みの日足に当日の途中値が含まれていても削除し、5分足から再構成したOHLC・出来高だけで当日行を置き換えます。全Prime銘柄の70%以上を取得できなければ成功扱いにせず、次のクラウド実行で再試行します。

`yahoo-validate` は主キー重複、OHLC矛盾、欠損、出来高ゼロ、異常な調整済みリターン、調整済み／未調整価格比率から推定した株式分割前後、銘柄一覧にあるのに価格を取得できないティッカーを検査します。

分析は予測モデルを学習しません。出来高0または価格が無効な行を取引不能として除外し、過去5営業日以内に高値が+5%へ到達した場面を集計します。現在のランキングは、当日の出来高・売買代金を各銘柄の直前20営業日平均およびPrime内順位と比較し、株価上昇・陽線・上昇日出来高の一致を確認した銘柄を対象にします。市場全体が弱い日でも観測事実は表示し、市場地合いは別の注意情報として残します。スコアは純買越額や上昇確率ではありません。

個人投資家フローは「発見→理解→期待→安心→行動」を、当日までに取得済みの情報だけで次のように代理評価します。

- 発見: 出来高比、絶対騰落率、売買代金、20日高値への接近、3日前からの注目変化
- 理解容易性: 17業種の騰落率、上昇銘柄比率、業種トレンド（企業内容の理解そのものではない代理値）
- 期待: 初動の5日騰落、Prime内相対強度、業種トレンド、高値への接近
- 安心: 流動性、適度な値幅、20日移動平均付近、直近下落の安定性
- 行動: 出来高・売買代金の立ち上がり、上昇日の出来高比率、陽線
- 減点: RSI・短期騰落・出来高の過熱と、下落トレンド・移動平均割れ・高ボラティリティ

ニュース件数、SNS言及数、決算・上方修正、信用残は現在の取得データに含まれないため、スコアへ入れていません。「理解容易性」は同業種が同時に買われているかの代理であり、事業の説明しやすさを直接測ったものではありません。これらを取得していない状態で推測値を埋めることもしません。

注意を引く銘柄が個人投資家の選択集合に入りやすいという設計根拠には、異常出来高・極端な値動き・ニュースと個人投資家の買い行動を検証した [Barber and Odean (2008)](https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/allthatglitters_rfs_2008.pdf) と、検索量を注意の直接指標として短期的な価格圧力と反転を検証した [Da, Engelberg and Gao (2011)](https://doi.org/10.1111/j.1540-6261.2011.01679.x) を参照しています。現在は予兆ではなく、出来高・売買代金・価格方向が同時に強まった観測事実を優先し、過熱ペナルティで極端な高値追いを抑えます。

伝統的な酒田五法には普遍的な数値閾値がないため、本システムではバックテスト可能な決定規則へ定量化しています。たとえば長い実体は当日値幅の55%以上、星は35%以下、三山・逆三山の3つの山谷は35営業日内かつ価格差3.5%以内です。この閾値は本システム固有であり、公式な唯一の定義ではありません。定性的な型の分類は[OANDA証券「酒田五法」](https://www.oanda.jp/lab-education/dictionary/sakata_method/)と[大和証券「酒田五法」](https://www.daiwa.jp/glossary/YST2036.html)を確認しています。

`yahoo-backtest-retail` は現在の資金流入観測方式、従来の仕込み形状、酒田五法、個人投資家フロー単独、混合方式を、翌営業日始値エントリー、+5%利確、5営業日目終値決済、往復コスト0.2%、5分割資金の同一条件で比較します。資金流入観測方式については、損切りなしに加え、ブレイクアウト前20日高値、資金流入日安値、確定押し安値、アンカーVWAP、5・10・25日移動平均線、上昇トレンドライン、最も近い有効サポートの各損切りも比較します。損切り価格はサポートから0.5 ATR14下、同日に利確と損切りの両方へ触れた場合は損切り優先です。結果はJSON・CSV・Markdownを `outputs/retail_flow_backtest/` に保存します。

2026年8月4日時点の保存データによる検証結果は次のとおりです。開始資金は100万円、1日上位10銘柄です。

| 期間 | 方式 | 最終資産 | 収益率 | 勝率 | +5%到達率 | 最大DD |
|---|---|---:|---:|---:|---:|---:|
| 2025-11-04〜2026-03-31 | 従来仕込み | 1,053,776円 | +5.38% | 56.15% | 25.19% | -4.23% |
| 2025-11-04〜2026-03-31 | 採用混合 | 1,052,262円 | +5.23% | 56.92% | 24.81% | -3.81% |
| 2026-04-01〜2026-07-23 | 酒田五法 | 1,008,372円 | +0.84% | 53.33% | 18.10% | -2.28% |
| 2026-04-01〜2026-07-23 | 従来仕込み | 1,022,966円 | +2.30% | 55.24% | 24.29% | -1.03% |
| 2026-04-01〜2026-07-23 | 個人投資家フロー単独 | 933,944円 | -6.61% | 47.43% | 25.52% | -9.69% |
| 2026-04-01〜2026-07-23 | 採用混合 | 1,024,871円 | +2.49% | 56.19% | 25.71% | -0.79% |

採用混合は、過去期間では従来方式より収益率がわずかに低い一方、勝率と最大DDが改善し、後続期間では収益率・勝率・+5%到達率・最大DDがすべて改善しました。個人投資家フロー単独は+5%到達を拾っても損失側の尾が大きく、注意の増加をそのまま買い条件にすると遅すぎることを示しています。期間が短く、同じデータで設計判断も行っているため、この結果だけで将来性能や統計的優位性を主張しません。

保存先:

```text
data/yahoo/
├─ raw/batch=.../                 # 取得バッチ
├─ raw/intraday/batch=.../        # 当日5分足
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

`.github/workflows/momentum5d-daily.yml` は前場・後場の終値反映を待ち、平日11:50と15:50（日本時間）にPython 3.12で更新を開始します。GitHub側のスケジュール遅延・取りこぼしに備え、前場は12:10・12:30、後場は16:10・16:30にも再試行します。同じ日・同じセッションが完了済みなら、後続処理はデータ取得前に終了します。

1. 単体テストと静的検査
2. Prime銘柄の直近365日の日足を取得
3. 当日の5分足を前場11:30または大引け15:30まで集計し、当日の日足へ反映
4. 品質検査と+5%局面分析
5. 全銘柄の日足をGitHubへ保存せず、ランキングと候補20銘柄の直近60営業日だけを表示用JSONへ保存

表示用JSONは公開リポジトリの `dashboard-data/latest.json` に保存されます。スマホ画面自体はChatGPTサインインとメールアドレス許可リストで保護します。

ActionsのcronはUTCで評価され、スケジュール実行は既定ブランチの最新版を使用します。祝日などで当日5分足のカバレッジが不足した実行は失敗として記録されますが、公開済みの前営業日データは上書きしません。

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
- GitHub Actionsの時刻指定は厳密な開始時刻を保証しません。本システムは各セッション後に3回の実行機会を設け、最初に成功した結果を採用します。
- YahooのCSV・日足データには銘柄や地域によるライセンス制約があります。本システムは個人研究・非公開利用に限定します。
- Yahooの日足には売買代金がないため、分析用 `turnover_value` は `close × volume` の近似値です。
- 直近候補の `sakata_score` はOHLCの酒田五法パターンを評価するルール値であり、予測確率や売買推奨ではありません。既存画面との互換性のため `setup_score` に同じ値も保存します。
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
