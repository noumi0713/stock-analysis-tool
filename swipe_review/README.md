# 掲示板100 スワイプ選別

## 目的
Yahoo掲示板ランキング当日100位までを母集団にし、各銘柄をチャートで確認しながら左右フリックで仕分けするモバイル向け画面です。

- 右フリック: 興味あり
- 左フリック: 興味なし
- 表示順: 5営業日騰落率の高い順
- 5営業日騰落率: 日本株120日データの直近6営業日 `adj_close` から画面側で計算
- 母集団入力: `swing-data-120d-latest/swipe_review_universe.json`
- 当日ランキングに存在する銘柄は、価格データ欠損でも母集団から除外せず末尾に残します
- 端末内には即時保存し、サーバー同期成功後は `swipe-decisions` ブランチへJSON保存します
- ChatGPT推奨銘柄は当日 recommendation JSON に入れると「おすすめ」バッジを表示します

## 入力データ
`swing-data-120d-latest` ブランチ:
- `swipe_review_universe.json`（当日の掲示板100位まで＋直近6営業日の調整後終値）
- `swipe_review_status.json`
- `stocks/<code>.csv`（表示中の銘柄だけ120日分を遅延読込）

母集団JSONが未生成の移行期間だけ、従来の `bbs_ranking_latest.csv` + 個別CSV読込へフォールバックします。

## ユーザー仕分け出力
`swipe-decisions` ブランチ:
- `swipe_review/data/YYYY-MM-DD.json`

例:

```json
{
  "date": "2026-09-18",
  "decisions": [
    {
      "date": "2026-09-18",
      "stock_code": "1605",
      "stock_name": "INPEX(株)",
      "bbs_rank": 12,
      "five_day_return_pct": 3.82,
      "decision": "interested",
      "recommended": false,
      "recorded_at": "2026-09-18T13:05:00.000Z"
    }
  ]
}
```

ChatGPTはこのJSONを読み、`interested` / `rejected` をユーザー選好として参照できます。売買判断そのものとは分離して扱います。

## おすすめ表示
`swipe-decisions` ブランチ:
- `swipe_review/recommendations/YYYY-MM-DD.json`

例:

```json
{
  "date": "2026-09-18",
  "recommendations": [
    {
      "stock_code": "1605",
      "reason": "5〜10営業日スイング候補"
    }
  ]
}
```

## Vercel
`/swipe` または `/` で画面を開きます。

サーバー保存にはVercelのサーバー側環境変数 `GITHUB_TOKEN` が必要です。権限はこのリポジトリの Contents read/write に限定してください。

任意:
- `GITHUB_REPO`（既定: `noumi0713/stock-analysis-tool`）
- `SWIPE_DATA_BRANCH`（既定: `swipe-decisions`）

公開URLで書き込みAPIを無防備に公開しないため、Vercel Deployment Protectionなどでこの画面自体へのアクセスを制限してください。ブラウザへGitHubトークンは渡しません。
