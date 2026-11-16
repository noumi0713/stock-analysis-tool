# stock-analysis-tool

このリポジトリは、今後使うデータ基盤だけに整理しています。

- `swing-data-120d-latest`: 日本株120営業日の最新サイト・分析データ。更新ごとに履歴を積み上げず、最新スナップショットだけを保持。
- `technical-data-7y`: 認証済み7年テクニカル特徴量。分割ParquetとSHA256付きmanifestを保持。
- `main`: 上記2データを維持する取得コード・品質検査・120日取得テストだけを保持。

旧サイト、旧バックテスト成果物、研究用STEP群、試作ブランチは削除済みです。

## 日足−2σ到達銘柄

120日データの更新時に、20日ボリンジャーバンド下限に安値が到達した銘柄を自動抽出します。
[最新一覧](https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest/bollinger_minus2.md) / [仕様・手動更新](swing_data/BOLLINGER_MINUS2.md)
