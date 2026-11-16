# 日足−2σ到達銘柄の自動抽出

既存の日本株120日データ更新に組み込み済み。平日17:30・19:30 JSTの更新後に実行します。
GitHub Actionsの混雑や配信遅延で公開時刻は遅れる場合があります。場中の即時通知ではありません。

## 一覧を見る

- [最新の到達銘柄一覧](https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest/bollinger_minus2.md)
- [CSV](https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest/bollinger_minus2_latest.csv)
- [JSON](https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest/bollinger_minus2_latest.json)
- [取得状態・対象日・除外理由](https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest/bollinger_minus2_status.json)
- [手動更新](https://github.com/noumi0713/stock-analysis-tool/actions/workflows/swing-data-120d.yml): Run workflow。

## 判定ルール

- 「−2a」は日足ボリンジャーバンドの「−2σ」と解釈。
- 期間20営業日。当日を含む調整後終値の単純平均−2×母標準偏差（ddof=0）。
- 当日の調整後安値が下限以下なら到達。等値と下限割れも含む。
- 当日終値も下限以下か、新規到達（前営業日は各日の下限に非到達）かを別列で表示。
- 新規到達を優先表示し、その中で当日の終値×出来高による推計売買代金の降順。
- 東証プライム・スタンダード・グロース内国普通株式の120営業日適格データが対象。
  ETF・ETN・REITを除外。掲示板順位、RSI、テーマ、流動性の追加フィルターなし。
- 120営業日不足の新規上場、欠損、異常OHLC、ゼロ出来高、古いデータは除外し件数を明記。
  標準偏差ゼロの横ばい系列は−2σへの下落として扱わない。
- 計算は調整後の同一尺度で行い、表示するMA・下限は最新のclose/adj_closeで最新終値の円尺度に戻す。
  配当調整方式・期間・標準偏差の定義によって証券会社のチャートと差が生じる場合があります。

これは引け後の確定バンドと当日安値の比較です。場中で変動したバンドへの到達時刻は再現しません。
条件到達の監視一覧であり、買い推奨・底打ち確認・反発確率ではありません。

## 更新失敗と出力

最新の終了済み東証営業日、manifest、更新run_idを照合します。古いデータや更新失敗時は
`not_ready` を出し、最新CSVをヘッダーだけ、JSONのmatchesを空にして過去候補の誤表示を防ぎます。
一部欠損または一部銘柄で実行した場合は `partial`。対象数・判定数・除外数を併記します。
全銘柄の株価品質だけを使い、海外指数などの欠損は独立に扱います。

生成されるのは最新スナップショットです。日ごとの過去シグナル履歴は蓄積せず、Actionsの実行結果に30日分の入力証跡を残します。

```bash
python -m swing_data.publish --source swing_data/runtime --target public-data
# 既に公開用データがある場合の再計算
python -m swing_data.bollinger_screen --target public-data
python -m pytest tests/swing_data -q
```

標準設定の参考: https://www.bollingerbands.com/_files/ugd/58be43_b120ddf0184540608baf19e2c0ae2019.pdf
