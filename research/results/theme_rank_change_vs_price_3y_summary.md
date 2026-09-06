# Theme rank change vs price: 3-year summary

Period: 2023-08-21 to 2026-08-21, 124 themes, 734 trading days.

Positive rank delta means the theme ranking improved (example: 50th -> 20th is +30).
The same daily 124-theme score is used: price strength 25%, turnover inflow 25%, breadth 20%, relative strength 15%, persistence 15%.
Theme price returns are median constituent returns.

## Correlation: rank improvement vs returns

### 1-day rank change
- Same-day theme return: Pearson 0.1821, Spearman 0.2252
- Forward 5d: Pearson 0.0005, Spearman 0.0007
- Forward 10d: Pearson -0.0003, Spearman -0.0001
- Forward 20d: Pearson -0.0022, Spearman -0.0021

### 5-day rank change
- Same-day theme return: Pearson 0.0995, Spearman 0.1189
- Forward 5d: Pearson -0.0015, Spearman -0.0051
- Forward 10d: Pearson -0.0000, Spearman -0.0038
- Forward 20d: Pearson -0.0058, Spearman -0.0032

## 1-day rank-change buckets

| Rank move | N | Same-day mean | Next 5d mean | Next 10d mean | Next 20d mean |
|---|---:|---:|---:|---:|---:|
| Improve 20+ | 15,105 | +0.3677% | +0.1926% | +0.4429% | +0.9190% |
| Improve 10-19 | 10,879 | +0.1297% | +0.0871% | +0.3184% | +0.8286% |
| Improve 5-9 | 7,853 | +0.0798% | +0.1324% | +0.3237% | +0.7599% |
| Flat -4 to +4 | 19,907 | -0.0289% | +0.1351% | +0.2977% | +0.8481% |
| Worsen 5-9 | 8,302 | -0.1089% | +0.0674% | +0.2615% | +0.7932% |
| Worsen 10-19 | 11,177 | -0.1517% | +0.0931% | +0.3122% | +0.8918% |
| Worsen 20+ | 15,313 | -0.3214% | +0.1860% | +0.4530% | +0.9593% |

## 5-day rank-change buckets

| Rank move | N | Same-day mean | Next 5d mean | Next 10d mean | Next 20d mean |
|---|---:|---:|---:|---:|---:|
| Improve 20+ | 27,453 | +0.1417% | +0.1456% | +0.3824% | +0.8710% |
| Improve 10-19 | 7,817 | +0.0325% | +0.1182% | +0.3699% | +0.7888% |
| Improve 5-9 | 4,380 | +0.0171% | +0.1584% | +0.4683% | +0.9319% |
| Flat -4 to +4 | 8,873 | -0.0077% | +0.0759% | +0.3572% | +0.8964% |
| Worsen 5-9 | 4,473 | -0.0458% | +0.1687% | +0.4134% | +0.9450% |
| Worsen 10-19 | 7,879 | -0.0391% | +0.0829% | +0.3292% | +0.8953% |
| Worsen 20+ | 27,165 | -0.1466% | +0.1687% | +0.3894% | +0.9708% |

## Top-N transitions

### Enter Top 10
- Next 5d: N=3,230, mean +0.2435%, median +0.2949%, positive 55.70%
- Next 10d: N=3,217, mean +0.6141%, median +0.6717%, positive 60.40%
- Next 20d: N=3,171, mean +1.2466%, median +1.1988%, positive 63.20%

### Exit Top 10
- Next 5d: N=3,220, mean +0.2121%, median +0.3274%, positive 56.52%
- Next 10d: N=3,207, mean +0.6215%, median +0.6848%, positive 60.18%
- Next 20d: N=3,161, mean +1.2882%, median +1.2395%, positive 62.64%

### Enter Top 20
- Next 5d: N=5,412, mean +0.2362%, median +0.2951%, positive 55.97%
- Next 10d: N=5,388, mean +0.5305%, median +0.6036%, positive 59.89%
- Next 20d: N=5,316, mean +1.0505%, median +1.0431%, positive 62.21%

### Exit Top 20
- Next 5d: N=5,392, mean +0.2321%, median +0.2946%, positive 55.88%
- Next 10d: N=5,368, mean +0.5377%, median +0.6203%, positive 59.67%
- Next 20d: N=5,296, mean +1.1184%, median +1.1035%, positive 61.69%

## Interpretation

Rank improvement is clearly associated with the price move occurring at the same time because price strength, breadth, relative strength and turnover are inputs to the rank itself. However, raw rank change alone has essentially zero linear or monotonic correlation with forward 5/10/20-day returns. Large rank improvement therefore works as an attention/momentum-state indicator, not a standalone forward-return predictor.

The more useful feature remains the rank level / state: entering Top 10 has better forward 10/20-day returns than the all-theme baseline from the separate daily-score test, but simply measuring how many positions the theme rose adds little predictive information.

Static current theme membership is applied historically; this is not a point-in-time constituent backtest.
