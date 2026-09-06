# Theme ABC Dashboard

Research/feature implementation for the 124-theme daily ranking dashboard.

## UI

`index.html` renders five tabs:

- TOP10: current 124-theme ranks 1-10
- A: current Top10 and previous trading day also Top10
- B: current rank 11-30 and Top10 at least once in the previous five trading days
- C: current rank 31+ and Top10 at least once in the previous five trading days
- Standalone: individual stocks outperforming their associated themes while broad theme rises are excluded

Each theme expands to show the top five driver stocks.

## Machine-readable data

Primary endpoint: `data/chatgpt_snapshot.json`.

The snapshot contains `market_date`, `updated_at`, quality flags, TOP10/A/B/C, standalone ranking, and `theme_index` for all eligible themes. ChatGPT or another client should require `quality.certified=true` and check `market_date` before analysis.

Split files (`top10.json`, `a.json`, `b.json`, `c.json`, `standalone.json`) are generated for lighter reads.

## Daily refresh

The default-branch launcher runs at 16:20, 17:20 and 18:20 JST on weekdays. The later two runs are retries for delayed Yahoo daily data. The generator has a coverage gate and the launcher has a Tokyo Stock Exchange session freshness gate. On a failed gate, the previous certified snapshot is preserved.

## Isolation

The dashboard feature lives on `feature/theme-abc-dashboard-auto`. The production trading strategy, signal thresholds, exits, position sizing and ranking rules are not changed by this feature.
