# Attention research pipeline v3

Research branch only: `research/first-pullback-watch-prototype`.
Production trading strategy remains frozen.

## Goal

Detect stocks where investor attention is starting to rise, verify why attention increased, then decide whether price timing is still favorable. IFIS and Minkabu attention are **discovery inputs**, not independent buy confirmations.

## Stage 1 — Attention discovery only

Inputs:

- IFIS attention increase TOP30, supplied manually as CSV. IFIS scraping remains prohibited.
- Minkabu POPULAR / RISING theme snapshot, supplied as CSV.
- Minkabu stock-theme relevance when available.
- Finalized 124-theme relevance master derived from Kabutan theme membership.

A stock is discovered only when:

1. it is in IFIS top N (default 30),
2. it belongs to a Minkabu POPULAR or RISING theme in the supplied snapshot,
3. the same stock-theme pair exists in the finalized Kabutan-derived 124-theme master.

No trading score or buy decision is created in this stage. IFIS attention and Minkabu attention are not double-counted.

Output: `attention_discovery_candidates.csv` and `attention_discovery_matches.csv`.

## Stage 2 — Material before technical

Every discovered stock requires a semantic material review using only information safely available at the attention snapshot.

Material classes:

- `STRONG`: reviewed positive catalyst passed the gate.
- `WEAK`: mixed, theme-sympathy, or weak catalyst.
- `NEGATIVE`: adverse catalyst; attention can be driven by bad news.
- `NONE`: no verified company-specific catalyst.
- `LOOKAHEAD_REJECT`: catalyst was not safely known at the snapshot.

Only `STRONG` can proceed to `BUY_NOW` or `WAIT_FIRST_PULLBACK` in the final classifier. Missing semantic reviews must block final acceptance testing.

This stage separates examples such as earnings upgrades, orders, policy benefits and specific partnerships from price-chasing, negative financing, scandals, or pure theme sympathy.

## Stage 3 — Entry timing

Technical monitor records:

- volume ratio vs 20-day average,
- optional float turnover when `float_shares` is supplied,
- RSI14,
- MA5 / MA25 / MA75,
- MA25 deviation,
- ATR14%,
- 1 / 5 / 10 / 20-day returns,
- upper-wick ratio,
- distance from 20-day high,
- overheat score used by the existing research prototype.

The system also records an `attention_price_sequence_proxy`:

- `ATTENTION_BEFORE_PRICE_OVERHEAT`
- `AMBIGUOUS`
- `PRICE_MOVE_PRECEDED_ATTENTION_OR_LATE`

This is a timing proxy, **not proof of causality**. Its purpose is to catch cases where price/volume had already run before the attention snapshot.

## Final three classes

### BUY_NOW — 今買う

Research prototype requirements include:

- material class `STRONG`,
- technical state `BUY_NOW` or `FIRST_PULLBACK_SIGNAL`,
- RSI <= 68,
- MA25 deviation <= 8%,
- 5-day return <= 10%,
- 20-day return <= 25%,
- volume ratio between 1.2x and 5x,
- upper-wick ratio <= 0.35,
- float turnover <= 60% when evaluable.

### WAIT_FIRST_PULLBACK — 初押し待ち

Strong material remains valid but current timing is not clean enough. This includes already-extended stocks, pullback-forming stocks, or cases where attention appears late relative to the price move.

### OVERHEAT_SKIP — 過熱・見送り

Includes weak/negative/absent materials, lookahead-unsafe evidence, invalidated technical states, fetch/data failures, or extreme overheat such as:

- RSI >= 80,
- MA25 deviation >= 15%,
- 10-day return >= 25%,
- 20-day return >= 40%,
- volume ratio >= 8x,
- long upper wick >= 0.50,
- daily float turnover >= 100% when evaluable.

These thresholds are **research parameters**, not changes to the frozen production strategy.

## Ranking rule

No composite buy score is created.

Sort in this order:

1. final class: BUY_NOW -> WAIT_FIRST_PULLBACK -> OVERHEAT_SKIP,
2. lower technical overheat,
3. lower positive MA25 deviation,
4. lower 5-day price rise,
5. original IFIS rank.

This intentionally ranks "attention is rising but price is least overheated" above "most viewed".

## Lookahead rules

- Current theme relevance master must not be retroactively applied to historical backtests.
- Catalyst data must be available at the discovery timestamp.
- Same-day catalyst dates without a timestamp are treated as ambiguous and rejected when the attention snapshot is earlier that day.
- All research outputs remain `RESEARCH_ONLY` until out-of-sample and forward validation are completed.
