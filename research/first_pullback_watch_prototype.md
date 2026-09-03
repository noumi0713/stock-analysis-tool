# First-pullback watch prototype

Research-only prototype for the new discovery/watch workflow.

## Goal

1. Discover stocks from the **Minkabu relevance × Kabutan membership** universe, with IFIS access-rank input.
2. Store the discovery context and the reason/catalyst.
3. Re-evaluate each watched ticker every trading day.
4. Keep daily technical snapshots instead of making a one-day-only decision.
5. End each episode with one of:
   - `BUY_NOW`
   - `WAIT_FIRST_PULLBACK`
   - `PULLBACK_FORMING`
   - `FIRST_PULLBACK_SIGNAL`
   - `EXCLUDE_RAN_AWAY`
   - `INVALIDATED`

`FIRST_PULLBACK_SIGNAL`, `EXCLUDE_RAN_AWAY`, and `INVALIDATED` are terminal for the current episode.

## Important design rule

The discovery price/date and the running reference high are preserved. The system does **not** reset the episode every day. This prevents a stock that keeps rising without a pullback from being chased indefinitely.

## Prototype decision order

Every day:

1. Invalidate if catalyst/theme/trend has materially broken.
2. If a proper pullback was already formed, test for rebound confirmation.
3. Otherwise test whether a normal first pullback has formed.
4. If overheat has cooled without a price pullback (time correction), allow `BUY_NOW`.
5. If price has risen too far without a qualifying pullback and overheat is high, use `EXCLUDE_RAN_AWAY`.
6. Otherwise keep `WAIT_FIRST_PULLBACK`.

## Coarse prototype thresholds

These values are intentionally simple and are **not production parameters**.

- Buy-now overheat: `<= 20`
- Buy-now RSI14: `<= 65`
- Normal pullback depth: `3% to 12%`
- Run-away candidate: `+10% from discovery`, no 3% pullback, overheat `>= 40`
- Invalidation: `-15% from discovery` or materially below MA25
- Rebound signal: prior proper pullback + close above previous high + volume re-acceleration + acceptable overheat

These thresholds should be backtested as coarse ranges before any production adoption.

## Daily input format

```json
{
  "as_of": "2026-09-03",
  "candidates": [
    {
      "ticker": "9999.T",
      "name": "Example",
      "discovery": {
        "ticker": "9999.T",
        "name": "Example",
        "discovery_price": 1000,
        "discovered_date": "2026-09-03",
        "theme": "AI",
        "minkabu_relevance": 90,
        "kabutan_member": true,
        "ifis_rank": 8,
        "ifis_access_change_pct": 120,
        "catalyst_date": "2026-09-03",
        "catalyst_reason": "Example catalyst"
      },
      "technical": {
        "open": 1010,
        "high": 1060,
        "low": 1005,
        "close": 1040,
        "volume": 1000000,
        "rsi14": 68,
        "ma25": 980,
        "atr14": 35,
        "overheat_score": 32,
        "ifis_rank": 8,
        "ifis_access_change_pct": 120,
        "minkabu_theme_rank": 3,
        "material_valid": true,
        "theme_active": true
      }
    }
  ]
}
```

For an already-registered ticker, `discovery` can be omitted; only the new daily `technical` block is needed.

## Run

```bash
python research/first_pullback_watch_prototype.py --self-test
```

Or process a daily snapshot:

```bash
python research/first_pullback_watch_prototype.py \
  --input daily_snapshot.json \
  --state research/state/first_pullback_watch.json
```

The state JSON contains the full episode history for every ticker.

## Not included yet

This prototype intentionally does not scrape Minkabu, IFIS, Yahoo Finance, or news sites. Data-source acquisition should be implemented only with an allowed/authorized method. The state machine accepts normalized inputs regardless of where they come from.

Next validation steps:

1. Feed historical discovery events into the state machine.
2. Compare `BUY_NOW`, first-pullback entry, and run-away exclusion outcomes.
3. Measure 5/10/20-day return, MFE/MAE, hit rates, signal count, and year-by-year stability.
4. Only after that, decide production thresholds and UI integration.
