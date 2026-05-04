# Changelog

Version history for laserfish, with performance findings and audit notes from
paper-trading runs. Numbers below are from paper-trading observations on
Hyperliquid, **not live capital** — read the audit section for context on what
the paper portfolio does and does not simulate.

---

## v2.6 — CRITICAL: fix paper-mode phantom-PnL bug (2026-05-04)

**Tag:** `v2.6-close-price-fix`

`HyperliquidExchange.close_position()` was passing `prices.get(symbol, 0.0)` to
the paper portfolio when `_live_prices()` failed (e.g., 429 rate limit).
The portfolio then computed `gross_pnl = signed_qty * (0 - entry_price)` —
booking a fake "profit" equal to the full short notional, or a fake "loss"
equal to the full long notional.

### Confirmed forensic evidence

On 2026-05-02 17:52:05 the bot logged `BNB: momentum reversed → closing sell`
on a 7.5817 BNB short opened at 618.22.  Equity jumped from $12,016 to $16,686
on the next reading — exactly +$4,670, which equals `7.5817 * 618.22` (the
short notional valued against a $0 close).  Independent MTM at the time of
audit showed the open positions were actually worth **-$347**, not the
+$6,719 the paper portfolio reported.

### Audit conclusion for prior runs

The "+33.6% over 3 days" headline from v2.4 was not real.  31 close events
fired across the run (Apr 25 – May 4), each with the same vulnerability.
Any one of them that hit a 429 during close would have booked a phantom gain
or loss.  All paper P&L numbers from v2.0 onward should be considered
suspect until re-run on v2.6.

### Fix

Fall back to the position's `entry_price` when no live price is available,
so the close is a no-op rather than a fake fill.  Log a warning so future
operators can spot this in real time.

```python
close_price = prices.get(symbol)
if not close_price or close_price <= 0:
    pos = self._paper_portfolio._positions.get(symbol)
    if pos is None:
        return
    close_price = pos.entry_price
    logger.warning("close_position(%s): no live price — using entry %.4f", ...)
self._paper_portfolio.close(symbol, close_price)
```

This was a pre-existing bug, not something introduced by v2.5.

---

## v2.5 — Realistic paper costs (2026-05-03)

**Tag:** `v2.5-realistic-costs` · **Commit:** `5da1792`

Add fee + slippage modeling to `trader/paper_portfolio.py` so paper-mode P&L
reflects what live execution would actually produce.

- Taker fee: 4.5 bps per fill (matches Hyperliquid 2026 schedule)
- Per-symbol slippage:
  - Majors (BTC/ETH/SOL): 3 bps
  - Mid-cap (BNB/AVAX/LINK/DOGE/XRP/ARB/OP/MATIC/LTC/ADA/DOT): 7 bps
  - Alts (WIF/HYPE/INJ/SUI/KPEPE/...): 15 bps fallback
- Slippage skews fill price in the adverse direction (buys above mid, sells below)
- New cost trackers: `_fees_total`, `_slippage_total` reported in summary
- Side normalization handles both `buy`/`sell` and `long`/`short`

### Why we added this

The previous paper portfolio filled at exact mark price with zero commission,
inflating gains. The audit (see end of this file) estimated that real-world
costs on the v2.4 run would have eaten **~$700–1,100** of the headline +$6,719
paper P&L over 3 days.

### Known issues still open

- `apply_paper_funding()` exists but isn't called from `scripts/run.py`. Open
  shorts don't accrue funding cashflow. Low impact (rates have been near zero)
  but should wire it up for completeness.
- No liquidation simulation — leverage tested only by what didn't break.
- All limit fills assumed; no partial-fill or fail-to-fill modeling.

---

## v2.4 — Monte Carlo + tuned grids + 429 fixes (2026-04-30)

**Tag:** `v2.4-mc-tuning` · **Commit:** `11101d5`

Built `scripts/monte_carlo.py` and `scripts/tune_grid.py` for offline strategy
validation, then fixed the rate-limit bursts that crashed the funding arb bot
during live restart loops.

### Monte Carlo results (300 random 30-day windows, 3 symbols, 4.4 years of data)

| Symbol | Win rate | Median Sharpe |
|--------|---------:|--------------:|
| BTC    | 98.6%    | 2.02          |
| SOL    | 100.0%   | 4.10          |
| ETH    | 87.3%    | 1.16          |
| **All**| **95.2%**| —             |

Saved as `models/monte_carlo.csv`.

### Grid tuning — auto-applied to scripts/run.py

Sweep over `spacing_pct × order_size_pct`, score = `median_sharpe × win_rate`:

| Symbol | Spacing | Order size | Score |
|--------|--------:|-----------:|------:|
| BTC    | 0.30%   | 5%         | 3.451 |
| ETH    | 0.50%   | 5%         | 1.434 |
| SOL    | 0.60%   | 3%         | 4.799 |

Tighter spacings won across the board because the regime filter validates
RANGE first — once in RANGE, tight grids fill more often without taking on
extra directional risk.

### Rate-limit and resilience fixes

- ccxt `enableRateLimit=True` + `rateLimit=200ms` on both perp/spot clients
- Pre-load market metadata once in `__init__` to avoid repeated
  `load_markets()` calls during fallback
- Disk-cached funding warmup (`models/funding_warmup_cache.json`, 8h TTL) so
  restarts don't re-fetch 25 symbols
- Non-blocking warmup: skip on first 429 instead of retrying for hours
- Fast-fail batch fallback: a 429 on the batch returns `[]` immediately so the
  caller's 600s sleep acts as backoff (was looping for 15+ min before)
- `_live_prices()` caches last-known closes so 429s don't cause equity to
  flicker between mark-to-market and entry-price-fallback
  (previously logged spurious 9.3% drawdowns)

---

## v2.3 — RL stack + multi-asset grid + regime stability filter (2026-04-29)

**Tag:** `v2.3-rl-multigrid` · **Commit:** `545044c`

Brought the RL stack online for BTC, expanded grids beyond BTC-only, and
added a 3-reading confirmation filter to stop regime flapping.

- BTC PPO agent trained walk-forward (12m train / 2m val / 1m test windows)
- Multi-asset grids: BTC + ETH + SOL with per-symbol spacing
- HMM regime tracker now requires `CONFIRM_N=3` consecutive readings before
  acting on a transition (kills flapping between RANGE/CRISIS/TREND)

### Observation

Even with v2.3 tunings, the BTC grid was still being closed on `CRISIS`
detections every 1–3 hours during volatile sessions, almost always with
`pnl=+0.00, trips=0`. This was the trigger for v2.4's tuning work.

---

## v2.2 — Grid trading for ranging markets (2026-04-28)

**Tag:** `v2.2-grids` · **Commit:** `355f07a`

Added the grid layer that opens N levels above and below center price during
RANGE regimes. Originally BTC-only; v2.3 expanded to ETH/SOL.

### Best observed grid run (pre-v2.3 BTC-only)

```
2026-04-29 16:15:20  Grid BTC | trips=10 | realized_pnl=+37.71
```

Ten round-trip wins between Apr 28–29 — the strongest grid stretch we've
captured. Required RANGE regime to persist for ~24h.

---

## v2.1 — Mean-reversion + funding carry (2026-04-27)

**Tag:** `v2.1-mr-carry` · **Commit:** `f2fd7f8`

Added two complementary entry types alongside the momentum strategy:

- **Mean reversion (MR):** counter-trend entries when price gaps below/above
  short-term volatility band
- **Funding carry:** short perp + (where available) long spot, capturing
  positive funding payments over time

Both can fire simultaneously with momentum but are sized smaller and capped
by `RiskManager` exposure limits.

---

## v2.0 — HMM regime detector + dynamic Kelly leverage (2026-04-26)

**Tag:** `v2.0-regime` · **Commit:** `5ccde4d`

Replaced the static-leverage v1 architecture with:

- HMM regime tracker (`trader.regime`) classifying RANGE / TREND / CRISIS
  every 5 minutes from BTC volatility + return features
- `LeverageEngine` applying a regime multiplier (RANGE 0.8x, TREND 1.5x,
  CRISIS 0.2x) on top of a Kelly-derived base leverage
- Strategies opt in by reading `regime_state.regime` and `regime_mult`

### Why this matters

v1 used a single leverage knob; in TREND it under-allocated and in CRISIS it
over-allocated. v2 dynamically adapts and prevents ruin during fast moves.

---

## v1 — Initial 5m perp scalper (2026-04-22)

**Tag:** `v1-initial` · **Commit:** `e4f380b`

Forked from defi-trader. 5-minute cross-sectional momentum on Hyperliquid
perps with funding-rate gating, Transformer-based ONNX classifier, and
basic risk caps. No regime awareness, no grids, no MR/carry.

---

## Audit: paper-mode honesty (added 2026-05-03)

This section captures what `trader/paper_portfolio.py` did and didn't simulate
across versions, so future maintainers can interpret historical numbers
correctly.

### What WAS modeled

- Position margin and cash accounting
- Mark-to-market unrealized P&L
- Funding payments via `apply_funding()` (but not always called — see below)
- Take-profit / stop-loss bracket fills

### What WAS NOT modeled (pre-v2.5)

| Cost | Real-world impact |
|------|-------------------|
| Trading fees (taker 4.5 bps + maker 1.5 bps) | ~0.07–0.09% round-trip |
| Slippage / spread | 3–25 bps per leg, depending on symbol |
| Failed limit fills | Strategy-dependent |
| Liquidation engine | Critical at >3x leverage |
| Funding payments on momentum/MR positions | `apply_paper_funding()` only called from `training_loop.py`, not `run.py` |

### Estimated correction for v2.4 paper run

- Paper headline gain over 3 days: **+$6,719 (+33.6% on $20K notional)**
- Estimated real-world drag from fees + slippage: **~$700–1,100**
- Realistic estimate: **~+$5,600–6,000 (+28–30%)**

Still strong but **3 days is far too short a sample** to draw conclusions
about edge persistence.

### What v2.5 fixes

The cost model in v2.5 deducts fees and slippage on every paper fill, so
P&L going forward is comparable to live execution within ~10 bps.

---

## Forward roadmap

Open items, ranked by expected lift:

1. **Wire `apply_paper_funding()` into the run loop** — small but trivial.
2. **Add liquidation simulation** — currently leverage caps are unenforced
   in paper mode.
3. **Investigate why grids close on CRISIS with `trips=0`** — frequent
   "open at center, mark crosses ±1%, close, reopen" cycle suggests the
   close trigger may be too tight or fires before the first level fills.
4. **Run side-by-side bots** (v2.4 free-cost vs v2.5 realistic-cost) for
   1–2 weeks to measure the gap empirically.
5. **Investigate funding arb dry spell** — 50+ hours, 0 signals. Per-asset
   thresholds may need re-calibration if rates stay structurally lower
   than the 2024 paper's calibration period.
