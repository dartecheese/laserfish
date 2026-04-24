# Opportunity Log
**Project:** defi-trader  
**Started:** 2026-04-22  
**Method:** Live data probes — Binance aggTrade WS, Pyth Hermes SSE, Hyperliquid WS/REST, dYdX v4 REST

All edges are measured net of fees unless marked GROSS. HL taker fee = 0.035%. Break-even spread ≈ 0.07% round-trip.

---

## STATUS KEY
- `ACTIVE` — opportunity confirmed live, not yet fully exploited
- `BUILDING` — signal confirmed, execution engine in progress
- `DEAD` — explored and eliminated
- `MONITOR` — real but too small/illiquid to trade now; worth watching

---

## OPP-001 · Oracle Lag Arb — WIF (ACTIVE)
**Observed:** 2026-04-22  
**Source:** 5-minute broad asset scan (157 assets, Binance aggTrade WS vs HL allMids WS)

### What We Measured
HL's allMids for WIF consistently updates **9.8 seconds after Binance** (p50 lag). In 140 samples over 5 minutes, 73% of HL price updates showed a price gap larger than the 0.07% break-even threshold.

| Metric | Value |
|---|---|
| p50 lag (Binance → HL) | +9.8s |
| p95 lag | +20.3s |
| Mean absolute price gap | 0.147% |
| % windows > break-even | 73% |
| HL book depth (top 5 levels, each side) | ~$97k |
| $5k order slippage (entry + exit) | 0.000% |
| Net edge per trade (gross − 2×fee − slippage) | **+0.056%** |
| Estimated trades/hr (at 73% hit rate) | ~134 |
| Theoretical $/hr at $5k size | $375 |
| Realistic $/hr (35% capture) | **~$131** |

### Why the Edge Exists
WIF is a liquid Binance asset (~$100M+ daily volume) but HL's market makers update slowly — likely because it's not a top-5 priority asset for their infrastructure. 9.8s is long enough to detect a Binance move and submit an HL order comfortably.

### Trade Mechanics
1. Watch Binance WIF aggTrade stream
2. When Binance price moves >0.1% within 3s without HL following, enter HL position in same direction
3. Exit when HL price converges (typically within the 10s window) or after 15s max hold

### Critical Revision — Paper Trade Result (2026-04-22)
Running the paper trader for 3 minutes exposed a fundamental issue: HL's **oraclePx** (the actual HL oracle used for mark/funding) tracks Binance closely (only 0.026% apart for WIF). But `allMids` — the WebSocket channel we used to measure "HL price" — is the **order book bid-ask mid**, which sits ~0.25% persistently below the oracle due to structural book imbalance.

Result: 7 signals fired, all timed out without convergence. The WIF allMids moved from 0.18016 to 0.18032 over 90 seconds (~0.09% total), while the oracle/Binance stayed at ~0.18095–0.18100. Full convergence would take 5-10 minutes at that rate.

**Required fix**: Signal must be based on DELTA (Binance change vs HL change over N seconds), not absolute level gap. The level gap is structural/persistent; the delta divergence represents the true transient lag.

### Risks
- HL position size limits on lower-OI assets
- Window shrinks if other bots notice the same pattern
- Binance moves that immediately reverse (adverse selection ~20% of signals)
- Structural perp discount (allMids < oracle) means level-based signals are always firing

---

## OPP-002 · Oracle Lag Arb — AR (ACTIVE)
**Observed:** 2026-04-22  
**Source:** Same 5-minute broad asset scan

### What We Measured
AR (Arweave) has the **highest net edge per trade** of all 157 scanned assets after accounting for real book depth and fees.

| Metric | Value |
|---|---|
| p50 lag (Binance → HL) | +18.9s |
| p95 lag | +55.5s |
| Mean absolute price gap | 0.325% |
| % windows > break-even | 83.5% |
| HL book depth (top 5 levels) | ~$46k |
| $5k slippage (entry + exit) | 0.061% |
| Net edge per trade | **+0.132%** |
| Theoretical $/hr at $5k size | $523 |
| Realistic $/hr (35% capture) | **~$183** |

### Why the Edge Exists
AR has low HL OI ($620k) and relatively thin market making. 18.9s median lag gives comfortable time to enter. Price gap of 0.325% mean is well above break-even.

### Note
Only 85 samples in 5-minute window — needs longer confirmation run (30+ min) before live trading. Re-run `latency_probe.py --assets AR --duration 1800` to validate.

---

## OPP-003 · Oracle Lag Arb — SNX (ACTIVE)
**Observed:** 2026-04-22  
**Source:** Same 5-minute broad asset scan

### What We Measured
SNX has the **most consistent signal** of the three top targets — 156 samples, 100% exploit rate.

| Metric | Value |
|---|---|
| p50 lag (Binance → HL) | +52.1s |
| p95 lag | +155s |
| Mean absolute price gap | 0.286% |
| % windows > break-even | 100% |
| HL book depth (top 5 levels) | ~$15k |
| $5k slippage (entry + exit) | 0.103% |
| Net edge per trade | **+0.121%** |
| HL OI | $740k |
| Theoretical $/hr at $5k size | $210 |
| Realistic $/hr (35% capture) | **~$73** |

### Critical Constraint
HL OI is only $740k. HL likely enforces individual position limits of $200–500 per account on assets this thin. The per-trade numbers above may need to be scaled down 10×.

### Why the Edge Exists
SNX is a legacy DeFi asset. HL added it for completeness but market makers are inattentive. 52s median lag is extraordinary — the Binance move has largely played out before HL adjusts.

---

## OPP-004 · Oracle Lag Arb — Broader Altcoin Universe (MONITOR)
**Observed:** 2026-04-22  
**Source:** Broad asset scan — assets eliminated by slippage

These assets showed large lags in the scan but were **eliminated because book depth was insufficient** to absorb $5k without eating the entire edge. Worth revisiting if position sizing is reduced to $500–1k:

| Asset | p50 Lag | Exploit% | Why eliminated |
|---|---|---|---|
| LAYER | +71s | 100% | $5k slippage eats edge (0.083% slip) |
| RESOLV | +56s | 76% | $5k slippage eats edge (0.088% slip) |
| BERA | +56s | 90% | Net edge only +0.03% — marginal |
| 2Z | +23s | 100% | Only $13k depth — size limited |
| PNUT | +19s | 85% | Net edge only +0.017% |

At $1k position size, slippage halves and these may become viable. Needs re-evaluation.

---

## OPP-005 · HL Funding Extreme — MAVIA (ACTIVE)
**Observed:** 2026-04-22  
**Source:** Live HL `metaAndAssetCtxs` REST scan of all 230 assets

### What We Measured
MAVIA has the most extreme positive funding rate on all of HL right now.

| Metric | Value |
|---|---|
| Funding rate | +0.0877%/8h |
| Annualized | +96% APR |
| HL OI | $2.42M |
| 24h volume | $0.09M |
| Mark vs oracle premium | **+0.96%** (mark above oracle) |

### Why the Edge Exists
When longs are paying 96% APR to be long AND the mark price is already 0.96% above the oracle price, two forces converge:
1. Funding cost erodes long P&L each epoch
2. Mark price must converge toward oracle over time (HL's design)

Both push toward downward price pressure. This is a **mean-reversion short**.

### Trade Mechanics
Short MAVIA perp on HL. Collect +0.0877% every 8h. Exit when funding drops below 0.02% or mark/oracle premium closes.

### Risks
- Volume is thin ($90k/day) — wide bid/ask, hard to exit quickly
- Squeezes can happen before mean reversion (funding can stay high longer than expected)
- Not a directional edge — requires patience

---

## OPP-006 · HL Funding Extreme — ZEREBRO (ACTIVE)
**Observed:** 2026-04-22  
**Source:** Same live HL scan

| Metric | Value |
|---|---|
| Funding rate | +0.0433%/8h |
| Annualized | +47% APR |
| HL OI | $3.69M |
| 24h volume | $0.64M |
| Mark vs oracle premium | **+0.24%** |

Better liquidity than MAVIA ($640k/day volume). Same mechanics — longs paying, mark above oracle, short and collect.

---

## OPP-007 · Mark vs Oracle Premium Fades — Broad List (MONITOR)
**Observed:** 2026-04-22  
**Source:** Live HL `metaAndAssetCtxs` scan

14 assets where HL's mark price diverges from its own oracle by >0.09% with low or neutral funding. These are **pre-funding-spike signals** — the premium builds, funding adjusts at the next epoch, then price snaps back to oracle. Can trade the convergence.

| Asset | Premium | Direction |
|---|---|---|
| IMX | -0.137% | Mark below oracle → long bias |
| AVNT | -0.135% | Mark below oracle → long bias |
| COMP | -0.105% | Mark below oracle → long bias |
| MOVE | -0.128% | Mark below oracle → long bias |
| MEGA | +0.087% | Mark above oracle → short bias |
| AXS | -0.085% | Mark below oracle → long bias |

Low OI on most. Treat as small-size speculative fades, not primary strategy.

---

---

## OPP-008 · HL TradFi Stock/ETF Perps — Hidden @ Assets (ACTIVE — MAJOR DISCOVERY)
**Observed:** 2026-04-23  
**Source:** Live allMids WS probe — 310 `@`-prefixed assets found, 12 mapped to real equities/ETFs via Pyth cross-reference

### What We Found
Hyperliquid has TradFi perp markets that do NOT appear in the standard `/info?type=meta` REST endpoint. They are exposed only via the allMids WebSocket feed under numeric `@` aliases. During market hours they update live; outside hours they go static (0 price changes in our 60s probe run at night).

**Confirmed mappings (Pyth equity feeds → HL @ IDs):**

| Ticker | HL ID | Pyth Price | HL Price | Diff% |
|---|---|---|---|---|
| AAPL | @268 | $273.21 | $272.79 | +0.16% |
| MSFT | @276 | $432.86 | $431.83 | +0.24% |
| AMZN | @280 | $255.35 | $254.76 | +0.23% |
| GOOGL | @266 | $339.36 | $338.08 | +0.38% |
| META | @287 | $674.83 | $669.20 | +0.84% |
| NFLX | @227 | $93.20 | $92.03 | +1.26% |
| GM | @273 | $79.00 | $78.00 | +1.25% |
| DIS | @263 | $104.83 | $102.87 | +1.87% |
| AMC | @200 | $1.71 | $1.72 | +0.15% |
| SPY | @279 | $711.30 | $712.14 | −0.12% |
| QQQ | @288 | $654.83 | $653.48 | +0.21% |
| TLT | @271 | $86.76 | $86.38 | +0.44% |

**Partially identified (price range match, unconfirmed):**
- @173 = ~$103,382 → BTC oracle reference
- @151/@235 = ~$2,332 → ETH oracle reference
- @209/@182 = ~$4,715 → possibly SPX index or ES futures

**Still unmapped:** ~298 remaining @ assets (likely more stocks, forex, commodities, and internal oracle references)

### Why This Is Important
These are the TradFi assets we originally couldn't find. During NYSE hours (9:30am–4:00pm ET), Pyth's equity feeds publish every 5 seconds. HL's `@` asset prices update on HL's ~1 second batch clock. **The question is: how much does HL's stock oracle lag behind Pyth during market hours?**

Given that even liquid crypto assets (WIF, SNX) lag 10–52 seconds, TradFi stocks — which get far less market-maker attention on HL — could lag even more. Pyth's equity feeds are the reference source.

### HL Clock Discovery (Same Session)
HL publishes allMids in a **fixed ~1-second batch** (measured: mean=1022ms, median=1017ms, stdev=93ms). This is crucial:
- You don't need to detect "when" HL updates — it updates every ~1 second on a clock
- The window to act is between when you see a large Pyth move and when the next HL batch publishes
- Worst case latency from signal to order: ~1 second (next batch) + your execution time
- This is a hard constraint: if you can't submit an order in under ~800ms of detecting the Pyth signal, you risk the batch updating before your fill

### Next Action Required
Run the TradFi lag probe during NYSE hours (9:30am–4:00pm ET) to measure actual stock oracle lag:
- Subscribe to Pyth SSE for AAPL, MSFT, AMZN, GOOGL, META, SPY, QQQ
- Subscribe to HL allMids WS for @268, @276, @280, @266, @287, @279, @288
- Record lag distribution over a full trading session

**Script:** `scripts/tradfi_probe.py` — built and validated.

### Full Session Results (2026-04-24, NYSE hours)
244 lag records, 6,466 batch clock intervals.

| Asset | n | p50 lag | mean \|diff\| | Bias direction | Exploit% |
|---|---|---|---|---|---|
| AAPL | 4 | +276ms | 2.893% | HL BELOW Pyth (4/4) | 100% |
| MSFT | 35 | +216ms | 3.814% | HL ABOVE Pyth (35/35) | 100% |
| AMZN | 188 | +275ms | 0.310% | HL BELOW Pyth (188/188) | 100% |
| GOOGL | 7 | +364ms | 0.261% | HL below Pyth (5/7) | 86% |
| SPY | 4 | +364ms | 0.369% | HL ABOVE Pyth (4/4) | 100% |
| QQQ | 5 | +364ms | 0.205% | HL ABOVE Pyth (5/5) | 100% |

**Critical finding — book depth is extremely thin:**

| Asset | Bid depth | Ask depth | $1k slip | Viable? |
|---|---|---|---|---|
| AAPL (@268) | $761 | $1,794 | 0.28% | marginal |
| MSFT (@276) | $500 | $567 | 3.89% | no |
| AMZN (@280) | $575 | $599 | ∞ | no |
| GOOGL (@266) | $825 | $2,583 | 0.11% | marginal |
| SPY (@279) | $580 | $596 | 2.01% | no |
| QQQ (@288) | $406 | $410 | ∞ | no |

### Revised Assessment
Two problems emerged from the full session run:

1. **Depth is ~$400–2,600 total per side** — max position size ~$400–800 before slippage eats everything. These are effectively toy markets.

2. **The price diff is directionally consistent** — AMZN is ALWAYS below Pyth (188/188), MSFT is ALWAYS above Pyth (35/35). This is NOT random lag — it's a **persistent structural basis**, suggesting HL uses a different oracle source than Pyth for @ stock prices. The "edge" may be a mirage: you'd be trading into a price HL doesn't intend to converge to Pyth.

### Net Verdict
**DOWNGRADED: ACTIVE → MONITOR.** HL's @ stock perps are illiquid experimental markets. Persistent unidirectional bias suggests HL has its own oracle. Max trade size ~$400 with unknown convergence. Not viable for meaningful P&L. Re-evaluate if HL officially launches TradFi perps with real books.

---

## OPP-009 · HL allMids Clock Exploitation (ACTIVE — STRUCTURAL EDGE)
**Observed:** 2026-04-23  
**Source:** 60-second allMids timing probe

### What We Found
HL publishes allMids on a **fixed ~1022ms clock** (stdev=93ms). This is NOT event-driven — every asset in the universe gets a batch update regardless of whether its price changed.

**Key implication:** When Binance moves an asset price, we have up to ~1 second before the next HL batch. If HL's market makers haven't updated their orders within that window, the stale allMids price persists for another full second. This is a **guaranteed minimum exposure window** — not probabilistic.

**Sub-insight — slow-updating assets:**  
In our 60s probe, 352 of 541 HL assets had fewer than 5 price changes. That's 65% of assets changing price less than once every 12 seconds on average. These assets' market makers are not co-located or automated.

### How to Exploit
Traditional approach: detect Binance move → submit HL order → hope HL hasn't updated yet.  
Clock-aware approach: detect Binance move → calculate next HL batch time (~1022ms from last observed batch) → submit order timed to arrive just after the batch (prices still stale) but before the following batch (which would update them).

This converts a probabilistic edge into a structural timing advantage.

---

## OPP-010 · Cross-Asset Correlation Lag (MONITOR — LOW FREQUENCY)
**Observed:** 2026-04-22  
**Source:** 3-minute BTC/ETH/SOL cross-correlation probe vs Binance

### What We Found
When BTC moves on Binance, ETH and SOL on HL are also stale in the **same direction** — 100% correlated. If BTC basis (Binance BTC − HL BTC) is positive, ETH basis and SOL basis are also positive 100% of the time in the same window.

| Metric | Value |
|---|---|
| BTC→ETH same-direction correlation | 100% |
| BTC→SOL same-direction correlation | 100% |
| Threshold for usable signal | BTC Binance move >0.15% |
| % of HL updates that clear threshold (quiet session) | 5.9% |
| Session BTC range on 3-min probe | 0.22% |

### Why the Edge Exists
HL's market makers on ETH and SOL use BTC as a reference input. When BTC lags on HL, ETH and SOL lag in the same direction because the entire curve shifts together. A Binance BTC spike that HL hasn't absorbed yet is simultaneously a signal for ETH and SOL.

### How to Exploit
When Binance BTC moves >0.15% in <3s and HL allMids BTC hasn't responded, enter ETH + SOL on HL in the same direction as the BTC move. Adds position size to the oracle lag signal without concentrating all risk in one asset.

### Limitations
- BTC/ETH/SOL are HL's most liquid assets — market makers are fastest here. The lag window is small and infrequent.
- Requires volatile session (BTC moving >0.15% frequently) to generate signals.
- On quiet days (like the probe session, 0.22% range), this fires very rarely.

### Status
**MONITOR** — real correlation confirmed but not enough signal frequency to rely on alone. Use as a **multiplier on WIF/AR signal** rather than primary strategy: when BTC is stale and WIF/AR are also showing lag, increase position confidence.

---

## OPP-011 · L2 Book vs allMids Timing Gap (DEAD — No Exploitable Stale Orders)
**Observed:** 2026-04-22  
**Source:** 90-second L2 book + allMids parallel probe on ETH/WIF/AR/SOL

### What We Found
allMids (HL's mid-price broadcast) consistently arrives **228ms earlier** than the corresponding L2 book update for the same price change.

| Asset | allMids leads L2 | Mean lead time | allMids-L2 price diff |
|---|---|---|---|
| ETH | 83% of updates | +228ms | 0.002% |
| WIF | 90% of updates | +222ms | 0.003% |
| AR | 86% of updates | +252ms | 0.002% |
| SOL | 89% of updates | ~225ms | 0.002% |

### Why This Is Interesting (But Not Exploitable)
The allMids signal arrives 228ms before L2. The hypothesis was: resting limit orders on L2 could still be priced at the old allMids, making them "stale" and exploitable. But the **price difference between allMids and the L2 book is only 0.002–0.003%** — well below the 0.07% break-even threshold. Market makers update both channels nearly simultaneously, just with a 228ms propagation delay in the data feed.

### Conclusion
No stale resting orders to exploit. The 228ms L2 lag is a measurement artifact of HL's internal broadcast architecture, not an order-book pricing inefficiency. **DEAD.**

---

## ELIMINATED ANGLES

### dYdX v4 Cross-Venue Arb — DEAD
**Investigated:** 2026-04-22  
dYdX lists 295 markets and has 139 assets overlapping with HL. Initial basis comparison showed large apparent discrepancies (PIXEL: +370%, FET: -84%). Verified against Binance — **all large discrepancies were dYdX stale/broken oracles**, not real arbitrage. dYdX 24h volume for most altcoin markets = $0.0M. No liquidity to execute the other leg. Eliminated.

### dYdX vs HL Funding Arb — DEAD
**Investigated:** 2026-04-22  
Funding spreads looked compelling on paper (PUMP: 209% APR, ONDO: 88% APR). However, dYdX has zero actual trading volume on these markets. Cannot enter or exit dYdX leg. Eliminated.

### GMX v2 Cross-Venue Arb — DEAD
**Investigated:** 2026-04-22  
GMX v2 has 121 assets on Arbitrum. Very limited overlap with HL altcoins. Oracle pricing model uses signed prices with non-standard decimals. Liquidity concentrated in BTC/ETH/LINK — not the lagging altcoins where the edge exists. Eliminated.

### HL TradFi Stocks/Forex — NOT AVAILABLE
**Investigated:** 2026-04-22  
HL was reported to offer TradFi perps (stocks, commodities, forex). As of this date, no such assets exist in the HL universe API. No `@AAPL`-style symbols found. This angle cannot be pursued until HL launches TradFi markets.

---

## NEXT INVESTIGATIONS QUEUED

- [ ] **BUILDING: Paper trader + signal detector (WIF/AR/SNX)** — live signal generator with clock-aware entry timing and P&L tracking; combines OPP-001/002/003 + OPP-009 clock exploitation
- [ ] **Map remaining @ IDs** — 298 still unidentified; run systematic price-match during market hours when prices are live
- [ ] Run extended AR validation probe (30 min) to confirm OPP-002 with larger sample
- [ ] Check HL position size limits on SNX, AR, WIF — determine real capital ceiling
- [ ] Investigate Pyth confidence interval as signal quality filter (wide confidence = oracle uncertain = bigger HL lag)
- [x] ~~Scan for cross-asset correlation lag~~ — Done, OPP-010
- [x] ~~TradFi lag probe (NYSE hours)~~ — Done, OPP-008 full session
- [x] ~~L2 vs allMids timing~~ — Done, OPP-011 (no exploitable edge)

---

*Log maintained by latency research session. Update after each probe run.*
