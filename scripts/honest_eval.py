"""
Honest funding-arb evaluation.

Rules:
  1. ONLY uses windows where real Hyperliquid funding rate data exists.
     No proxy signals. No (high-low)/close substitution.
     If real funding is unavailable for a symbol/period, it is excluded.

  2. Execution model includes:
     - Spread cost: entry/exit at mid ± half the live spread per symbol
     - Slippage tier: normal (1× spread), stressed (3× spread when
       funding z-score > 3.0 — elevated basis suggests elevated spread)
     - Fill uncertainty: 8% of entries miss (position stays flat that period)
       modelling limit orders that don't fill before signal expires
     - Liquidation: hard stop at -1/leverage of entry price (3x → -33%)

  3. PnL is split into:
     - Funding income: cumulative 8h payments received/paid
     - Directional PnL: mark-to-market moves while position is open
     - Costs: fees + spread + slippage
     So we can see whether the edge is actually funding or directional drift.

  4. Parameter sweep runs the FULL grid including zones outside the
     "safe" curated region, so fragility is visible.

  5. Crash model:
     - Spread blowout: during simulated stress, spread widens 5×
     - Funding flip: sign of funding rate inverts mid-crash
     - Fill degradation: fill rate drops to 50% (half entries miss)
     - Liquidity gap: 5% of exits execute 2% worse than expected
"""
from __future__ import annotations

import math
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CAPITAL     = 10_000.0
BASE_FEE    = 0.00035    # 3.5bps taker
LEVERAGE    = 3
ALLOC       = 0.10

# Symbols with liquid spot markets on Hyperliquid — can be fully delta-hedged
SPOT_HEDGEABLE = frozenset(["BTC", "ETH", "SOL", "AVAX", "HYPE", "LINK", "BNB"])

# Live measured spreads (bps) — used as slippage per symbol
LIVE_SPREADS_BPS: dict[str, float] = {
    "BTC": 0.13, "ETH": 0.43, "SOL": 0.12, "ARB": 2.36,
    "OP":  0.81, "AVAX": 5.99, "DOGE": 0.63, "XRP": 0.70,
    "NEAR": 4.37, "HYPE": 1.50, "INJ": 4.28, "BNB": 1.11,
    "KPEPE": 5.30, "WIF": 0.50, "SUI": 1.91, "LINK": 0.96,
}
DEFAULT_SPREAD_BPS = 2.0


# ─── data fetching ────────────────────────────────────────────────────────────

def fetch_real_data() -> tuple[dict, dict, dict]:
    """
    Returns:
      funding_8h:  {sym: [(ts_ms, rate_8h)]}    — aggregated from hourly
      candles_4h:  {sym: [[ts,o,h,l,c,v]]}      — 4h price candles
      spreads_bps: {sym: float}                  — live measured spread
    """
    from trader.exchanges.hyperliquid import HyperliquidExchange, _hl_symbol
    import time

    SYMS = list(LIVE_SPREADS_BPS.keys())
    ex   = HyperliquidExchange(paper=True)

    funding_8h: dict[str, list] = {}
    candles_4h: dict[str, list] = {}

    print("Fetching real funding history (hourly → aggregated to 8h)…")
    for sym in SYMS:
        for attempt in range(3):
            try:
                raw = ex._client.fetch_funding_rate_history(_hl_symbol(sym), limit=500)
                hourly = [(int(r["timestamp"]), float(r.get("fundingRate") or 0)) for r in raw]
                # Aggregate: sum 8 consecutive hourly rates per 8h settlement
                agg = []
                for i in range(0, len(hourly) - 7, 8):
                    ts   = hourly[i][0]
                    rate = sum(r for _, r in hourly[i:i+8])
                    agg.append((ts, rate))
                funding_8h[sym] = agg
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"  WARNING: {sym} funding skipped: {e}")
                    funding_8h[sym] = []

    print("Fetching 4h candles (aligned to funding window)…")
    for sym in SYMS:
        for attempt in range(3):
            try:
                rows = ex._client.fetch_ohlcv(_hl_symbol(sym), "4h", limit=120)
                candles_4h[sym] = rows
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    candles_4h[sym] = []

    # Confirm real coverage
    real_periods = {sym: len(funding_8h.get(sym, [])) for sym in SYMS}
    usable = [s for s, n in real_periods.items() if n >= 5]
    print(f"\n  Symbols with ≥5 real 8h periods: {len(usable)}/{len(SYMS)}")
    for sym in SYMS:
        n = real_periods[sym]
        if n:
            t0 = funding_8h[sym][0][0]; t1 = funding_8h[sym][-1][0]
            import datetime
            d0 = datetime.datetime.fromtimestamp(t0/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.datetime.fromtimestamp(t1/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            print(f"    {sym:6s}: {n:3d} 8h periods  {d0} → {d1}")

    return funding_8h, candles_4h, LIVE_SPREADS_BPS


# ─── core sim (real funding only) ────────────────────────────────────────────

def _honest_sim(
    funding_8h: dict[str, list],
    candles_4h: dict[str, list],
    spreads_bps: dict[str, float],
    z_entry: float  = 1.5,
    z_exit: float   = 0.2,
    alloc: float    = ALLOC,
    leverage: int   = LEVERAGE,
    fee_rate: float = BASE_FEE,
    capital: float  = CAPITAL,
    # Friction controls
    fill_rate: float        = 0.92,   # fraction of entries that fill
    stressed_spread_mult: float = 3.0, # spread multiplier when z > 3
    liq_threshold: float    = None,   # auto = -1/leverage
    # Stress injection
    stress_from_period: int | None = None,  # period index to start stress
    stress_spread_mult: float      = 5.0,
    stress_fill_rate: float        = 0.50,
    stress_funding_flip: bool      = False,
    stress_exit_slip_pct: float    = 0.02,
    seed: int = 42,
) -> dict:
    """
    Simulate funding arb using ONLY real 8h funding data.
    Every cost is modelled; PnL is split into components.
    """
    rng = np.random.default_rng(seed)

    if liq_threshold is None:
        liq_threshold = 1.0 / leverage   # 33% adverse move at 3x

    rolling = {s: deque(maxlen=90) for s in funding_8h}
    consec  = {s: 0 for s in funding_8h}
    cash    = capital

    # Position state
    pos: dict[str, dict] = {}

    # PnL components
    funding_income  = 0.0
    directional_pnl = 0.0
    cost_total      = 0.0
    liquidations    = 0

    equity_curve  = [capital]
    trades: list[dict] = []

    def _z(sym, rate):
        h = list(rolling[sym])
        if len(h) < 10:
            return None
        mu, sd = float(np.mean(h)), float(np.std(h))
        return (rate - mu) / sd if sd > 1e-9 else 0.0

    def _spread_cost(sym, price, is_stressed, z_abs):
        raw_bps = spreads_bps.get(sym, DEFAULT_SPREAD_BPS)
        mult    = stressed_spread_mult if z_abs > 3.0 else 1.0
        if is_stressed:
            mult = max(mult, stress_spread_mult)
        return price * (raw_bps / 10_000) * mult / 2  # half-spread per side

    # Align candle prices to funding periods (2 × 4h = one 8h period)
    # Build price lookup: for each symbol, map period index → close price
    price_series: dict[str, list[float]] = {}
    for sym, rows in candles_4h.items():
        # Take every 2nd candle close (end of each 8h window)
        prices = [float(rows[i][4]) for i in range(1, len(rows), 2)]
        price_series[sym] = prices

    n_periods = min(
        len(v) for v in funding_8h.values() if v
    )

    for i in range(n_periods):
        is_stressed = stress_from_period is not None and i >= stress_from_period
        eff_fill    = stress_fill_rate if is_stressed else fill_rate
        period_prices: dict[str, float] = {}

        for sym in funding_8h:
            fh = funding_8h.get(sym, [])
            if i >= len(fh):
                continue
            _, rate_8h = fh[i]
            if stress_funding_flip and is_stressed:
                rate_8h = -rate_8h

            # Price for this period
            ps = price_series.get(sym, [])
            price = ps[i] if i < len(ps) else 0.0
            if price <= 0:
                continue
            period_prices[sym] = price

            rolling[sym].append(rate_8h)

            # ── Accrue funding on open positions ──────────────────────────
            if sym in pos:
                p   = pos[sym]
                direction = 1 if p["side"] == "long" else -1
                payment   = -direction * rate_8h * p["qty"] * price
                p["funding_pnl"] += payment
                funding_income   += payment
                cash             += payment

                # ── Liquidation check ─────────────────────────────────────
                adverse_pct = direction * (price / p["entry"] - 1)
                if adverse_pct <= -liq_threshold:
                    # Liquidated: lose the margin
                    margin = p["entry"] * p["qty"] / leverage
                    loss   = -margin   # simplification: margin wiped
                    directional_pnl += loss + p["funding_pnl"]
                    cost_total      += p["entry"] * p["qty"] * fee_rate
                    liquidations    += 1
                    trades.append({
                        "sym": sym, "side": p["side"],
                        "entry": p["entry"], "exit": price,
                        "pnl_net": loss + p["funding_pnl"],
                        "type": "liquidation",
                        "funding_pnl": p["funding_pnl"],
                        "directional": loss,
                    })
                    pos.pop(sym)
                    continue

                # ── Exit check ────────────────────────────────────────────
                z = _z(sym, rate_8h)
                if z is not None and abs(z) <= z_exit:
                    sc    = _spread_cost(sym, price, is_stressed, abs(z))
                    # Stress exit gap
                    slip  = price * stress_exit_slip_pct if is_stressed and rng.random() < 0.05 else 0.0
                    exit_price = price - slip * direction
                    direction  = 1 if p["side"] == "long" else -1
                    dir_pnl    = direction * p["qty"] * (exit_price - p["entry"])
                    fee        = p["qty"] * exit_price * fee_rate
                    cost       = fee + sc * p["qty"]
                    total_pnl  = dir_pnl + p["funding_pnl"] - cost

                    directional_pnl += dir_pnl
                    cost_total      += cost
                    margin = p["entry"] * p["qty"] / leverage
                    cash  += margin + total_pnl
                    trades.append({
                        "sym": sym, "side": p["side"],
                        "entry": p["entry"], "exit": exit_price,
                        "pnl_net": total_pnl, "type": "normal",
                        "funding_pnl": p["funding_pnl"],
                        "directional": dir_pnl, "cost": cost,
                        "hold": i - p["open_i"],
                    })
                    pos.pop(sym)

        # ── Entry signals ─────────────────────────────────────────────────
        for sym in funding_8h:
            if sym in pos:
                continue
            fh = funding_8h.get(sym, [])
            if i >= len(fh):
                continue
            _, rate_8h = fh[i]
            if stress_funding_flip and is_stressed:
                rate_8h = -rate_8h
            price = period_prices.get(sym, 0.0)
            if price <= 0:
                continue

            z = _z(sym, rate_8h)
            if z is None or abs(z) < z_entry:
                consec[sym] = 0
                continue
            consec[sym] += 1
            if consec[sym] < 2:
                continue

            # Fill uncertainty
            if rng.random() > eff_fill:
                continue   # order didn't fill

            side     = "short" if z > 0 else "long"
            sc       = _spread_cost(sym, price, is_stressed, abs(z))
            notional = capital * alloc * leverage
            margin   = notional / leverage
            if margin > cash:
                continue
            qty      = notional / price
            entry_cost = notional * fee_rate + sc * qty
            cash    -= margin + entry_cost
            cost_total += entry_cost
            pos[sym] = {
                "side": side, "qty": qty, "entry": price,
                "funding_pnl": 0.0, "open_i": i, "z_open": abs(z),
            }

        # Mark equity
        upnl = 0.0
        for sym, p in pos.items():
            pr = period_prices.get(sym, p["entry"])
            d  = 1 if p["side"] == "long" else -1
            upnl += d * p["qty"] * (pr - p["entry"]) + p["funding_pnl"]
        margin_held = sum(p["entry"] * p["qty"] / leverage for p in pos.values())
        equity_curve.append(cash + margin_held + upnl)

    # Force-close at last known price
    for sym, p in list(pos.items()):
        price = period_prices.get(sym, p["entry"])
        direction = 1 if p["side"] == "long" else -1
        dir_pnl = direction * p["qty"] * (price - p["entry"])
        fee     = p["qty"] * price * fee_rate
        sc      = _spread_cost(sym, price, False, 0)
        cost    = fee + sc * p["qty"]
        total   = dir_pnl + p["funding_pnl"] - cost
        directional_pnl += dir_pnl
        cost_total      += cost
        margin = p["entry"] * p["qty"] / leverage
        cash  += margin + total
        trades.append({
            "sym": sym, "side": p["side"],
            "entry": p["entry"], "exit": price,
            "pnl_net": total, "type": "forced",
            "funding_pnl": p["funding_pnl"],
            "directional": dir_pnl, "cost": cost,
            "hold": n_periods - p["open_i"],
        })

    final = equity_curve[-1]
    wr    = sum(1 for t in trades if t.get("pnl_net", 0) > 0) / max(len(trades), 1) * 100

    eq   = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / (peak + 1e-9)).min() * 100)

    return {
        "ret_pct":       (final - capital) / capital * 100,
        "final":         final,
        "n_trades":      len(trades),
        "win_rate":      wr,
        "max_dd":        max_dd,
        "funding_income": funding_income,
        "directional_pnl": directional_pnl,
        "cost_total":    cost_total,
        "liquidations":  liquidations,
        "equity_curve":  equity_curve,
        "trades":        trades,
        "n_periods":     n_periods,
    }


# ─── worker for parallel sweep ────────────────────────────────────────────────

def _sweep_worker(job: dict) -> dict:
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    import numpy as np
    from collections import deque
    r = _honest_sim(
        funding_8h=job["funding_8h"],
        candles_4h=job["candles_4h"],
        spreads_bps=job["spreads_bps"],
        z_entry=job["z_entry"],
        z_exit=job["z_exit"],
        alloc=job["alloc"],
        leverage=job["leverage"],
        fee_rate=BASE_FEE,
    )
    return {**r, "z_entry": job["z_entry"], "z_exit": job["z_exit"],
            "alloc": job["alloc"], "leverage": job["leverage"]}


# ─── reporting ────────────────────────────────────────────────────────────────

def _sharpe(curve: list[float]) -> float:
    if len(curve) < 2:
        return 0.0
    r = np.diff(curve) / (np.array(curve[:-1]) + 1e-9)
    return float(np.mean(r) / (np.std(r) + 1e-9) * math.sqrt(len(r)))


def report_main(r: dict, label: str = "REAL FUNDING EVAL") -> None:
    capital = CAPITAL
    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"{'═'*62}")
    print(f"  Real 8h periods    : {r['n_periods']}")
    print(f"  Trades             : {r['n_trades']}")
    print(f"  Liquidations       : {r['liquidations']}")
    print(f"  Starting capital   : ${capital:>10,.2f}")
    print(f"  Final equity       : ${r['final']:>10,.2f}")
    print(f"  Total return       : {r['ret_pct']:>+.2f}%")
    print(f"  Max drawdown       : {r['max_dd']:>+.2f}%")
    print(f"  Sharpe             : {_sharpe(r['equity_curve']):.2f}")
    print(f"  Win rate           : {r['win_rate']:.1f}%")
    print()
    print("  PnL breakdown:")
    print(f"    Funding income   : ${r['funding_income']:>+9.2f}  "
          f"({r['funding_income']/(r['final']-capital)*100 if r['final']!=capital else 0:.0f}% of total edge)")
    print(f"    Directional PnL  : ${r['directional_pnl']:>+9.2f}")
    print(f"    Total costs      : ${-r['cost_total']:>+9.2f}")
    net = r['funding_income'] + r['directional_pnl'] - r['cost_total']
    print(f"    Net              : ${net:>+9.2f}")
    print()


def report_sweep(results: list[dict]) -> None:
    rets = [r["ret_pct"] for r in results]
    pos  = sum(1 for x in rets if x > 0)
    neg  = len(rets) - pos

    print(f"\n{'═'*72}")
    print(f"  HONEST PARAMETER SWEEP (full grid including danger zones)")
    print(f"  {len(results)} combos  ·  z_entry∈[0.5,4.0]  ·  leverage∈[1,5,10]  ·  alloc∈[5%,15%,25%]")
    print(f"{'═'*72}")
    print(f"  {'z_entry':>7}  {'z_exit':>6}  {'alloc%':>6}  {'lev':>3}  "
          f"{'Ret%':>8}  {'DD%':>7}  {'Liq':>4}")
    print(f"  {'─'*65}")

    for r in sorted(results, key=lambda x: x["ret_pct"], reverse=True):
        flag = "✓" if r["ret_pct"] > 0 else "✗"
        print(f"  {r['z_entry']:>7.1f}  {r['z_exit']:>6.2f}  "
              f"{r['alloc']*100:>5.0f}%  {r['leverage']:>3}x  "
              f"  {r['ret_pct']:>+7.1f}%{flag}  "
              f"{r['max_dd']:>+6.1f}%  {r['liquidations']:>4}")

    print(f"  {'─'*65}")
    print(f"  {pos}/{len(results)} positive  ·  "
          f"median {float(np.median(rets)):>+.1f}%  ·  "
          f"worst {min(rets):>+.1f}%  ·  "
          f"best {max(rets):>+.1f}%")
    print(f"  {'─'*65}")
    print(f"  NOTE: {neg} combos lost money — this is the honest picture.")


def report_crash(results: list[dict]) -> None:
    print(f"\n{'═'*62}")
    print("  REALISTIC CRASH MODEL")
    print("  (spread 5×, fill rate 50%, funding flip, 2% exit gap on 5% of trades)")
    print(f"{'═'*62}")
    survived = sum(1 for r in results if r["ret_pct"] > -50)
    wiped    = sum(1 for r in results if r["ret_pct"] <= -100)
    rets     = [r["ret_pct"] for r in results]
    print(f"  Trials             : {len(results)}")
    print(f"  Survived (>-50%)   : {survived}/{len(results)}")
    print(f"  Wiped out (≤-100%) : {wiped}/{len(results)}")
    print(f"  Avg return         : {float(np.mean(rets)):>+.1f}%")
    print(f"  Worst outcome      : {min(rets):>+.1f}%")
    print(f"  Liquidations total : {sum(r['liquidations'] for r in results)}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Fetch real data
    funding_8h, candles_4h, spreads_bps = fetch_real_data()

    usable_syms = [s for s in funding_8h if len(funding_8h[s]) >= 5]
    if not usable_syms:
        print("ERROR: No real funding data available.")
        return

    # Narrow to usable symbols only
    funding_8h  = {s: funding_8h[s]  for s in usable_syms}
    candles_4h  = {s: candles_4h.get(s, []) for s in usable_syms}
    spreads_bps = {s: spreads_bps.get(s, DEFAULT_SPREAD_BPS) for s in usable_syms}

    n_periods = min(len(v) for v in funding_8h.values() if v)
    print(f"\n  Evaluation window: {n_periods} real 8h periods "
          f"(~{n_periods/3:.1f} days, {len(usable_syms)} symbols)\n")

    # ── SECTION 1: Baseline honest result ────────────────────────────────
    print("Running baseline honest evaluation (z_entry=1.5, z_exit=0.2)…")
    baseline = _honest_sim(funding_8h, candles_4h, spreads_bps)
    report_main(baseline, "SECTION 1 — BASELINE (real funding, realistic frictions)")

    # ── SECTION 1b: Hedged carry model ───────────────────────────────────
    # Restrict to spot-hedgeable symbols only. Zero out directional PnL
    # (the spot leg offsets it); double costs (two-leg round trip).
    # This answers: "would pure carry be profitable after costs?"
    hedgeable_funding  = {s: v for s, v in funding_8h.items()  if s in SPOT_HEDGEABLE}
    hedgeable_candles  = {s: v for s, v in candles_4h.items()  if s in SPOT_HEDGEABLE}
    hedgeable_spreads  = {s: v for s, v in spreads_bps.items() if s in SPOT_HEDGEABLE}
    if hedgeable_funding:
        print("Running hedged carry model (spot-hedgeable symbols only)…")
        hedged_raw = _honest_sim(hedgeable_funding, hedgeable_candles, hedgeable_spreads,
                                 fee_rate=BASE_FEE * 2)   # two-leg cost
        # Neutralise directional component — spot leg offsets it exactly
        hedged_funding_only = dict(hedged_raw)
        hedged_funding_only["directional_pnl"] = 0.0
        hedged_net = hedged_raw["funding_income"] - hedged_raw["cost_total"]
        hedged_ret = hedged_net / CAPITAL * 100
        print(f"\n{'═'*62}")
        print("  SECTION 1b — HEDGED CARRY MODEL")
        print(f"  (spot-hedgeable symbols: {', '.join(sorted(SPOT_HEDGEABLE))})")
        print(f"{'═'*62}")
        print(f"  Symbols             : {len(hedgeable_funding)}")
        print(f"  Trades (perp side)  : {hedged_raw['n_trades']}")
        print(f"  Funding collected   : ${hedged_raw['funding_income']:>+9.2f}")
        print(f"  Two-leg costs       : ${-hedged_raw['cost_total']:>+9.2f}")
        print(f"  Net carry           : ${hedged_net:>+9.2f}")
        print(f"  Return (carry only) : {hedged_ret:>+.2f}%")
        print()
        if hedged_net > 0:
            print("  ✓ Carry COVERS costs in this window — hedge makes the strategy viable")
        else:
            print("  ✗ Carry does NOT cover costs — market is in a low-rate regime")
            per_trade = hedged_net / max(hedged_raw["n_trades"], 1)
            ann_needed = abs(per_trade / (CAPITAL * ALLOC)) * 1095 * 100
            print(f"    Need ~{ann_needed:.0f}% annualized funding to break even at current trade rate")
        print()

    # ── SECTION 2: No-friction comparison (shows what frictions cost) ─────
    print("Running no-friction comparison…")
    no_friction = _honest_sim(
        funding_8h, candles_4h, spreads_bps,
        fill_rate=1.0, stressed_spread_mult=1.0, fee_rate=0.0,
    )
    print(f"\n  SECTION 2 — FRICTION COST BREAKDOWN")
    print(f"  With realistic frictions : {baseline['ret_pct']:>+.2f}%")
    print(f"  Zero frictions (fantasy) : {no_friction['ret_pct']:>+.2f}%")
    friction_drag = no_friction['ret_pct'] - baseline['ret_pct']
    print(f"  Friction drag            : {-friction_drag:>+.2f}% of return eaten by costs/fills")

    # ── SECTION 3: Full parameter sweep including danger zones ────────────
    print("\nBuilding full parameter sweep…")
    sweep_grid = []
    for ze in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        for zx in [0.1, 0.2, 0.5, 1.0]:
            for al in [0.05, 0.15, 0.25]:
                for lv in [1, 5, 10]:
                    if ze <= zx:
                        continue   # degenerate: entry ≤ exit
                    sweep_grid.append({
                        "funding_8h": funding_8h,
                        "candles_4h": candles_4h,
                        "spreads_bps": spreads_bps,
                        "z_entry": ze, "z_exit": zx,
                        "alloc": al, "leverage": lv,
                    })

    print(f"  {len(sweep_grid)} combos — running in parallel…")
    sweep_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_sweep_worker, j): j for j in sweep_grid}
        for fut in as_completed(futs):
            try:
                sweep_results.append(fut.result())
            except Exception as e:
                pass

    report_sweep(sweep_results)

    # ── SECTION 4: Realistic crash model ─────────────────────────────────
    print("\nRunning realistic crash scenarios (20 trials)…")
    rng = np.random.default_rng(77)
    crash_results = []
    for trial in range(20):
        crash_at = int(rng.integers(n_periods // 4, 3 * n_periods // 4))
        r = _honest_sim(
            funding_8h, candles_4h, spreads_bps,
            stress_from_period=crash_at,
            stress_spread_mult=5.0,
            stress_fill_rate=0.50,
            stress_funding_flip=True,
            stress_exit_slip_pct=0.02,
            seed=int(trial * 13 + 7),
        )
        r["crash_at"] = crash_at
        crash_results.append(r)

    report_crash(crash_results)

    # ── SECTION 5: Honest summary ─────────────────────────────────────────
    sweep_pos = sum(1 for r in sweep_results if r["ret_pct"] > 0)
    crash_surv = sum(1 for r in crash_results if r["ret_pct"] > -50)
    funding_pct = (baseline["funding_income"] /
                   (baseline["final"] - CAPITAL) * 100
                   if baseline["final"] != CAPITAL else 0)

    print(f"\n{'═'*62}")
    print("  HONEST SUMMARY — What we actually know")
    print(f"{'═'*62}")
    print(f"  Data window     : {n_periods} real 8h periods  (~{n_periods/3:.0f} days)")
    print(f"  Real return     : {baseline['ret_pct']:>+.2f}%  "
          f"(${baseline['final']-CAPITAL:>+,.2f} on ${CAPITAL:,.0f})")
    print(f"  Funding share   : {funding_pct:.0f}% of edge came from actual funding payments")
    print(f"  Directional     : {100-funding_pct:.0f}% came from price moves during hold")
    print()
    print(f"  Param space     : {sweep_pos}/{len(sweep_results)} combos profitable "
          f"(honest: includes danger zones)")
    print(f"  Crash survival  : {crash_surv}/20 under realistic stress model")
    print()
    print("  What this evaluation cannot tell you:")
    print("  ✗  How the strategy performs in a sustained high-funding regime")
    print(f"  ✗  Whether {n_periods/3:.0f} days is enough to separate skill from luck")
    print("  ✗  How live slippage actually behaves at scale (depth was thin for ARB/NEAR/INJ)")
    print("  ✗  Whether funding rate patterns in this window are representative")
    print()
    print("  Minimum bar before going live with real capital:")
    print("  → Run paper loop for ≥14 days, report funding_income vs directional_pnl")
    print("  → Require funding_income > 50% of gross PnL")
    print("  → Require ≥10 completed real-funding-triggered trades")
    print()


if __name__ == "__main__":
    main()
