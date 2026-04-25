"""Backtest of laserfish v2: momentum + HMM regime + dynamic Kelly leverage.

Compares three configs over the last N months of cached 5m data:
  A) Old baseline   : z=1.2, static 2x, 24h rebal
  B) Tuned v1       : z=1.8, static 2x, 48h rebal
  C) v2 (dynamic)   : z=1.8, HMM regime, Kelly leverage, 48h rebal

Usage:
    python scripts/backtest_v2.py
    python scripts/backtest_v2.py --months 3 --symbols BTC ETH SOL AVAX LINK
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.data import load, cache_path

CAPITAL   = 10_000.0
FEE       = 0.00035     # 3.5bps taker
BARS_DAY  = 288         # 288 × 5m = 1 day

# Momentum params (shared)
MOM_WIN   = 2016        # 2016 × 5m = 7d
VOL_WIN   = 8640        # 8640 × 5m = 30d
Z_ENTRY   = 1.8
Z_EXIT    = 0.3
TOP_N     = 3
SL_PCT    = 0.03
TP_PCT    = 0.05
REBAL_BAR = BARS_DAY * 2   # 48h rebalance

# Kelly leverage params
TARGET_VOL   = 0.50   # annualized
MIN_LEV      = 1.0
MAX_LEV      = 5.0
DD_SOFT      = 0.04   # halve leverage
DD_HARD      = 0.08   # floor to min

# HMM regime multipliers (trend/range/crisis)
REGIME_MULT = [1.5, 0.8, 0.2]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def momentum_z(closes: list[float]) -> float | None:
    if len(closes) < VOL_WIN + 1:
        return None
    ret = (closes[-1] - closes[-MOM_WIN - 1]) / closes[-MOM_WIN - 1]
    rolling = [
        (closes[i] - closes[max(0, i - MOM_WIN)]) / closes[max(0, i - MOM_WIN)]
        for i in range(MOM_WIN, min(len(closes), VOL_WIN + 1))
    ]
    if len(rolling) < 5:
        return None
    return (ret - float(np.mean(rolling))) / (float(np.std(rolling)) + 1e-9)


def realized_vol_ann(closes: list[float], window: int = 288) -> float:
    """Annualized realized vol from last `window` 5m bars."""
    if len(closes) < window + 1:
        return 0.60
    lr = np.diff(np.log(closes[-window:]))
    return float(np.std(lr)) * np.sqrt(BARS_DAY * 365)


# ─────────────────────────────────────────────────────────────────────────────
# HMM regime (fit once on warmup BTC data, then predict rolling)
# ─────────────────────────────────────────────────────────────────────────────

def build_hmm_obs(btc_closes: list[float], t: int) -> list[float] | None:
    window = 288
    if t < window + 12:
        return None
    vol = realized_vol_ann(btc_closes[:t+1], window)
    ret_1h = (btc_closes[t] - btc_closes[t - 12]) / btc_closes[t - 12]
    return [vol, ret_1h]


def fit_hmm(btc_closes: list[float], warmup_end: int) -> GaussianHMM | None:
    obs = []
    for t in range(300, warmup_end):
        o = build_hmm_obs(btc_closes, t)
        if o is not None:
            obs.append(o)
    if len(obs) < 50:
        return None
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200,
                      random_state=42)
    hmm.fit(np.array(obs))
    # Sort states by vol: 0=trend(low), 1=range, 2=crisis(high)
    order = np.argsort(hmm.means_[:, 0])
    remap = {int(order[i]): i for i in range(3)}
    return hmm, remap


def get_regime(hmm_tuple, btc_closes: list[float], t: int) -> int:
    """Return regime 0/1/2 given HMM fit and history up to bar t."""
    if hmm_tuple is None:
        return 0
    hmm, remap = hmm_tuple
    obs = []
    lookback = min(t + 1, 500)
    for i in range(t - lookback + 1, t + 1):
        o = build_hmm_obs(btc_closes, i)
        if o is not None:
            obs.append(o)
    if len(obs) < 5:
        return 0
    try:
        states = hmm.predict(np.array(obs))
        raw = int(states[-1])
        return remap.get(raw, 0)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(
    closes_by_sym: dict[str, list[float]],
    symbols: list[str],
    start: int,
    end: int,
    static_leverage: float = 2.0,
    dynamic_lev: bool = False,
    hmm_tuple=None,
    rebal_every: int = REBAL_BAR,
    z_entry: float = Z_ENTRY,
) -> dict:
    equity = CAPITAL
    curve = [equity]
    hwm = equity
    positions: dict[str, dict] = {}
    trades = 0

    btc = closes_by_sym.get("BTC", [])

    for t in range(start, end):
        # ── Exits ──────────────────────────────────────────────────────
        to_close = []
        for sym, pos in positions.items():
            cl = closes_by_sym[sym]
            if t >= len(cl):
                to_close.append(sym); continue
            price = cl[t]
            sm = 1 if pos["side"] == "buy" else -1
            pnl_pct = sm * (price - pos["entry"]) / pos["entry"]
            if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT:
                to_close.append(sym); continue
            window = cl[max(0, t - VOL_WIN - MOM_WIN): t + 1]
            z = momentum_z(window)
            if z is not None:
                if pos["side"] == "buy"  and z <  Z_EXIT: to_close.append(sym)
                elif pos["side"] == "sell" and z > -Z_EXIT: to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            cl = closes_by_sym[sym]
            if t < len(cl):
                price = cl[t]
                sm = 1 if pos["side"] == "buy" else -1
                pnl = sm * (price - pos["entry"]) / pos["entry"] * pos["notional"]
                equity += pnl - FEE * 2 * pos["notional"]
                trades += 1
        equity = max(equity, 1.0)
        hwm = max(hwm, equity)

        # ── Entries (every rebal_every bars) ───────────────────────────
        if (t - start) % rebal_every != 0:
            curve.append(equity); continue

        # Regime check (v2 only)
        if hmm_tuple is not None:
            regime = get_regime(hmm_tuple, btc, t)
            if regime == 2:  # crisis — no new entries
                curve.append(equity); continue
            regime_mult = REGIME_MULT[regime]
        else:
            regime_mult = 1.0

        # Drawdown for circuit breaker
        dd = (hwm - equity) / hwm if hwm > 0 else 0.0

        scores: list[tuple[float, str, str]] = []
        for sym in symbols:
            if sym in positions: continue
            cl = closes_by_sym[sym]
            if t >= len(cl): continue
            window = cl[max(0, t - VOL_WIN - MOM_WIN): t + 1]
            z = momentum_z(window)
            if z is None or abs(z) < z_entry: continue
            scores.append((abs(z), sym, "buy" if z > 0 else "sell"))

        scores.sort(reverse=True)
        longs  = [(z, s, d) for z, s, d in scores if d == "buy" ][:TOP_N]
        shorts = [(z, s, d) for z, s, d in scores if d == "sell"][:TOP_N]
        legs   = longs + shorts

        for _, sym, side in legs:
            if len(positions) >= TOP_N * 2: break
            cl = closes_by_sym[sym]
            if t >= len(cl): continue
            price = cl[t]

            if dynamic_lev:
                rv = realized_vol_ann(cl[:t+1])
                base = TARGET_VOL / max(rv, 0.10) * regime_mult
                if dd >= DD_HARD:   base = MIN_LEV
                elif dd >= DD_SOFT: base = base * 0.5
                lev = float(np.clip(base, MIN_LEV, MAX_LEV))
            else:
                lev = static_leverage

            notional = equity * lev / max(len(legs), 1)
            equity -= FEE * notional
            positions[sym] = {"side": side, "entry": price, "notional": notional}

        curve.append(equity)

    # Close remaining
    for sym, pos in positions.items():
        cl = closes_by_sym[sym]
        if end - 1 < len(cl):
            price = cl[end - 1]
            sm = 1 if pos["side"] == "buy" else -1
            pnl = sm * (price - pos["entry"]) / pos["entry"] * pos["notional"]
            equity += pnl - FEE * 2 * pos["notional"]
            trades += 1

    rets = np.diff(curve) / (np.array(curve[:-1]) + 1e-9)
    sharpe = float(np.mean(rets)) / (float(np.std(rets)) + 1e-9) * np.sqrt(BARS_DAY * 365)
    peak = CAPITAL
    max_dd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    return {
        "ret_pct":    (equity - CAPITAL) / CAPITAL * 100,
        "sharpe":     sharpe,
        "max_dd_pct": max_dd * 100,
        "trades":     trades,
        "equity":     equity,
        "curve":      curve,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "AVAX", "LINK"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--months", type=int, default=3,
                    help="How many months to backtest (from end of dataset)")
    ap.add_argument("--warmup-months", type=int, default=6,
                    help="Months of warm-up before OOS window")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    closes_by_sym: dict[str, list[float]] = {}
    for sym in args.symbols:
        path = cache_path(data_dir, f"{sym}USDT", "5m")
        if not path.exists():
            print(f"Missing {sym} — run fetch.py first"); sys.exit(1)
        klines = load(path)
        closes_by_sym[sym] = [k.close for k in klines]

    n = min(len(v) for v in closes_by_sym.values())
    test_bars   = args.months * 30 * BARS_DAY
    warmup_bars = args.warmup_months * 30 * BARS_DAY

    test_end   = n
    test_start = test_end - test_bars
    warmup_end = test_start   # warmup runs from (test_start - warmup_bars) to test_start

    if test_start - warmup_bars < 0:
        print(f"Not enough data. Need {warmup_bars + test_bars} bars, have {n}")
        sys.exit(1)

    # Date range info
    btc_klines = load(cache_path(data_dir, "BTCUSDT", "5m"))
    def bar_date(i):
        return datetime.fromtimestamp(btc_klines[i].open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"\n  Data range : {bar_date(0)} → {bar_date(n-1)}")
    print(f"  Warmup     : {bar_date(test_start - warmup_bars)} → {bar_date(test_start)}")
    print(f"  TEST OOS   : {bar_date(test_start)} → {bar_date(test_end-1)}")
    print(f"  Symbols    : {', '.join(args.symbols)}")

    # Fit HMM on warmup window
    print("\n  Fitting HMM on warmup data…", end=" ", flush=True)
    btc = closes_by_sym["BTC"]
    hmm_result = fit_hmm(btc, warmup_end)
    if hmm_result:
        hmm_tuple, remap = hmm_result
        vol_means = hmm_tuple.means_[:, 0]
        order = np.argsort(vol_means)
        print(f"OK  (trend={vol_means[order[0]]*100:.0f}%  range={vol_means[order[1]]*100:.0f}%  crisis={vol_means[order[2]]*100:.0f}%)")
    else:
        hmm_tuple = None
        print("FAILED — running without regime filter")

    # BTC benchmark
    btc_ret = (btc[test_end-1] - btc[test_start]) / btc[test_start] * 100

    # Run all three configs
    configs = [
        ("Old baseline  (z=1.2, 2x static, 24h)", dict(static_leverage=2.0, dynamic_lev=False,
            hmm_tuple=None, rebal_every=BARS_DAY, z_entry=1.2)),
        ("Tuned v1      (z=1.8, 2x static, 48h)", dict(static_leverage=2.0, dynamic_lev=False,
            hmm_tuple=None, rebal_every=REBAL_BAR, z_entry=1.8)),
        ("v2 dynamic    (z=1.8, HMM+Kelly, 48h)", dict(static_leverage=2.0, dynamic_lev=True,
            hmm_tuple=(hmm_tuple, remap) if hmm_tuple else None, rebal_every=REBAL_BAR, z_entry=1.8)),
    ]

    print(f"\n{'='*80}")
    print(f"  Last {args.months} months — {bar_date(test_start)} to {bar_date(test_end-1)}")
    print(f"{'='*80}")
    print(f"  {'Config':<42} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-'*76}")

    results = []
    for label, kwargs in configs:
        r = run_sim(closes_by_sym, args.symbols, test_start, test_end, **kwargs)
        results.append((label, r))
        print(f"  {label:<42} {r['ret_pct']:>+7.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['max_dd_pct']:>6.1f}%  {r['trades']:>6}")

    print(f"  {'-'*76}")
    print(f"  {'BTC buy-and-hold':<42} {btc_ret:>+7.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
