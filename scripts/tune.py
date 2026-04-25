"""10-round parameter tuning sweep for laserfish momentum strategy.

Runs 12×30d walk-forward windows on cached 5m data.
Each round tests one hypothesis. Results printed in a ranked table.

Usage:
    python scripts/tune.py
    python scripts/tune.py --rounds 1,3,5   # run specific rounds only
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.data import load, cache_path

# ── Constants ────────────────────────────────────────────────────────────────
CAPITAL  = 10_000.0
FEE      = 0.00035     # 3.5bps taker
BARS_DAY = 288         # 5m bars per day
WARMUP   = 6 * 30 * BARS_DAY   # 6 months warmup
WLEN     = 30 * BARS_DAY       # 30-day OOS windows
N_WIN    = 12
ALL_SYMBOLS = ["BTC", "ETH", "SOL", "AVAX", "LINK"]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_closes(symbols: list[str]) -> dict[str, list[float]]:
    closes: dict[str, list[float]] = {}
    for sym in symbols:
        path = cache_path(Path("data"), f"{sym}USDT", "5m")
        if not path.exists():
            print(f"  WARNING: no data for {sym}")
            continue
        closes[sym] = [k.close for k in load(path)]
    return closes

# ── Core momentum z-score ─────────────────────────────────────────────────────
def mz(cl: list[float], mom: int, vol: int) -> float | None:
    if len(cl) < vol + 1:
        return None
    ret = (cl[-1] - cl[-mom - 1]) / cl[-mom - 1]
    r = [(cl[i] - cl[max(0, i - mom)]) / cl[max(0, i - mom)]
         for i in range(mom, min(len(cl), vol + 1))]
    return None if len(r) < 5 else (ret - np.mean(r)) / (np.std(r) + 1e-9)

# ── BTC MA regime filter ──────────────────────────────────────────────────────
def btc_above_ma(closes: dict, t: int, ma_win: int) -> bool:
    cl = closes.get("BTC", [])
    if t < ma_win or t >= len(cl):
        return True
    return cl[t] > np.mean(cl[t - ma_win:t])

# ── Single window simulation ──────────────────────────────────────────────────
def run_window(
    closes: dict[str, list[float]],
    symbols: list[str],
    s: int, e: int,
    mom: int, vol: int, rebal: int,
    z_entry: float, z_exit: float,
    sl: float, tp: float,
    top_n: int, leverage: float,
    regime_filter: bool = False,
    ma_win: int = 1440,     # 5-day MA
    long_only: bool = False,
    scale_by_alpha: bool = False,
) -> tuple[float, int]:
    eq = CAPITAL
    pos: dict[str, dict] = {}
    trades = 0

    for t in range(s, e):
        # exits
        for sym in list(pos):
            cl = closes[sym]
            if t >= len(cl):
                pos.pop(sym); continue
            p = cl[t]
            pm = 1 if pos[sym]["s"] == "b" else -1
            pct = pm * (p - pos[sym]["e"]) / pos[sym]["e"]
            z = mz(cl[max(0, t - vol - mom):t + 1], mom, vol)
            exit_z = z is not None and (
                (pos[sym]["s"] == "b" and z < z_exit) or
                (pos[sym]["s"] == "s" and z > -z_exit)
            )
            if pct <= -sl or pct >= tp or exit_z:
                eq += pm * pct * pos[sym]["n"] - FEE * 2 * pos[sym]["n"]
                trades += 1
                pos.pop(sym)
        eq = max(eq, 1.0)

        if (t - s) % rebal != 0:
            continue

        # regime gate
        if regime_filter and not btc_above_ma(closes, t, ma_win):
            # in bear regime: close any longs, allow shorts
            for sym in list(pos):
                if pos[sym]["s"] == "b":
                    cl = closes[sym]
                    if t < len(cl):
                        p = cl[t]
                        eq += (p - pos[sym]["e"]) / pos[sym]["e"] * pos[sym]["n"] - FEE * 2 * pos[sym]["n"]
                        trades += 1
                        pos.pop(sym)

        sc: list[tuple[float, str, str]] = []
        for sym in symbols:
            if sym in pos:
                continue
            cl = closes[sym]
            if t >= len(cl):
                continue
            z = mz(cl[max(0, t - vol - mom):t + 1], mom, vol)
            if z is None or abs(z) < z_entry:
                continue
            side = "b" if z > 0 else "s"
            if long_only and side == "s":
                continue
            sc.append((abs(z), sym, side))

        sc.sort(reverse=True)
        longs  = [(z, s2, d) for z, s2, d in sc if d == "b"][:top_n]
        shorts = [(z, s2, d) for z, s2, d in sc if d == "s"][:top_n]
        legs = longs + shorts

        al_base = eq * leverage / max(len(legs), 1) if legs else 0
        for z_score, sym, side in legs:
            if len(pos) >= top_n * 2:
                break
            cl = closes[sym]
            if t >= len(cl):
                continue
            # optional: scale notional by conviction
            al = al_base * min(z_score / z_entry, 2.0) if scale_by_alpha else al_base
            al = min(al, eq * leverage)
            eq -= FEE * al
            pos[sym] = {"s": side, "e": cl[t], "n": al}

    # close remaining
    for sym, p2 in pos.items():
        cl = closes[sym]
        if e - 1 < len(cl):
            p = cl[e - 1]
            pm = 1 if p2["s"] == "b" else -1
            eq += pm * (p - p2["e"]) / p2["e"] * p2["n"] - FEE * 2 * p2["n"]
            trades += 1

    return (eq - CAPITAL) / CAPITAL * 100, trades


def run_suite(closes: dict, symbols: list[str], configs: list[dict]) -> list[dict]:
    min_len = min(len(v) for v in closes.values())
    results = []
    for cfg in configs:
        window_results = []
        for i in range(N_WIN):
            s = WARMUP + i * WLEN
            e = s + WLEN
            if e > min_len:
                break
            ret, tr = run_window(closes, symbols, s, e, **cfg["params"])
            window_results.append((ret, tr))
        if not window_results:
            continue
        rets = [r[0] for r in window_results]
        results.append({
            "label":   cfg["label"],
            "avg_ret": float(np.mean(rets)),
            "sharpe":  float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(12)),
            "win_rate": sum(1 for r in rets if r > 0) / len(rets),
            "max_dd":  float(max(0, -min(rets))),
            "avg_trades": float(np.mean([r[1] for r in window_results])),
            "n_windows": len(window_results),
        })
    return results


def print_results(title: str, results: list[dict], baseline: dict | None = None) -> dict:
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Config':<42} {'Avg/mo':>7}  {'Sharpe':>6}  {'WinRate':>7}  {'MaxDD':>6}  {'Trades':>6}")
    print(f"  {'-'*86}")
    best = results[0] if results else None
    for r in results:
        marker = " ◄ BEST" if r is best else ""
        base_delta = f" (+{r['avg_ret']-baseline['avg_ret']:+.1f}%)" if baseline and r["label"] != baseline["label"] else ""
        print(f"  {r['label']:<42} {r['avg_ret']:>+6.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['win_rate']:>6.0%}  {r['max_dd']:>5.1f}%  {r['avg_trades']:>6.0f}{base_delta}{marker}")
    print(f"{'='*90}")
    return best or {}


# ── BASELINE ─────────────────────────────────────────────────────────────────
BASELINE_PARAMS = dict(
    mom=2016, vol=8640, rebal=BARS_DAY,
    z_entry=1.2, z_exit=0.3, sl=0.03, tp=0.05,
    top_n=3, leverage=2.0,
    regime_filter=False, long_only=False, scale_by_alpha=False,
)


# ════════════════════════════════════════════════════════════════════════════════
# ROUND DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

def round_1(closes, symbols):
    """Z-entry threshold sweep — find the conviction bar that maximises Sharpe."""
    configs = []
    for z in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
        p = {**BASELINE_PARAMS, "z_entry": z}
        configs.append({"label": f"z_entry={z}", "params": p})
    return run_suite(closes, symbols, configs)


def round_2(closes, symbols):
    """Momentum window sweep — test 3d/7d/14d/21d signal horizons."""
    configs = []
    for days, vol_days in [(3, 14), (5, 21), (7, 30), (14, 60), (21, 90)]:
        mom = days * BARS_DAY
        vol = vol_days * BARS_DAY
        p = {**BASELINE_PARAMS, "mom": mom, "vol": vol}
        configs.append({"label": f"mom={days}d vol={vol_days}d", "params": p})
    return run_suite(closes, symbols, configs)


def round_3(closes, symbols):
    """SL/TP grid — find the risk:reward ratio that survives drawdowns."""
    configs = []
    for sl, tp in product([0.02, 0.03, 0.04, 0.05], [0.03, 0.05, 0.08, 0.12]):
        if tp <= sl:
            continue
        p = {**BASELINE_PARAMS, "sl": sl, "tp": tp}
        configs.append({"label": f"SL={sl*100:.0f}% TP={tp*100:.0f}%", "params": p})
    return run_suite(closes, symbols, configs)


def round_4(closes, symbols):
    """Regime filter — BTC MA gate to suppress longs in bear markets."""
    configs = [
        {"label": "no regime filter (baseline)", "params": {**BASELINE_PARAMS}},
        {"label": "BTC > 5d MA  (gate longs)", "params": {**BASELINE_PARAMS, "regime_filter": True, "ma_win": 5*BARS_DAY}},
        {"label": "BTC > 10d MA (gate longs)", "params": {**BASELINE_PARAMS, "regime_filter": True, "ma_win": 10*BARS_DAY}},
        {"label": "BTC > 20d MA (gate longs)", "params": {**BASELINE_PARAMS, "regime_filter": True, "ma_win": 20*BARS_DAY}},
        {"label": "long-only + BTC > 20d MA",  "params": {**BASELINE_PARAMS, "regime_filter": True, "ma_win": 20*BARS_DAY, "long_only": True}},
    ]
    return run_suite(closes, symbols, configs)


def round_5(closes, symbols):
    """Top-N and leverage — portfolio concentration vs diversification."""
    configs = []
    for n, lev in product([2, 3, 4, 5], [1.0, 1.5, 2.0, 3.0]):
        p = {**BASELINE_PARAMS, "top_n": n, "leverage": lev}
        configs.append({"label": f"top_n={n} lev={lev:.1f}x", "params": p})
    return run_suite(closes, symbols, configs)


def round_6(closes, symbols):
    """Alpha-scaled sizing — size positions by conviction z-score, not equally."""
    configs = [
        {"label": "equal sizing (baseline)",        "params": {**BASELINE_PARAMS, "scale_by_alpha": False}},
        {"label": "alpha-scaled sizing",            "params": {**BASELINE_PARAMS, "scale_by_alpha": True}},
        {"label": "alpha-scaled + z=1.4",           "params": {**BASELINE_PARAMS, "scale_by_alpha": True, "z_entry": 1.4}},
        {"label": "alpha-scaled + z=1.6",           "params": {**BASELINE_PARAMS, "scale_by_alpha": True, "z_entry": 1.6}},
        {"label": "alpha-scaled + lev=3 + z=1.4",  "params": {**BASELINE_PARAMS, "scale_by_alpha": True, "z_entry": 1.4, "leverage": 3.0}},
    ]
    return run_suite(closes, symbols, configs)


def round_7(closes, symbols):
    """Rebalance frequency — daily vs weekly vs at-signal entry."""
    configs = []
    for rebal_h in [1, 4, 12, 24, 48, 168]:
        rebal = rebal_h * 12   # hours → 5m bars
        p = {**BASELINE_PARAMS, "rebal": rebal}
        configs.append({"label": f"rebal every {rebal_h}h", "params": p})
    return run_suite(closes, symbols, configs)


def round_8(closes, symbols):
    """Symbol universe — does a larger cross-section improve signal quality?"""
    # need to load extra symbols
    extra = ["DOGE", "BNB", "XRP", "NEAR", "WIF", "SUI", "HYPE", "INJ", "ARB"]
    extra_closes = {}
    for sym in extra:
        path = cache_path(Path("data"), f"{sym}USDT", "5m")
        if path.exists():
            extra_closes[sym] = [k.close for k in load(path)]

    sets = [
        ("5 symbols (baseline)",   symbols),
        ("+ DOGE BNB XRP (8)",     symbols + [s for s in ["DOGE","BNB","XRP"] if s in extra_closes]),
        ("all available (14)",     symbols + [s for s in extra if s in extra_closes]),
    ]
    results = []
    for label, sym_set in sets:
        all_closes = {**closes, **{s: extra_closes[s] for s in sym_set if s in extra_closes}}
        cfg = {"label": label, "params": BASELINE_PARAMS}
        res = run_suite(all_closes, sym_set, [cfg])
        results.extend(res)
    return results


def round_9(closes, symbols):
    """Z-exit threshold — how quickly to cut a fading position."""
    configs = []
    for z_exit in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        p = {**BASELINE_PARAMS, "z_exit": z_exit}
        configs.append({"label": f"z_exit={z_exit}", "params": p})
    return run_suite(closes, symbols, configs)


def round_10(closes, symbols):
    """Best-of-all composite — combine the top finding from each round."""
    # Will be filled after rounds 1-9 complete with best params from each
    # Hardcoded candidates to test against each other:
    combos = [
        {"label": "baseline",
         "params": {**BASELINE_PARAMS}},
        {"label": "z=1.6 + regime + alpha-scaled",
         "params": {**BASELINE_PARAMS, "z_entry": 1.6, "regime_filter": True, "ma_win": 20*BARS_DAY, "scale_by_alpha": True}},
        {"label": "z=1.4 + regime + SL4%TP8%",
         "params": {**BASELINE_PARAMS, "z_entry": 1.4, "regime_filter": True, "ma_win": 20*BARS_DAY, "sl": 0.04, "tp": 0.08}},
        {"label": "z=1.6 + regime + N=2 + lev=3",
         "params": {**BASELINE_PARAMS, "z_entry": 1.6, "regime_filter": True, "ma_win": 20*BARS_DAY, "top_n": 2, "leverage": 3.0}},
        {"label": "z=1.8 + regime + alpha + SL5%TP10%",
         "params": {**BASELINE_PARAMS, "z_entry": 1.8, "regime_filter": True, "ma_win": 20*BARS_DAY,
                    "scale_by_alpha": True, "sl": 0.05, "tp": 0.10, "leverage": 2.5}},
        {"label": "z=1.4 + alpha + SL3%TP8% + lev=2.5",
         "params": {**BASELINE_PARAMS, "z_entry": 1.4, "scale_by_alpha": True, "sl": 0.03, "tp": 0.08, "leverage": 2.5}},
        {"label": "z=1.6 + regime + alpha + rebal=4h + SL4%TP8%",
         "params": {**BASELINE_PARAMS, "z_entry": 1.6, "regime_filter": True, "ma_win": 20*BARS_DAY,
                    "scale_by_alpha": True, "rebal": 48, "sl": 0.04, "tp": 0.08}},
    ]
    return run_suite(closes, symbols, combos)


# ── Main ──────────────────────────────────────────────────────────────────────

ROUNDS = {
    1: ("Z-Entry Threshold Sweep",        round_1),
    2: ("Momentum Window Sweep",           round_2),
    3: ("Stop-Loss / Take-Profit Grid",    round_3),
    4: ("Regime Filter (BTC MA Gate)",     round_4),
    5: ("Top-N & Leverage Grid",           round_5),
    6: ("Alpha-Scaled Position Sizing",    round_6),
    7: ("Rebalance Frequency",             round_7),
    8: ("Symbol Universe Expansion",       round_8),
    9: ("Z-Exit Threshold Sweep",          round_9),
    10: ("Best Composite Configurations",  round_10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default="all",
                    help="Comma-separated round numbers, or 'all'")
    ap.add_argument("--no-extra-data", action="store_true",
                    help="Skip round 8 if extra symbol data not downloaded")
    args = ap.parse_args()

    if args.rounds == "all":
        to_run = list(ROUNDS.keys())
    else:
        to_run = [int(r) for r in args.rounds.split(",")]

    print("\nLoading data…")
    closes = load_closes(ALL_SYMBOLS)
    symbols = list(closes.keys())
    print(f"  {len(symbols)} symbols, {min(len(v) for v in closes.values()):,} bars each")

    # Print baseline first
    baseline_results = run_suite(closes, symbols, [{"label": "BASELINE", "params": BASELINE_PARAMS}])
    baseline = baseline_results[0] if baseline_results else {}
    print(f"\n  BASELINE: avg={baseline.get('avg_ret',0):+.1f}%/mo  "
          f"sharpe={baseline.get('sharpe',0):.2f}  "
          f"win_rate={baseline.get('win_rate',0):.0%}")

    all_bests = []
    t_total = time.time()

    for rnum in to_run:
        if rnum not in ROUNDS:
            print(f"  Unknown round {rnum}, skipping.")
            continue
        title, fn = ROUNDS[rnum]
        if rnum == 8 and args.no_extra_data:
            print(f"\n  Skipping Round 8 (--no-extra-data set)")
            continue
        print(f"\nRunning Round {rnum}: {title}…")
        t0 = time.time()
        results = fn(closes, symbols)
        elapsed = time.time() - t0
        best = print_results(f"Round {rnum}: {title}  ({elapsed:.0f}s)", results, baseline)
        if best:
            all_bests.append((rnum, title, best))

    # Final summary
    if len(all_bests) > 1:
        print(f"\n{'='*90}")
        print(f"  BEST CONFIG PER ROUND  (total time: {time.time()-t_total:.0f}s)")
        print(f"{'='*90}")
        print(f"  {'Rnd':<4} {'Round':<32} {'Config':<30} {'Sharpe':>6}  {'Avg/mo':>7}")
        print(f"  {'-'*86}")
        all_bests.sort(key=lambda x: x[2]["sharpe"], reverse=True)
        for rnum, title, best in all_bests:
            print(f"  R{rnum:<3d} {title:<32} {best['label']:<30} {best['sharpe']:>6.2f}  {best['avg_ret']:>+6.1f}%")
        print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
