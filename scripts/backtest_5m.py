"""Walk-forward backtest of the momentum strategy on cached 5m data.

Uses only locally cached Binance 5m klines — no live API calls.
Walk-forward: warmup on first N months, then evaluate on rolling 30-day OOS windows.

Usage:
    python scripts/backtest_5m.py
    python scripts/backtest_5m.py --symbols BTC ETH SOL --windows 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.data import load, cache_path

CAPITAL  = 10_000.0
FEE      = 0.00035   # 3.5bps taker
MAX_LEV  = 2.0
TOP_N    = 3
# 5m windows
MOM_WIN  = 12        # 12 × 5m = 1h momentum
VOL_WIN  = 72        # 72 × 5m = 6h vol window
BARS_DAY = 288       # 288 × 5m = 1 day


def _momentum_z(closes: list[float]) -> float | None:
    if len(closes) < VOL_WIN + 1:
        return None
    ret_1h = (closes[-1] - closes[-MOM_WIN - 1]) / closes[-MOM_WIN - 1]
    rolling = [
        (closes[i] - closes[max(0, i - MOM_WIN)]) / closes[max(0, i - MOM_WIN)]
        for i in range(MOM_WIN, min(len(closes), VOL_WIN + 1))
    ]
    if len(rolling) < 5:
        return None
    sd = float(np.std(rolling)) + 1e-9
    return (ret_1h - float(np.mean(rolling))) / sd


def run_window(
    closes_by_sym: dict[str, list[float]],
    symbols: list[str],
    start: int,
    end: int,
    z_entry: float = 1.2,
    z_exit: float = 0.3,
    rebal_every: int = 12,   # every 12 bars = 1h rebalance
) -> dict:
    equity = CAPITAL
    curve = [equity]
    positions: dict[str, dict] = {}
    trades = 0

    for t in range(start, end):
        # ── Exits ────────────────────────────────────────────────────────
        to_close = []
        for sym, pos in positions.items():
            cl = closes_by_sym[sym]
            if t >= len(cl):
                to_close.append(sym)
                continue
            price = cl[t]
            side_m = 1 if pos["side"] == "buy" else -1
            pnl_pct = side_m * (price - pos["entry"]) / pos["entry"]
            if pnl_pct <= -0.02 or pnl_pct >= 0.03:  # SL 2% / TP 3%
                to_close.append(sym)
                continue
            window = cl[max(0, t - VOL_WIN - MOM_WIN): t + 1]
            z = _momentum_z(window)
            if z is None:
                continue
            if pos["side"] == "buy"  and z <  z_exit:
                to_close.append(sym)
            elif pos["side"] == "sell" and z > -z_exit:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            cl = closes_by_sym[sym]
            if t < len(cl):
                price = cl[t]
                side_m = 1 if pos["side"] == "buy" else -1
                pnl = side_m * (price - pos["entry"]) / pos["entry"] * pos["notional"]
                equity += pnl - FEE * 2 * pos["notional"]
                trades += 1
        equity = max(equity, 1.0)

        # ── Entries (every rebal_every bars) ─────────────────────────────
        if (t - start) % rebal_every != 0:
            curve.append(equity)
            continue

        scores: list[tuple[float, str, str]] = []
        for sym in symbols:
            if sym in positions:
                continue
            cl = closes_by_sym[sym]
            if t >= len(cl):
                continue
            window = cl[max(0, t - VOL_WIN - MOM_WIN): t + 1]
            z = _momentum_z(window)
            if z is None or abs(z) < z_entry:
                continue
            scores.append((abs(z), sym, "buy" if z > 0 else "sell"))

        scores.sort(reverse=True)
        longs  = [(z, s, d) for z, s, d in scores if d == "buy" ][:TOP_N]
        shorts = [(z, s, d) for z, s, d in scores if d == "sell"][:TOP_N]
        legs = longs + shorts
        alloc = equity * MAX_LEV / max(len(legs), 1) if legs else 0

        for _, sym, side in legs:
            if len(positions) >= TOP_N * 2:
                break
            cl = closes_by_sym[sym]
            if t >= len(cl):
                continue
            price = cl[t]
            equity -= FEE * alloc
            positions[sym] = {"side": side, "entry": price, "notional": alloc}

        curve.append(equity)

    # Close remaining at end
    for sym, pos in positions.items():
        cl = closes_by_sym[sym]
        if end - 1 < len(cl):
            price = cl[end - 1]
            side_m = 1 if pos["side"] == "buy" else -1
            pnl = side_m * (price - pos["entry"]) / pos["entry"] * pos["notional"]
            equity += pnl - FEE * 2 * pos["notional"]
            trades += 1

    rets = np.diff(curve) / (np.array(curve[:-1]) + 1e-9)
    sharpe = float(np.mean(rets)) / (float(np.std(rets)) + 1e-9) * np.sqrt(BARS_DAY * 365)
    max_dd = 0.0
    peak = CAPITAL
    for eq in curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    return {
        "ret_pct": (equity - CAPITAL) / CAPITAL * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "trades": trades,
        "equity": equity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "AVAX", "LINK"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--warmup-months", type=int, default=3)
    ap.add_argument("--z-entry", type=float, default=1.2)
    ap.add_argument("--z-exit",  type=float, default=0.3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    closes_by_sym: dict[str, list[float]] = {}
    for sym in args.symbols:
        path = cache_path(data_dir, f"{sym}USDT", "5m")
        if not path.exists():
            print(f"Missing data for {sym} — run fetch.py first")
            sys.exit(1)
        klines = load(path)
        closes_by_sym[sym] = [k.close for k in klines]
        print(f"  {sym}: {len(klines)} bars")

    min_len = min(len(v) for v in closes_by_sym.values())
    warmup = args.warmup_months * 30 * BARS_DAY
    window_bars = args.window_days * BARS_DAY

    if warmup + args.windows * window_bars > min_len:
        print(f"Not enough data. Have {min_len} bars, need {warmup + args.windows * window_bars}")
        sys.exit(1)

    print(f"\n{'='*68}")
    print(f"  5m Momentum Backtest | {len(args.symbols)} symbols | {args.windows}×{args.window_days}d OOS windows")
    print(f"{'='*68}")
    print(f"  {'WIN':<4}  {'REGIME':<6}  {'RETURN%':>8}  {'SHARPE':>7}  {'MAXDD%':>7}  {'TRADES':>6}  {'BTC%':>6}")
    print(f"  {'-'*64}")

    results = []
    for i in range(args.windows):
        start = warmup + i * window_bars
        end = start + window_bars
        r = run_window(closes_by_sym, args.symbols, start, end,
                       z_entry=args.z_entry, z_exit=args.z_exit)

        btc = closes_by_sym.get("BTC", [])
        btc_ret = 0.0
        if btc and end - 1 < len(btc):
            btc_ret = (btc[end - 1] - btc[start]) / btc[start] * 100
        regime = "BULL" if btc_ret > 5 else ("BEAR" if btc_ret < -5 else "SIDE")
        r["btc_ret"] = btc_ret
        results.append(r)
        print(f"  W{i+1:<3d}  {regime:<6}  {r['ret_pct']:>+7.1f}%  {r['sharpe']:>7.2f}  "
              f"{r['max_dd_pct']:>6.1f}%  {r['trades']:>6}  {btc_ret:>+5.1f}%")

    print(f"  {'-'*64}")
    wins = sum(1 for r in results if r["ret_pct"] > 0)
    print(f"  AVG          {np.mean([r['ret_pct'] for r in results]):>+7.1f}%  "
          f"{np.mean([r['sharpe'] for r in results]):>7.2f}  "
          f"{np.mean([r['max_dd_pct'] for r in results]):>6.1f}%  "
          f"       {wins}/{len(results)} windows +ve")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
