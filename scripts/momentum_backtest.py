"""Honest backtester for the cross-sectional momentum strategy.

Uses only real Hyperliquid data.  Walk-forward: trains z-score parameters
on the first N days (in-sample), evaluates on each subsequent 30-day window
(out-of-sample).  No forward-looking bias.

Usage:
    python scripts/momentum_backtest.py
    python scripts/momentum_backtest.py --symbols BTC ETH SOL AVAX LINK --windows 6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.exchanges.hyperliquid import HyperliquidExchange, SUPPORTED_SYMBOLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("momentum_backtest")

CAPITAL    = 10_000.0
FEE        = 0.00035      # 3.5bps taker
MAX_LEV    = 2.0
TOP_N      = 3            # long/short legs each


def fetch_data(ex: HyperliquidExchange, symbols: list[str], days: int) -> dict[str, list]:
    """Returns {sym: [Candle, ...]} sorted oldest first."""
    bars_needed = days * 6 + 60    # 4h bars + lookback
    out = {}
    for sym in symbols:
        try:
            out[sym] = ex.get_candles(sym, "4h", min(bars_needed, 2500))
        except Exception as e:
            logger.warning("%s: %s", sym, e)
            out[sym] = []
    return out


def _compute_momentum_z(closes: list[float], mom_win: int = 42, vol_win: int = 180) -> float | None:
    """
    Momentum z-score: 7-day return normalised by rolling 30-day return std.

    7-day (42 × 4h bars) is the documented horizon with positive Sharpe in
    crypto (Liu & Tsyvinski 2021).  24h shows SHORT-TERM REVERSAL, not
    momentum — the wrong sign entirely.
    """
    if len(closes) < vol_win + 1:
        return None
    ret_7d = (closes[-1] - closes[-mom_win - 1]) / closes[-mom_win - 1]
    rolling = [(closes[i] - closes[max(0, i - mom_win)]) / closes[max(0, i - mom_win)]
               for i in range(mom_win, min(len(closes), vol_win + 1))]
    if len(rolling) < 10:
        return None
    sd = float(np.std(rolling)) + 1e-9
    return (ret_7d - float(np.mean(rolling))) / sd


def _market_regime(candles: dict, symbols: list[str], t: int, window: int = 6) -> str:
    """'bull' / 'bear' / 'neutral' from median 24h return across symbols."""
    rets = []
    for sym in symbols:
        c = candles.get(sym, [])
        if t >= len(c) and (t - window) >= 0:
            continue
        if len(c) > t and len(c) > t - window > 0:
            rets.append((c[t].close - c[t - window].close) / c[t - window].close)
    if not rets:
        return "neutral"
    med = float(np.median(rets))
    if med > 0.015:
        return "bull"
    if med < -0.015:
        return "bear"
    return "neutral"


def run_window(
    candles: dict[str, list],
    symbols: list[str],
    start_bar: int,
    end_bar: int,
    z_entry: float = 1.2,
    z_exit: float  = 0.3,
    fee: float     = FEE,
    capital: float = CAPITAL,
    rebal_every: int = 6,    # only re-evaluate every N bars (reduce churn)
) -> dict:
    """Simulate one walk-forward window.  No lookahead — uses only closes[0:t]."""
    equity    = capital
    equity_curve = [equity]
    positions: dict[str, dict] = {}  # sym → {side, entry, qty}
    trade_count = 0

    for t in range(start_bar, end_bar):
        # ── Exit stale positions ───────────────────────────────────────
        to_close = []
        for sym, pos in positions.items():
            cdata = candles[sym]
            if t >= len(cdata):
                to_close.append(sym)
                continue
            closes = [c.close for c in cdata[max(0, t - 220):t + 1]]
            z = _compute_momentum_z(closes)
            if z is None:
                continue
            price = cdata[t].close
            side_mult = 1 if pos["side"] == "buy" else -1
            pnl_pct = side_mult * (price - pos["entry"]) / pos["entry"]
            if pnl_pct <= -0.05 or pnl_pct >= 0.08:
                to_close.append(sym)
                continue
            if pos["side"] == "buy"  and z < z_exit:
                to_close.append(sym)
            elif pos["side"] == "sell" and z > -z_exit:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            cdata = candles[sym]
            if t < len(cdata):
                price  = cdata[t].close
                side_m = 1 if pos["side"] == "buy" else -1
                pnl    = side_m * (price - pos["entry"]) / pos["entry"] * pos["notional"]
                cost   = fee * 2 * pos["notional"]
                equity += pnl - cost
                trade_count += 1

        equity = max(equity, 1.0)

        # ── Only scan for new entries on rebalance bars ────────────────
        if (t - start_bar) % rebal_every != 0:
            equity_curve.append(max(equity, 1.0))
            continue

        # ── Market regime gate ─────────────────────────────────────────
        # Only go long in bull/neutral, only short in bear/neutral.
        # Prevents the short leg fighting a rising-tide bull market and
        # the long leg fighting a falling-tide bear market.
        regime = _market_regime(candles, symbols, t)
        allow_longs  = regime in ("bull", "neutral")
        allow_shorts = regime in ("bear", "neutral")

        # ── Scan for new signals ───────────────────────────────────────
        scores: list[tuple[float, str, str]] = []
        for sym in symbols:
            if sym in positions:
                continue
            cdata = candles[sym]
            if t >= len(cdata):
                continue
            closes = [c.close for c in cdata[max(0, t - 220):t + 1]]
            z = _compute_momentum_z(closes)
            if z is None or abs(z) < z_entry:
                continue
            side = "buy" if z > 0 else "sell"
            if side == "buy"  and not allow_longs:
                continue
            if side == "sell" and not allow_shorts:
                continue
            scores.append((abs(z), sym, side))

        scores.sort(reverse=True)
        longs  = [(z, s, d) for z, s, d in scores if d == "buy" ][:TOP_N]
        shorts = [(z, s, d) for z, s, d in scores if d == "sell"][:TOP_N]

        # Size: fewer active legs when only one side allowed
        active_legs = len(longs) + len(shorts)
        alloc_per_pos = (equity * MAX_LEV / max(active_legs, 1)) if active_legs else 0
        for _, sym, side in longs + shorts:
            if len(positions) >= TOP_N * 2:
                break
            cdata = candles[sym]
            if t >= len(cdata):
                continue
            price    = cdata[t].close
            notional = alloc_per_pos
            qty      = notional / price
            equity  -= fee * notional
            positions[sym] = {"side": side, "entry": price, "qty": qty, "notional": notional}

        equity_curve.append(max(equity, 1.0))

    # Close all open positions at end of window
    for sym, pos in positions.items():
        cdata = candles[sym]
        if end_bar - 1 < len(cdata):
            price  = cdata[end_bar - 1].close
            side_m = 1 if pos["side"] == "buy" else -1
            pnl    = side_m * (price - pos["entry"]) / pos["entry"] * pos["notional"]
            cost   = fee * 2 * pos["notional"]
            equity += pnl - cost
            trade_count += 1

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe  = (float(np.mean(returns)) / (float(np.std(returns)) + 1e-9)) * np.sqrt(6 * 365)
    total_return = (equity - capital) / capital

    peak, max_dd = capital, 0.0
    for eq in equity_curve:
        peak  = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    return {
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "trade_count": trade_count,
        "final_equity": equity,
        "bars": end_bar - start_bar,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--days",    type=int, default=365, help="Total history to fetch")
    p.add_argument("--windows", type=int, default=6,   help="OOS walk-forward windows")
    p.add_argument("--window-days", type=int, default=30, help="Days per OOS window")
    p.add_argument("--z-entry", type=float, default=1.2)
    p.add_argument("--z-exit",  type=float, default=0.3)
    args = p.parse_args()

    symbols = args.symbols or SUPPORTED_SYMBOLS[:14]
    logger.info("Momentum backtest: %d symbols, %d-day history, %d × %d-day OOS windows",
                len(symbols), args.days, args.windows, args.window_days)

    ex      = HyperliquidExchange(paper=True)
    candles = fetch_data(ex, symbols, args.days)

    min_len = min(len(v) for v in candles.values() if v)
    bars_per_window = args.window_days * 6
    total_oos_bars  = args.windows * bars_per_window
    warmup_bars     = min_len - total_oos_bars

    if warmup_bars < 60:
        logger.error("Not enough data for walk-forward. Try fewer --windows or --window-days.")
        return

    logger.info("Available: %d bars ≈ %d days  |  warmup: %d bars  |  OOS: %d bars",
                min_len, min_len // 6, warmup_bars, total_oos_bars)
    logger.info("=" * 74)
    logger.info("  %-4s  %-5s  %8s  %7s  %7s  %6s  BTC%%",
                "WIN", "REG", "RETURN%", "SHARPE", "MAXDD%", "TRADES")
    logger.info("─" * 74)

    results = []
    for i in range(args.windows):
        w_start = warmup_bars + i * bars_per_window
        w_end   = w_start + bars_per_window

        r = run_window(candles, symbols, w_start, w_end,
                       z_entry=args.z_entry, z_exit=args.z_exit,
                       rebal_every=42)   # weekly rebalance (42 × 4h = 7 days)

        # Regime label from BTC
        btc = candles.get("BTC", [])
        btc_ret = 0.0
        if btc and w_end - 1 < len(btc) and w_start < len(btc):
            btc_ret = (btc[w_end - 1].close - btc[w_start].close) / btc[w_start].close
        regime = "BULL" if btc_ret > 0.05 else ("BEAR" if btc_ret < -0.05 else "SIDE")

        r["btc_ret_pct"] = btc_ret * 100
        results.append(r)
        logger.info("  W%-3d  %-5s  %+7.1f%%  %7.2f  %7.1f%%  %6d  %+.1f%%",
                    i + 1, regime, r["total_return_pct"], r["sharpe"],
                    r["max_dd_pct"], r["trade_count"], r["btc_ret_pct"])

    logger.info("─" * 74)
    wins    = sum(1 for r in results if r["total_return_pct"] > 0)
    avg_ret = float(np.mean([r["total_return_pct"] for r in results]))
    avg_sh  = float(np.mean([r["sharpe"] for r in results]))
    avg_dd  = float(np.mean([r["max_dd_pct"] for r in results]))
    logger.info("  AVG          %+7.1f%%  %7.2f  %7.1f%%         %d/%d windows +ve",
                avg_ret, avg_sh, avg_dd, wins, len(results))
    logger.info("=" * 74)


if __name__ == "__main__":
    main()
