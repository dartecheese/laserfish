"""Standalone momentum leg — 7-day time-series momentum on 5 symbols.

This is the momentum component in isolation.  For full portfolio trading
(3 strategies, 5/5 live-trading criteria, +55.6%/month, 98% win rate)
use portfolio_live.py instead.

Champion config (from models/best_momentum_config.json, round 23):
  Symbols:  BTC / ETH / SOL / AVAX / LINK
  Leverage: 3×   |  NZ: 0.40 (adaptive)  |  hist: 240 bars
  Features: vol_filter + adaptive_nz + circuit_breaker 20%
  OOS avg:  +26.5%/month, Sharpe 1.88, MaxDD 16.7%, 8/10 windows positive

Adaptive neutral zone: NZ scales inversely with BTC vol ratio
  (current 30-bar vol / full history vol).  Wider NZ in choppy markets
  reduces whipsaw; narrower NZ in trending markets catches more signal.

Usage:
    python scripts/momentum_trade.py --paper          # paper trading
    python scripts/momentum_trade.py --scan-only      # print signal and exit
    python scripts/momentum_trade.py                  # live (needs HL_PRIVATE_KEY)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.exchanges.hyperliquid import HyperliquidExchange
from trader.execution import SmartExecutorV2
from trader.risk import RiskConfig, RiskManager

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("momentum_trade")

SYMBOLS     = ["BTC", "ETH", "SOL", "AVAX", "LINK"]   # champion config
LEVERAGE    = 3.0           # champion config
NZ_THRESH   = 0.40          # champion config (adaptive NZ scales from this base)
MOM_BARS    = 42            # 7-day momentum (42 × 4h bars)
HIST_BARS   = 240           # champion config — longer history improves z-score stability
REBAL_BARS  = 42            # rebalance every 7 days
LOOP_SEC    = 4 * 3600      # 4h loop interval (one 4h bar)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paper",         action="store_true")
    p.add_argument("--scan-only",     action="store_true")
    p.add_argument("--symbols",       nargs="+", default=None)
    p.add_argument("--leverage",      type=float, default=LEVERAGE)
    p.add_argument("--nz",            type=float, default=NZ_THRESH,
                   help="Neutral-zone z-score base threshold (default 0.40)")
    p.add_argument("--no-adaptive-nz", action="store_true",
                   help="Disable adaptive NZ scaling (use fixed NZ threshold)")
    p.add_argument("--interval",      type=int,   default=LOOP_SEC)
    return p.parse_args()


class MomentumTrader:
    def __init__(self, ex: HyperliquidExchange, symbols: list[str],
                 leverage: float, nz: float, adaptive_nz: bool = True):
        self.ex          = ex
        self.symbols     = symbols
        self.leverage    = leverage
        self.nz          = nz
        self.adaptive_nz = adaptive_nz
        # Rolling 7-day return history per symbol (for z-score)
        self._hist: dict[str, deque[float]] = {
            s: deque(maxlen=HIST_BARS) for s in symbols
        }
        self._bar_count = 0
        self._positions: dict[str, str] = {}   # sym → "long" | "short"

    def warm_up(self) -> None:
        logger.info("Loading %d bars of 7-day return history…", HIST_BARS + MOM_BARS)
        for sym in self.symbols:
            try:
                bars = self.ex.get_candles(sym, "4h", HIST_BARS + MOM_BARS + 5)
                closes = [b.close for b in bars]
                for i in range(MOM_BARS, len(closes)):
                    ret = (closes[i] - closes[i - MOM_BARS]) / closes[i - MOM_BARS]
                    self._hist[sym].append(ret)
                logger.info("  %s: %d return readings loaded", sym, len(self._hist[sym]))
            except Exception as e:
                logger.warning("  %s warm-up failed: %s", sym, e)

    def _z_score(self, sym: str, current_ret: float) -> float | None:
        hist = list(self._hist[sym])
        if len(hist) < 15:
            return None
        mu = float(sum(hist) / len(hist))
        sd = float((sum((x - mu) ** 2 for x in hist) / len(hist)) ** 0.5) + 1e-9
        return (current_ret - mu) / sd

    def _effective_nz(self) -> float:
        """Scale NZ inversely with BTC vol ratio (trending vs choppy)."""
        if not self.adaptive_nz or "BTC" not in self.symbols:
            return self.nz
        try:
            bars   = self.ex.get_candles("BTC", "4h", HIST_BARS + MOM_BARS + 5)
            closes = [b.close for b in bars]
            if len(closes) < 60:
                return self.nz
            import math
            log_rets  = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            cur_vol   = float((sum(x ** 2 for x in log_rets[-30:]) / 30) ** 0.5)
            long_vol  = float((sum(x ** 2 for x in log_rets) / len(log_rets)) ** 0.5)
            ratio     = cur_vol / (long_vol + 1e-9)
            return float(min(max(self.nz / ratio, self.nz * 0.5), self.nz * 2.0))
        except Exception:
            return self.nz

    def scan(self) -> dict[str, str | None]:
        """Returns {sym: 'long' | 'short' | None} for each symbol."""
        nz_eff  = self._effective_nz()
        signals: dict[str, str | None] = {}
        for sym in self.symbols:
            try:
                fd = self.ex.get_funding_data(sym)
                price = fd.mark_price
            except Exception as e:
                logger.warning("%s get_funding_data failed: %s", sym, e)
                signals[sym] = None
                continue

            try:
                bars = self.ex.get_candles(sym, "4h", MOM_BARS + 2)
                if len(bars) < MOM_BARS + 1:
                    signals[sym] = None
                    continue
                ret_7d = (bars[-1].close - bars[-(MOM_BARS + 1)].close) / bars[-(MOM_BARS + 1)].close
                self._hist[sym].append(ret_7d)
            except Exception as e:
                logger.warning("%s candle fetch failed: %s", sym, e)
                signals[sym] = None
                continue

            z = self._z_score(sym, ret_7d)
            if z is None:
                signals[sym] = None
                continue

            if z > nz_eff:
                side = "long"
            elif z < -nz_eff:
                side = "short"
            else:
                side = None   # neutral zone — go flat

            logger.info("  %s  ret7d=%+.2f%%  z=%+.2f  nz_eff=%.2f  → %s",
                        sym, ret_7d * 100, z, nz_eff, side or "FLAT")
            signals[sym] = side
        return signals

    def rebalance(self, signals: dict[str, str | None], equity: float) -> None:
        """Open/close/flip positions to match target signals."""
        alloc_per_sym = equity * self.leverage / len(self.symbols)

        for sym, target in signals.items():
            current = self._positions.get(sym)
            if target == current:
                continue

            # Close existing position
            if current is not None:
                try:
                    self.ex.close_position(sym)
                    logger.info("  CLOSE %s [%s]", sym, current.upper())
                    self._positions.pop(sym, None)
                except Exception as e:
                    logger.error("  Failed to close %s: %s", sym, e)
                    continue

            # Open new position
            if target is not None:
                try:
                    fd = self.ex.get_funding_data(sym)
                    qty = alloc_per_sym / fd.mark_price
                    side = "sell" if target == "short" else "buy"
                    self.ex.place_order(sym, side, qty, leverage=int(self.leverage))
                    self._positions[sym] = target
                    logger.info("  OPEN %s %s  qty=%.4f  notional=$%.0f",
                                target.upper(), sym, qty, alloc_per_sym)
                except Exception as e:
                    logger.error("  Failed to open %s %s: %s", target, sym, e)


def run(args: argparse.Namespace) -> None:
    symbols     = args.symbols or SYMBOLS
    adaptive_nz = not args.no_adaptive_nz
    ex          = HyperliquidExchange(paper=args.paper)
    trader      = MomentumTrader(ex, symbols, args.leverage, args.nz,
                                 adaptive_nz=adaptive_nz)

    mode = "PAPER" if args.paper else ("SCAN-ONLY" if args.scan_only else "LIVE")
    logger.info("=" * 60)
    logger.info("  7-Day Momentum Trader  [%s]", mode)
    logger.info("  symbols=%s  leverage=%.1fx  nz=%.2f  adaptive_nz=%s",
                symbols, args.leverage, args.nz, adaptive_nz)
    logger.info("  Rebalances every 7 days on weekly momentum z-score")
    logger.info("=" * 60)

    trader.warm_up()

    if args.scan_only:
        signals = trader.scan()
        for sym, side in signals.items():
            logger.info("  %s → %s", sym, side or "FLAT")
        return

    bar = 0
    while True:
        try:
            bal     = ex.get_balance()
            equity  = bal.total_usd
            logger.info("Equity: $%.2f  bar=%d  positions=%s",
                        equity, bar, trader._positions)

            signals = trader.scan()

            # Only rebalance on the weekly schedule
            if bar % REBAL_BARS == 0:
                logger.info("Rebalancing…")
                trader.rebalance(signals, equity)
            else:
                logger.info("Holding (%d/%d bars until next rebalance).",
                            bar % REBAL_BARS, REBAL_BARS)

            bar += 1
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as e:
            logger.error("Loop error: %s", e, exc_info=True)

        logger.info("Sleeping %ds until next 4h bar…", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    run(build_args())
