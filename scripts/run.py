"""Laserfish — momentum + Transformer + HMM regime + dynamic leverage on Hyperliquid.

Signal stack (in order of application):
  1. 7d cross-sectional momentum (primary direction signal)
  2. HMM regime filter (3-state: trend/range/crisis — gates entries + scales leverage)
  3. Dynamic Kelly leverage (vol-targeting + regime multiplier + drawdown circuit breaker)
  4. Transformer entry filter (P(BUY)-P(SELL) must agree with momentum direction)

Dry-run (paper mode, default):
    python scripts/run.py

With Transformer filter:
    python scripts/run.py --model models/transformer_5m.onnx

Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \\
    python scripts/run.py --model models/transformer_5m.onnx --live
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.exchanges.hyperliquid import HyperliquidExchange
from trader.exchanges.base import BracketParams
from trader.strategies.momentum import MomentumConfig, MomentumStrategy
from trader.risk import RiskConfig, RiskManager
from trader.bars import make_volume_bars, auto_threshold
from trader.data import Kline
from trader.features import live_sequence
from trader.model import load_onnx, infer_alpha
from trader.regime import RegimeDetector, REGIME_LABEL, REGIME_COLORS
from trader.leverage import DynamicLeverage, LeverageConfig

# Minimum Transformer alpha to confirm a momentum entry.
TRANSFORMER_MIN_CONFIRM = 0.10

# Do not enter new positions in crisis regime — only manage existing ones.
BLOCK_ENTRIES_IN_CRISIS = True


def _candles_to_klines(candles) -> list[Kline]:
    return [Kline(
        open_time=int(c.timestamp), open=float(c.open), high=float(c.high),
        low=float(c.low), close=float(c.close), volume=float(c.volume),
        close_time=int(c.timestamp) + 300_000, quote_volume=float(c.quote_volume), trades=0,
    ) for c in candles]


def transformer_confirms(session, exchange, symbol: str, side: str) -> bool:
    """Return True if the Transformer agrees with the momentum direction."""
    try:
        candles = exchange.get_candles(symbol, "5m", 500)
        klines = _candles_to_klines(candles)
        if not klines:
            return True
        threshold = auto_threshold(klines, 48.0, "volume")
        bars = make_volume_bars(klines, threshold)
        if len(bars) < 32:
            return True
        seq = live_sequence(bars)
        alpha = infer_alpha(session, seq)
        return (side == "buy" and alpha >= TRANSFORMER_MIN_CONFIRM) or \
               (side == "sell" and alpha <= -TRANSFORMER_MIN_CONFIRM)
    except Exception:
        return True


class RegimeTracker:
    """
    Maintains per-symbol price/funding/OI history for regime feature construction,
    fits the HMM on BTC anchor data during warm-up, and updates online each scan.
    """

    def __init__(self, exchange: HyperliquidExchange, anchor: str = "BTC"):
        self.ex = exchange
        self.anchor = anchor
        self.detector = RegimeDetector()

        # Rolling history for anchor symbol
        self._prices:   deque[float] = deque(maxlen=1000)
        self._funding:  deque[float] = deque(maxlen=200)
        self._oi:       deque[float] = deque(maxlen=200)

    def warm_up(self, n_candles: int = 500) -> None:
        """Fetch BTC anchor history and fit HMM."""
        log = logging.getLogger(__name__)
        try:
            candles = self.ex.get_candles(self.anchor, "5m", n_candles)
            for c in candles:
                self._prices.append(float(c.close))

            hist = self.ex.get_funding_rate_history(self.anchor, limit=100)
            for _, rate in hist:
                self._funding.append(rate)

            oi = self.ex.get_open_interest(self.anchor)
            for _ in range(50):
                self._oi.append(oi)  # seed with current value

            # Build obs matrix from price history
            obs_list = []
            prices = list(self._prices)
            funding = list(self._funding)
            oi_list = list(self._oi)
            for i in range(300, len(prices)):
                obs = RegimeDetector.make_obs(prices[:i+1], funding, oi_list)
                if obs is not None:
                    obs_list.append(obs)
                    self.detector.add_obs(obs)

            if obs_list:
                self.detector.fit()
                log.info("Regime HMM fitted | current state: %s %s",
                         REGIME_COLORS[self.detector.regime()],
                         self.detector.regime_label())
            else:
                log.warning("Regime HMM: insufficient obs for fit — defaulting to TREND")
        except Exception as e:
            log.warning("Regime warm-up failed: %s — defaulting to TREND state", e)

    def update(self) -> int:
        """Fetch latest BTC anchor data and return current regime state."""
        try:
            candles = self.ex.get_candles(self.anchor, "5m", 2)
            if candles:
                self._prices.append(float(candles[-1].close))
            fd = self.ex.get_funding_data(self.anchor)
            self._funding.append(fd.funding_rate_8h)
            oi = self.ex.get_open_interest(self.anchor)
            self._oi.append(oi)

            obs = RegimeDetector.make_obs(
                list(self._prices), list(self._funding), list(self._oi)
            )
            if obs is not None:
                return self.detector.regime(obs)
        except Exception:
            pass
        return self.detector.regime()


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(MomentumConfig().symbols))
    ap.add_argument("--model", default="models/transformer_5m.onnx",
                    help="ONNX model path. Omit or set to '' to run momentum-only.")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--max-leverage", type=float, default=5.0,
                    help="Hard cap on dynamic leverage (default 5x).")
    ap.add_argument("--target-vol", type=float, default=0.50,
                    help="Annualized vol target for Kelly sizing (default 0.50 = 50%%).")
    ap.add_argument("--max-position-pct", type=float, default=0.20)
    ap.add_argument("--z-entry", type=float, default=1.8)
    ap.add_argument("--poll-seconds", type=int, default=172800)  # 48h — tuned R7
    ap.add_argument("--live", action="store_true",
                    help="Place real orders. Default is paper/dry-run.")
    args = ap.parse_args()

    dry_run = not args.live
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Load Transformer if available
    model_path = Path(args.model) if args.model else None
    session = None
    if model_path and model_path.exists():
        session = load_onnx(model_path)
        log.info("Transformer filter loaded: %s", model_path)
    else:
        log.info("No model found — running momentum-only mode.")

    exchange = HyperliquidExchange(paper=dry_run)
    risk = RiskManager(RiskConfig(max_position_pct=args.max_position_pct))

    cfg = MomentumConfig(
        symbols=symbols,
        top_n=args.top_n,
        max_leverage=args.max_leverage,
        z_entry=args.z_entry,
    )
    strategy = MomentumStrategy(exchange, cfg)

    # Dynamic leverage engine
    lev_cfg = LeverageConfig(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )
    lev_engine = DynamicLeverage(lev_cfg)

    # HMM regime tracker (anchored on BTC)
    regime_tracker = RegimeTracker(exchange, anchor="BTC")

    log.info("Laserfish v2 starting | symbols=%d | transformer=%s | dynamic_leverage=True | dry_run=%s",
             len(symbols), session is not None, dry_run)

    log.info("Warming up momentum strategy…")
    strategy.warm_up()

    log.info("Warming up HMM regime detector…")
    regime_tracker.warm_up()

    log.info("Warm-up complete. Entering scan loop (poll=%ds).", args.poll_seconds)

    open_positions: dict[str, str] = {}

    while True:
        try:
            balance = exchange.get_balance()
            equity = balance.total_usd
            positions = exchange.get_positions()
            pos_map = {p.symbol: p for p in positions}

            # Update high-water mark for drawdown tracking
            lev_engine.update_hwm(equity)
            drawdown = lev_engine.daily_drawdown(equity)

            # Update regime state
            regime_state = regime_tracker.update()
            regime_mult = regime_tracker.detector.leverage_multiplier()
            regime_name = regime_tracker.detector.regime_label()
            regime_icon = REGIME_COLORS.get(regime_state, "?")

            log.info("Equity=%.0f  Drawdown=%.1f%%  Regime=%s %s  RegimeMult=%.1fx",
                     equity, drawdown * 100, regime_icon, regime_name, regime_mult)

            # Hard circuit breaker from risk manager
            if risk.check_drawdown(equity):
                log.warning("Max drawdown breached — skipping entries (equity=%.0f)", equity)
            else:
                # Exits: check momentum reversal regardless of regime
                for sym, entry_side in list(open_positions.items()):
                    if strategy.should_exit(sym, entry_side):
                        log.info("%s: momentum reversed → closing %s", sym, entry_side)
                        exchange.close_position(sym)
                        del open_positions[sym]

                # Block new entries in crisis regime
                if BLOCK_ENTRIES_IN_CRISIS and regime_state == 2:
                    log.info("Regime=CRISIS — blocking new entries, managing existing positions only")
                else:
                    signals = strategy.scan()
                    for sig in signals:
                        log.info(MomentumStrategy.format_signal(sig))

                        if sig.symbol in open_positions or sig.symbol in pos_map:
                            continue

                        if not risk.can_open_position(sig.symbol, positions, balance, equity):
                            log.debug("%s: risk limits prevent entry", sig.symbol)
                            continue

                        # Transformer confirmation gate
                        if session is not None:
                            if not transformer_confirms(session, exchange, sig.symbol, sig.side):
                                log.info("%s: Transformer veto — skipping %s", sig.symbol, sig.side)
                                continue
                            log.debug("%s: Transformer confirmed", sig.symbol)

                        # Dynamic leverage: vol-target + regime multiplier + circuit breaker
                        realized_vol = sig.vol_1w   # 7d realized vol from momentum signal
                        lev = lev_engine.compute(
                            realized_vol_24h=realized_vol,
                            regime_multiplier=regime_mult,
                            drawdown_pct=drawdown,
                            funding_z=abs(sig.funding_z),
                        )

                        qty = lev_engine.size_position(
                            equity=equity,
                            mark_price=sig.mark_price,
                            leverage=lev,
                            alpha=sig.alpha,
                            max_position_pct=args.max_position_pct,
                        )
                        if qty <= 0:
                            continue

                        log.info("%s: OPEN %s  qty=%.4f  mark=%.4f  lev=%.2fx  regime=%s",
                                 sig.symbol, sig.side.upper(), qty, sig.mark_price,
                                 lev, regime_name)

                        exchange.place_bracket_order(BracketParams(
                            symbol=sig.symbol,
                            side=sig.side,
                            quantity=qty,
                            price=None,
                            take_profit_pct=cfg.take_profit_pct,
                            stop_loss_pct=cfg.stop_loss_pct,
                            leverage=int(round(lev)),
                        ))

                        open_positions[sig.symbol] = sig.side

        except Exception as e:
            log.error("Scan loop error: %s", e)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
