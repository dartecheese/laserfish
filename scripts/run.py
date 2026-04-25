"""Laserfish — momentum + Transformer 5m perp scalper on Hyperliquid.

Primary entry point. Momentum (7d signal) provides direction; the Transformer
filters each entry on 5m bar timing. If no model is present, runs momentum-only.

Dry-run (paper mode, default):
    python scripts/run.py

With Transformer filter:
    python scripts/run.py --model models/transformer_5m.onnx

Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \
    python scripts/run.py --model models/transformer_5m.onnx --live
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

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

# Minimum Transformer alpha to confirm a momentum entry.
# P(BUY) - P(SELL) must exceed this threshold in the same direction as momentum.
TRANSFORMER_MIN_CONFIRM = 0.10


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
            return True  # fail open — don't block on data errors
        threshold = auto_threshold(klines, 48.0, "volume")
        bars = make_volume_bars(klines, threshold)
        if len(bars) < 32:
            return True
        seq = live_sequence(bars)
        alpha = infer_alpha(session, seq)
        confirms = (side == "buy" and alpha >= TRANSFORMER_MIN_CONFIRM) or \
                   (side == "sell" and alpha <= -TRANSFORMER_MIN_CONFIRM)
        return confirms
    except Exception:
        return True  # fail open on any error


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
    ap.add_argument("--leverage", type=float, default=2.0)
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
        max_leverage=args.leverage,
        z_entry=args.z_entry,
    )
    strategy = MomentumStrategy(exchange, cfg)

    log.info("Laserfish starting | symbols=%d | transformer=%s | dry_run=%s",
             len(symbols), session is not None, dry_run)
    log.info("Warming up price + funding histories…")
    strategy.warm_up()
    log.info("Warm-up complete. Entering scan loop (poll=%ds).", args.poll_seconds)

    open_positions: dict[str, str] = {}

    while True:
        try:
            balance = exchange.get_balance()
            equity = balance.total_usd
            positions = exchange.get_positions()
            pos_map = {p.symbol: p for p in positions}

            if risk.check_drawdown(equity):
                log.warning("Max drawdown breached — skipping entries (equity=%.0f)", equity)
            else:
                for sym, entry_side in list(open_positions.items()):
                    if strategy.should_exit(sym, entry_side):
                        log.info("%s: momentum reversed → closing %s", sym, entry_side)
                        exchange.close_position(sym)
                        del open_positions[sym]

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

                    qty = risk.size_position(
                        sig.alpha, equity, sig.mark_price, int(cfg.max_leverage),
                    )
                    if qty <= 0:
                        continue

                    log.info("%s: OPEN %s  qty=%.4f  mark=%.4f",
                             sig.symbol, sig.side.upper(), qty, sig.mark_price)

                    exchange.place_bracket_order(BracketParams(
                        symbol=sig.symbol,
                        side=sig.side,
                        quantity=qty,
                        price=None,
                        take_profit_pct=cfg.take_profit_pct,
                        stop_loss_pct=cfg.stop_loss_pct,
                        leverage=int(cfg.max_leverage),
                    ))

                    open_positions[sig.symbol] = sig.side

        except Exception as e:
            log.error("Scan loop error: %s", e)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
