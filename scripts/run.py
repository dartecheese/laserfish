"""Laserfish — momentum-driven 5m perp scalper on Hyperliquid.

This is the primary entry point. Runs the cross-sectional momentum strategy
immediately — no trained model required.

Dry-run (paper mode, default):
    python scripts/run.py

Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \
    python scripts/run.py --live

For the Transformer-based agent (requires training first), use scripts/trade.py.
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


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(MomentumConfig().symbols),
                    help="Comma-separated symbols to trade")
    ap.add_argument("--top-n", type=int, default=3,
                    help="Max simultaneous long and short positions each")
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--max-position-pct", type=float, default=0.20,
                    help="Max equity per position")
    ap.add_argument("--z-entry", type=float, default=1.2)
    ap.add_argument("--poll-seconds", type=int, default=300,
                    help="Seconds between scans (one 5m candle = 300s)")
    ap.add_argument("--live", action="store_true",
                    help="Place real orders. Default is paper/dry-run.")
    args = ap.parse_args()

    dry_run = not args.live
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    exchange = HyperliquidExchange(paper=dry_run)
    risk = RiskManager(RiskConfig(max_position_pct=args.max_position_pct))

    cfg = MomentumConfig(
        symbols=symbols,
        top_n=args.top_n,
        max_leverage=args.leverage,
        z_entry=args.z_entry,
    )
    strategy = MomentumStrategy(exchange, cfg)

    log.info("Laserfish starting | symbols=%d | dry_run=%s", len(symbols), dry_run)
    log.info("Warming up price + funding histories…")
    strategy.warm_up()
    log.info("Warm-up complete. Entering scan loop (poll=%ds).", args.poll_seconds)

    # Track open positions: symbol → side ("buy"|"sell")
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
                # Exit stale positions whose momentum has reversed
                for sym, entry_side in list(open_positions.items()):
                    if strategy.should_exit(sym, entry_side):
                        log.info("%s: momentum reversed → closing %s", sym, entry_side)
                        if not dry_run:
                            exchange.close_position(sym)
                        del open_positions[sym]

                # Scan for new signals
                signals = strategy.scan()
                for sig in signals:
                    log.info(MomentumStrategy.format_signal(sig))

                    if sig.symbol in open_positions or sig.symbol in pos_map:
                        continue

                    if not risk.can_open_position(sig.symbol, positions, balance, equity):
                        log.debug("%s: risk limits prevent entry", sig.symbol)
                        continue

                    qty = risk.size_position(
                        sig.alpha, equity,
                        sig.mark_price,
                        int(cfg.max_leverage),
                    )
                    if qty <= 0:
                        continue

                    log.info(
                        "%s: OPEN %s  qty=%.4f  mark=%.4f",
                        sig.symbol, sig.side.upper(), qty, sig.mark_price,
                    )

                    if not dry_run:
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
