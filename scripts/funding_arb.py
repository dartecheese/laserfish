"""Funding Rate Arbitrage — live trading loop.

Strategy: "Fundamentals of Perpetual Futures" (He, Manela, Ross, von Wachter 2022).
  - Short perp when 8h funding rate z-score > Z_ENTRY (basis is anomalously positive)
  - Long  perp when 8h funding rate z-score < -Z_ENTRY (basis is anomalously negative)
  - Exit when z-score returns to neutral (< Z_EXIT)
  - Momentum alignment (24h return) amplifies conviction per paper's R²>50% finding

Usage:
    # Paper trading (no real orders):
    python scripts/funding_arb.py --paper

    # Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... python scripts/funding_arb.py

    # Scan once and print signals (no trading):
    python scripts/funding_arb.py --scan-only

    # Custom symbols and parameters:
    python scripts/funding_arb.py --symbols BTC ETH SOL --z-entry 2.0 --max-leverage 2

Environment:
    HL_PRIVATE_KEY       Hyperliquid wallet private key
    HL_WALLET_ADDRESS    Hyperliquid wallet address
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.exchanges.hyperliquid import SUPPORTED_SYMBOLS, HyperliquidExchange
from trader.execution import SmartExecutorV2
from trader.risk import RiskConfig, RiskManager
from trader.strategies.funding_arb import FundingArbConfig, FundingArbStrategy, Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("funding_arb")
_arb_logger = logging.getLogger("trader.strategies.funding_arb")

LOOP_INTERVAL = 600   # seconds between full scan cycles (10 min)
WARMUP_LIMIT  = 90    # 8h periods to load on startup (~30 days)


# ────────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Funding rate arb bot for Hyperliquid perps")
    p.add_argument("--paper", action="store_true", help="Paper-trade mode (no real orders)")
    p.add_argument("--scan-only", action="store_true", help="Print signals and exit")
    p.add_argument("--symbols", nargs="+", default=None, help="Override symbol list")
    p.add_argument("--z-entry", type=float, default=1.5, help="Z-score entry threshold")
    p.add_argument("--z-exit",  type=float, default=0.2, help="Z-score exit threshold")
    p.add_argument("--min-funding", type=float, default=0.08,
                   help="Min annualized funding rate to consider (fraction, e.g. 0.15 = 15%%)")
    p.add_argument("--stop-loss",   type=float, default=0.03, help="Stop-loss %% (0.03 = 3%%)")
    p.add_argument("--take-profit", type=float, default=0.025, help="Take-profit %% (0.025 = 2.5%%)")
    p.add_argument("--max-leverage", type=int, default=3, help="Max leverage")
    p.add_argument("--max-positions", type=int, default=4, help="Max concurrent positions")
    p.add_argument("--no-hedge", action="store_true",
                   help="Disable spot hedge leg (directional mode, not recommended)")
    p.add_argument("--interval", type=int, default=LOOP_INTERVAL,
                   help="Loop interval in seconds")
    p.add_argument("--verbose", action="store_true",
                   help="Log per-symbol z-scores and filter reasons each scan")
    return p.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    if args.verbose:
        _arb_logger.setLevel(logging.DEBUG)
    mode = "PAPER" if args.paper else ("SCAN-ONLY" if args.scan_only else "LIVE")
    logger.info("=" * 60)
    logger.info("  Funding Rate Arbitrage Bot  [%s]", mode)
    logger.info("  He, Manela, Ross, von Wachter (2022)")
    logger.info("  z_entry=%.1f  z_exit=%.1f  lev=%dx  sl=%.1f%%  tp=%.1f%%",
                args.z_entry, args.z_exit, args.max_leverage,
                args.stop_loss * 100, args.take_profit * 100)
    logger.info("=" * 60)


# ────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print_banner(args)

    exchange = HyperliquidExchange(paper=args.paper)
    executor = SmartExecutorV2(exchange, timeout_s=30.0)

    symbols = args.symbols or SUPPORTED_SYMBOLS
    cfg = FundingArbConfig(
        symbols=symbols,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        min_funding_annualized=args.min_funding,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_leverage=args.max_leverage,
        hedge_spot=not args.no_hedge,
    )
    strategy = FundingArbStrategy(exchange, cfg)

    risk = RiskManager(RiskConfig(
        max_positions=args.max_positions,
        max_position_pct=0.20,
        max_total_exposure=0.70,
        min_alpha=cfg.min_alpha,
        max_drawdown_pct=15.0,
    ))

    logger.info("Seeding funding rate history (cache + best-effort API)…")
    strategy.warm_up()

    if args.scan_only:
        _scan_and_print(strategy)
        return

    logger.info("Entering live loop (interval=%ds)…", args.interval)
    while True:
        try:
            _loop_iteration(exchange, executor, strategy, risk, cfg, args)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as exc:
            logger.error("Loop error: %s", exc, exc_info=True)

        logger.info("Sleeping %ds until next scan…", args.interval)
        time.sleep(args.interval)


def _scan_and_print(strategy: FundingArbStrategy) -> None:
    logger.info("Scanning %d symbols…", len(strategy.cfg.symbols))
    signals = strategy.scan()
    if not signals:
        logger.info("No signals above threshold.")
        return
    logger.info("Found %d signal(s):", len(signals))
    for s in signals:
        logger.info("  %s", FundingArbStrategy.format_signal(s))


def _loop_iteration(
    exchange: HyperliquidExchange,
    executor: SmartExecutorV2,
    strategy: FundingArbStrategy,
    risk: RiskManager,
    cfg: FundingArbConfig,
    args: argparse.Namespace,
) -> None:
    balance = exchange.get_balance()
    positions = exchange.get_positions()
    equity = balance.total_usd

    logger.info("Equity: $%.2f  Open positions: %d", equity, len(positions))

    # ── Drawdown check ──────────────────────────────────────────────────
    if risk.check_drawdown(equity):
        logger.warning("Drawdown limit hit — skipping new entries.")
        _check_exits(exchange, strategy, positions)
        return

    # ── Exit stale positions ────────────────────────────────────────────
    _check_exits(exchange, strategy, positions)

    # ── Refresh positions after exits ───────────────────────────────────
    positions = exchange.get_positions()

    # ── Scan for new signals ────────────────────────────────────────────
    signals = strategy.scan()
    if signals:
        logger.info("%d signal(s) found:", len(signals))
        for s in signals:
            logger.info("  %s", FundingArbStrategy.format_signal(s))
    else:
        logger.info("No signals above threshold.")
        return

    # ── Execute top signals ─────────────────────────────────────────────
    open_syms = {p.symbol for p in positions}
    for sig in signals:
        if sig.symbol in open_syms:
            continue

        if not risk.can_open_position(sig.symbol, positions, balance, equity):
            logger.info("Risk gate blocked %s (positions=%d, exposure limit).",
                        sig.symbol, len(positions))
            break

        qty = risk.size_position(
            alpha=sig.alpha,
            equity=equity,
            asset_price=sig.mark_price,
            leverage=cfg.max_leverage,
        )
        if qty <= 0:
            logger.info("Zero size for %s (alpha=%.2f below min).", sig.symbol, sig.alpha)
            continue

        hedge_label = "HEDGED" if cfg.hedge_spot and sig.symbol in __import__(
            "trader.exchanges.hyperliquid", fromlist=["SPOT_HEDGEABLE"]
        ).SPOT_HEDGEABLE else "UNHEDGED"
        logger.info("OPEN %s %s %s  qty=%.4f  alpha=%.2f  leverage=%dx",
                    "SHORT" if sig.side == "sell" else "LONG",
                    sig.symbol, hedge_label, qty, sig.alpha, cfg.max_leverage)

        perp_order, spot_order = strategy.open_hedged_position(sig, qty)
        logger.info("  → perp %s filled @ %.4f", perp_order.order_id, perp_order.price)
        if spot_order:
            logger.info("  → spot %s filled @ %.4f", spot_order.order_id, spot_order.price)

        open_syms.add(sig.symbol)
        positions = exchange.get_positions()


def _check_exits(
    exchange: HyperliquidExchange,
    strategy: FundingArbStrategy,
    positions: list,
) -> None:
    for pos in positions:
        if strategy.should_exit(pos.symbol):
            hedge_label = "HEDGED" if strategy.is_hedged(pos.symbol) else "UNHEDGED"
            logger.info("EXIT %s [%s] — funding z-score normalized.", pos.symbol, hedge_label)
            strategy.close_hedged_position(pos.symbol)
        else:
            logger.debug("HOLD %s  pnl=%.2f", pos.symbol, pos.unrealized_pnl)


# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(build_args())
