"""Transformer-based live trading on Hyperliquid perps (5m candles).

Dry-run (paper mode, no orders placed):
    python scripts/trade.py --symbols BTC,ETH,SOL

Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \
    python scripts/trade.py --symbols BTC,ETH,SOL --live
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.agent import AgentConfig, TradingAgent
from trader.exchanges.hyperliquid import HyperliquidExchange
from trader.risk import RiskConfig


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC,ETH,SOL")
    ap.add_argument("--model", default="models/transformer_5m.onnx")
    ap.add_argument("--pt", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.015)
    ap.add_argument("--t-max", type=int, default=12)
    ap.add_argument("--leverage", type=int, default=2)
    ap.add_argument("--max-position-pct", type=float, default=0.25)
    ap.add_argument("--live", action="store_true",
                    help="Place real orders. Default is paper/dry-run.")
    args = ap.parse_args()

    dry_run = not args.live
    exchange = HyperliquidExchange(paper=dry_run)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    cfg = AgentConfig(
        symbols=symbols,
        model_path=args.model,
        pt=args.pt,
        sl=args.sl,
        t_max_bars=args.t_max,
        leverage=args.leverage,
        dry_run=dry_run,
        risk=RiskConfig(max_position_pct=args.max_position_pct),
    )

    agent = TradingAgent(exchange, cfg)
    agent.run()


if __name__ == "__main__":
    main()
