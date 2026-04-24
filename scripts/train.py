"""Train the Transformer classifier on 5m data and export to ONNX.

Usage:
    # 1. Fetch data first:
    #    python scripts/fetch.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT,LINKUSDT

    # 2. Train:
    python scripts/train.py --symbols BTC,ETH,SOL,AVAX,LINK \
        --bar-type volume --out models/transformer_5m.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.data import load, cache_path
from trader.train import TrainConfig, train_walk_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC,ETH,SOL",
                    help="Base symbols (no USDT). Data files must exist in --data-dir.")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--bar-type", choices=["volume", "dollar"], default="volume")
    ap.add_argument("--target-bars-day", type=float, default=48.0)
    ap.add_argument("--cusum-h", type=float, default=0.003)
    ap.add_argument("--pt", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.015)
    ap.add_argument("--t-max", type=int, default=12)   # 12 × 5m = 1h hold
    ap.add_argument("--train-months", type=int, default=18)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)  # larger batch for 5m density
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/transformer_5m.onnx")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cached klines. Binance symbols append USDT.
    klines_by_symbol: dict = {}
    for sym in symbols:
        binance_sym = sym if sym.endswith("USDT") else f"{sym}USDT"
        path = cache_path(data_dir, binance_sym, args.interval)
        if not path.exists():
            raise FileNotFoundError(
                f"No data for {binance_sym} at {path}. "
                f"Run `python scripts/fetch.py --symbols {binance_sym}` first."
            )
        klines_by_symbol[sym] = load(path)
        print(f"Loaded {len(klines_by_symbol[sym])} klines for {sym}")

    cfg = TrainConfig(
        bar_type=args.bar_type,
        target_bars_day=args.target_bars_day,
        cusum_h=args.cusum_h,
        pt=args.pt,
        sl=args.sl,
        t_max=args.t_max,
        train_months=args.train_months,
        test_months=args.test_months,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    train_walk_forward(klines_by_symbol, cfg, out_path)


if __name__ == "__main__":
    main()
