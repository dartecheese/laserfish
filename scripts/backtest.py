"""Run a backtest on historical data using a trained ONNX model.

Usage:
    python scripts/backtest.py --symbol BTC --interval 1h \
        --model models/transformer.onnx --data-dir data/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.bars import make_volume_bars, make_dollar_bars, auto_threshold
from trader.backtest import BacktestConfig, run_backtest
from trader.data import load, cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="models/transformer.onnx")
    ap.add_argument("--bar-type", choices=["volume", "dollar"], default="volume")
    ap.add_argument("--target-bars-day", type=float, default=20.0)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--fee-bps", type=float, default=3.5)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pt", type=float, default=0.025)
    ap.add_argument("--sl", type=float, default=0.025)
    ap.add_argument("--t-max", type=int, default=24)
    args = ap.parse_args()

    sym = args.symbol.upper()
    binance_sym = sym if sym.endswith("USDT") else f"{sym}USDT"
    path = cache_path(Path(args.data_dir), binance_sym, args.interval)

    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path}. Run fetch.py first.")

    klines = load(path)
    print(f"Loaded {len(klines)} klines")

    threshold = auto_threshold(klines, args.target_bars_day, args.bar_type)
    if args.bar_type == "volume":
        bars = make_volume_bars(klines, threshold)
    else:
        bars = make_dollar_bars(klines, threshold)
    print(f"Constructed {len(bars)} {args.bar_type} bars (threshold={threshold:.0f})")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found at {args.model}. Run train.py first.")

    cfg = BacktestConfig(
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        pt=args.pt,
        sl=args.sl,
        t_max=args.t_max,
    )

    print(f"\nRunning backtest on {sym} with {len(bars)} bars…")
    result = run_backtest(bars, args.model, cfg)
    print(f"\n{result.summary()}")
    print(f"Final equity: ${result.equity_curve[-1]:,.0f}")


if __name__ == "__main__":
    main()
