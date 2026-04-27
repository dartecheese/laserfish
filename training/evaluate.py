"""OOS evaluation of a trained PPO agent.

Runs the saved model on a held-out period and prints:
  - Total return, annualized Sharpe, Calmar ratio, max drawdown
  - Equity curve saved to models/eval_curve.csv
  - Trade log saved to models/eval_trades.csv

Usage:
    python training/evaluate.py
    python training/evaluate.py --model models/rl_btc_perp.zip --data data/BTCUSDT_5m.json --start 2025-10-01
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

BARS_PER_DAY = 96


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rl_btc_perp.zip")
    parser.add_argument("--data",  default="data/BTCUSDT_5m.json")
    parser.add_argument("--start", default=None, help="ISO date to start eval period")
    parser.add_argument("--end",   default=None)
    parser.add_argument("--balance", type=float, default=10_000.0)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from training.train import load_data, build_features, df_to_env_array, make_env

    logger.info("Loading model: %s", args.model)
    model = PPO.load(args.model)

    logger.info("Loading data: %s", args.data)
    df = load_data(args.data)
    df, _ = build_features(df)

    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <  args.end]

    logger.info("Evaluating on %d bars (%s → %s)", len(df), df.index[0], df.index[-1])

    feat, prices = df_to_env_array(df)
    env = make_env(feat, prices, initial_balance=args.balance)

    obs, _ = env.reset()
    equity_curve = [args.balance]
    positions    = [0.0]
    timestamps   = [df.index[0]]
    trades       = []
    done = truncated = False
    step = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        eq  = info.get("equity", equity_curve[-1])
        pos = info.get("position", 0.0)
        equity_curve.append(eq)
        positions.append(pos)
        if step < len(df) - 1:
            timestamps.append(df.index[step + 1])
        step += 1

    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)

    peak    = np.maximum.accumulate(equity_curve)
    dd      = (peak - equity_curve) / (peak + 1e-9)
    max_dd  = float(dd.max())
    total_r = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    sharpe  = float(np.mean(returns)) / (float(np.std(returns)) + 1e-9) * np.sqrt(BARS_PER_DAY * 365)
    calmar  = total_r / (max_dd + 1e-9)

    print("\n" + "=" * 55)
    print(f"  RL Agent OOS Evaluation")
    print(f"  Period  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Bars    : {len(df)}")
    print("-" * 55)
    print(f"  Return  : {total_r * 100:+.2f}%")
    print(f"  Sharpe  : {sharpe:.3f}")
    print(f"  Calmar  : {calmar:.3f}")
    print(f"  Max DD  : {max_dd * 100:.2f}%")
    print(f"  Trades  : {env._trade_count}")
    print("=" * 55 + "\n")

    out = Path("models")
    out.mkdir(exist_ok=True)

    curve_df = pd.DataFrame({"equity": equity_curve[1:], "position": positions[1:]},
                             index=timestamps[:len(equity_curve) - 1])
    curve_df.to_csv(out / "eval_curve.csv")
    logger.info("Equity curve → %s", out / "eval_curve.csv")


if __name__ == "__main__":
    main()
