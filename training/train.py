"""PPO walk-forward training loop for the Hyperliquid BTC perp agent.

Rolling-window walk-forward:
  - Train window: 12 months (~35,040 bars at 15m)
  - Val window:    2 months
  - Test window:   1 month (OOS evaluation)
  - Step:          1 month (advance one month per fold)

Training:
  - PPO with MlpPolicy, hidden layers [256, 256, 128]
  - Device: MPS (Apple Silicon) or CPU
  - EvalCallback saves best model per fold
  - Early stopping if no improvement after 3 eval rounds

Output:
  - models/rl_btc_perp_{fold}.zip  — best per fold
  - models/rl_btc_perp.zip         — copy of the best fold overall

Usage:
    python training/train.py
    python training/train.py --data data/BTCUSDT_5m.json --folds 6 --timesteps 500000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

BARS_PER_HOUR  = 4    # 15m bars
BARS_PER_DAY   = 96
BARS_PER_MONTH = BARS_PER_DAY * 30
TRAIN_MONTHS   = 12
VAL_MONTHS     = 2
TEST_MONTHS    = 1


# ------------------------------------------------------------------ #
# Data loading                                                        #
# ------------------------------------------------------------------ #

def load_data(data_path: str) -> pd.DataFrame:
    """Load BTCUSDT JSON cache → DataFrame with close, volume, funding_rate columns.

    Handles two formats:
      - list of dicts with keys: open_time, open, high, low, close, volume
      - list of lists: [ts_ms, open, high, low, close, volume, ...]
    """
    with open(data_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("candles", []))

    if not raw:
        raise ValueError(f"No data in {data_path}")

    if isinstance(raw[0], dict):
        ts_key = "open_time" if "open_time" in raw[0] else "timestamp"
        df = pd.DataFrame(raw)[
            [ts_key, "open", "high", "low", "close", "volume"]
        ].rename(columns={ts_key: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    else:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame([r[:6] for r in raw], columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)

    df = df.set_index("timestamp").sort_index().astype(float)

    # Resample 5m → 15m if needed
    inferred_freq = _infer_freq(df)
    if inferred_freq == "5min":
        df = df.resample("15min").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    df["funding_rate"] = 0.0
    return df


def _infer_freq(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "unknown"
    delta = (df.index[1] - df.index[0]).seconds
    return f"{delta // 60}min"


# ------------------------------------------------------------------ #
# Feature + regime pipeline                                           #
# ------------------------------------------------------------------ #

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return (feature_df, price_array).
    price_array shape: (N,) — raw close prices for env.price_col.
    """
    from src.data.features import FeatureEngineer, FEATURE_COLUMNS
    from src.regime.detector import RegimeDetector

    fe = FeatureEngineer()
    df = fe.build(df)

    # Fit regime detector on returns + funding
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
    funding = df["funding_rate"].values[1:]  # align
    det = RegimeDetector()
    det.fit(returns, funding)

    labels, probs = det.predict(returns, funding)
    # Align back to df (drop first row due to return diff)
    df = df.iloc[1:].copy()
    df["regime_label"] = labels.astype(float)
    df["regime_prob"]  = probs

    # regime_duration_norm: bars since last regime change, normalized
    durations = np.zeros(len(labels))
    count = 0
    for i in range(len(labels)):
        if i > 0 and labels[i] != labels[i-1]:
            count = 0
        count += 1
        durations[i] = count
    df["regime_duration_norm"] = np.minimum(durations / (BARS_PER_DAY * 30), 1.0)

    df = df.dropna(subset=FEATURE_COLUMNS)
    return df, df["close"].values


def df_to_env_array(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (feature_array, price_array) for the env."""
    from src.data.features import FeatureEngineer, FEATURE_COLUMNS
    fe = FeatureEngineer()
    feat = fe.to_array(df)
    prices = df["close"].values.astype(np.float32)
    return feat, prices


# ------------------------------------------------------------------ #
# Training helpers                                                    #
# ------------------------------------------------------------------ #

def make_env(feat: np.ndarray, price_col_data: np.ndarray, initial_balance: float = 10_000.0):
    """Build HyperliquidBTCEnv with a dedicated price column prepended."""
    from src.agents.env import HyperliquidBTCEnv
    # Prepend raw price as column 0, then features
    data = np.column_stack([price_col_data.reshape(-1, 1), feat])
    env = HyperliquidBTCEnv(
        data=data,
        price_col=0,
        funding_col=10,   # funding_annualized is index 9 in FEATURE_COLUMNS, +1 for price col
        initial_balance=initial_balance,
        use_maker=True,
        slippage_bps=2.0,
        reward_scaling=100.0,
    )
    return env


def evaluate_policy(model, env) -> dict:
    """Run one full episode, return performance metrics."""
    obs, _ = env.reset()
    equity_curve = [env.initial_balance]
    done = truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info.get("equity", equity_curve[-1]))

    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / (peak + 1e-9)
    max_dd = float(drawdowns.max())

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    sharpe = (
        float(np.mean(returns)) / (float(np.std(returns)) + 1e-9) * np.sqrt(BARS_PER_DAY * 365)
        if len(returns) > 0 else 0.0
    )
    calmar = total_return / (max_dd + 1e-9)

    return {
        "total_return": total_return,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "max_dd":       max_dd,
        "n_bars":       len(equity_curve),
    }


# ------------------------------------------------------------------ #
# Walk-forward loop                                                   #
# ------------------------------------------------------------------ #

def walk_forward_train(
    df: pd.DataFrame,
    out_dir: Path,
    n_folds: int = 6,
    timesteps: int = 300_000,
) -> dict:
    """
    Rolling walk-forward training.
    Returns dict of fold metrics and path to best model.
    """
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"
    logger.info("Training device: %s", device)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnNoModelImprovement,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    total_bars = len(df)
    train_n = TRAIN_MONTHS * BARS_PER_MONTH
    val_n   = VAL_MONTHS   * BARS_PER_MONTH
    test_n  = TEST_MONTHS  * BARS_PER_MONTH
    window  = train_n + val_n + test_n

    if total_bars < window:
        logger.warning("Not enough data for walk-forward (%d < %d bars)", total_bars, window)
        n_folds = 1
        train_n = int(total_bars * 0.70)
        val_n   = int(total_bars * 0.15)
        test_n  = total_bars - train_n - val_n
        window  = total_bars

    all_metrics = []
    best_sharpe = -np.inf
    best_model_path = None

    for fold in range(n_folds):
        start = fold * BARS_PER_MONTH
        end   = start + window
        if end > total_bars:
            logger.info("Fold %d: not enough data, stopping at fold %d", fold, fold)
            break

        fold_df = df.iloc[start:end]
        train_df = fold_df.iloc[:train_n]
        val_df   = fold_df.iloc[train_n : train_n + val_n]
        test_df  = fold_df.iloc[train_n + val_n :]

        logger.info(
            "Fold %d/%d | train=%s→%s (%d bars) | val=%d | test=%d",
            fold + 1, n_folds,
            train_df.index[0].strftime("%Y-%m-%d"),
            train_df.index[-1].strftime("%Y-%m-%d"),
            len(train_df), len(val_df), len(test_df),
        )

        # Build train + val envs
        train_feat, train_prices = df_to_env_array(train_df)
        val_feat,   val_prices   = df_to_env_array(val_df)

        train_env = make_env(train_feat, train_prices)
        val_env   = make_env(val_feat,   val_prices)

        # PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256, 128]},
        )

        fold_model_path = str(out_dir / f"rl_btc_perp_fold{fold}")
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, min_evals=5, verbose=0
        )
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=str(out_dir),
            log_path=str(out_dir / "logs"),
            eval_freq=max(timesteps // 20, 2048),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
            callback_after_eval=stop_callback,
        )

        model.learn(total_timesteps=timesteps, callback=eval_callback)

        # Rename best_model → fold-specific name
        best_path = out_dir / "best_model.zip"
        if best_path.exists():
            fold_zip = out_dir / f"rl_btc_perp_fold{fold}.zip"
            shutil.copy(best_path, fold_zip)

        # Evaluate on test set
        if len(test_df) > 100:
            test_feat, test_prices = df_to_env_array(test_df)
            test_env = make_env(test_feat, test_prices)
            metrics = evaluate_policy(model, test_env)
            metrics["fold"] = fold
            all_metrics.append(metrics)

            logger.info(
                "Fold %d OOS | return=%+.1f%% | Sharpe=%.2f | Calmar=%.2f | MaxDD=%.1f%%",
                fold + 1,
                metrics["total_return"] * 100,
                metrics["sharpe"],
                metrics["calmar"],
                metrics["max_dd"] * 100,
            )

            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_model_path = str(out_dir / f"rl_btc_perp_fold{fold}.zip")

    # Copy best overall model to canonical path
    final_path = out_dir / "rl_btc_perp.zip"
    if best_model_path and Path(best_model_path).exists():
        shutil.copy(best_model_path, final_path)
        logger.info("Best model (Sharpe=%.2f) → %s", best_sharpe, final_path)
    elif (out_dir / "best_model.zip").exists():
        shutil.copy(out_dir / "best_model.zip", final_path)

    return {"folds": all_metrics, "best_sharpe": best_sharpe, "model_path": str(final_path)}


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="PPO walk-forward training")
    parser.add_argument("--data",       default="data/BTCUSDT_5m.json", help="Path to price data JSON")
    parser.add_argument("--out",        default="models",               help="Output directory")
    parser.add_argument("--folds",      type=int,   default=6,          help="Number of walk-forward folds")
    parser.add_argument("--timesteps",  type=int,   default=300_000,    help="PPO timesteps per fold")
    parser.add_argument("--balance",    type=float, default=10_000.0,   help="Initial paper balance")
    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = load_data(args.data)
    logger.info("Loaded %d bars (%s → %s)", len(df), df.index[0], df.index[-1])

    logger.info("Building features + regime labels…")
    df, _ = build_features(df)
    logger.info("Feature matrix ready: %d bars × %d features", *df.shape[:2])

    results = walk_forward_train(
        df,
        out_dir=Path(args.out),
        n_folds=args.folds,
        timesteps=args.timesteps,
    )

    logger.info("=" * 60)
    logger.info("Walk-forward complete")
    if results["folds"]:
        sharpes = [m["sharpe"] for m in results["folds"]]
        returns = [m["total_return"] for m in results["folds"]]
        max_dds = [m["max_dd"] for m in results["folds"]]
        logger.info(
            "Mean OOS | return=%+.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            np.mean(returns) * 100, np.mean(sharpes), np.mean(max_dds) * 100,
        )
    logger.info("Best model: %s", results["model_path"])


if __name__ == "__main__":
    main()
