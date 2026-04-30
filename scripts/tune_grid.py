"""Grid spacing tuner — sweeps spacing values per symbol using Monte Carlo windows.

Regime labels are computed once per symbol, then every (spacing, order_size) combo
is evaluated across the same sampled windows. Scoring: median Sharpe × win_rate.

Best configs are printed and optionally written back to scripts/run.py.

Usage:
    python scripts/tune_grid.py
    python scripts/tune_grid.py --symbols BTC ETH SOL --n 200 --no-write
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
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
# Search space                                                        #
# ------------------------------------------------------------------ #

SPACINGS_TO_TEST = {
    "BTC": [0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
    "ETH": [0.005, 0.006, 0.007, 0.008, 0.009, 0.010],
    "SOL": [0.006, 0.007, 0.008, 0.009, 0.010, 0.012],
}
ORDER_SIZES_TO_TEST = [0.03, 0.04, 0.05]

# Fixed params
FEE            = 0.00015
N_LEVELS       = 10
MAX_LEVERAGE   = 2.0
INITIAL_EQUITY = 10_000.0
DRIFT_PCT      = 0.04
CONFIRM_N      = 3
WINDOW_DAYS    = 30
MIN_GAP_DAYS   = 15


# ------------------------------------------------------------------ #
# Data loading                                                        #
# ------------------------------------------------------------------ #

def load_symbol(symbol: str, data_dir: str = "data") -> pd.DataFrame | None:
    path = Path(data_dir) / f"{symbol}USDT_5m.json"
    if not path.exists():
        logger.warning("No data file: %s", path)
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw[0], dict):
        ts_key = "open_time" if "open_time" in raw[0] else "timestamp"
        df = pd.DataFrame(raw)[[ts_key, "open", "high", "low", "close", "volume"]].rename(
            columns={ts_key: "timestamp"})
    else:
        df = pd.DataFrame([r[:6] for r in raw],
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().astype(float)
    df = df.resample("15min").agg({"open": "first", "high": "max",
                                   "low": "min", "close": "last", "volume": "sum"}).dropna()
    df["funding_rate"] = 0.0
    return df


def compute_regime_labels(closes: np.ndarray, funding: np.ndarray) -> np.ndarray:
    from src.regime.detector import RegimeDetector
    returns = np.diff(np.log(closes + 1e-9))
    fund    = funding[1:]
    det = RegimeDetector()
    if len(returns) < 200:
        return np.ones(len(closes), dtype=int)
    det.fit(returns, fund)
    labels, _ = det.predict(returns, fund)
    return np.concatenate([[labels[0]], labels])


def sample_windows(n_total: int, window_bars: int, n_samples: int,
                   min_gap: int, rng: np.random.Generator) -> list[int]:
    max_start = n_total - window_bars
    if max_start <= 0:
        return []
    starts: list[int] = []
    attempts = 0
    while len(starts) < n_samples and attempts < n_samples * 20:
        s = int(rng.integers(0, max_start))
        if all(abs(s - e) >= min_gap for e in starts):
            starts.append(s)
        attempts += 1
    return sorted(starts)


# ------------------------------------------------------------------ #
# Fast vectorised simulation (no PaperGrid object overhead)          #
# ------------------------------------------------------------------ #

def simulate_windows(
    closes: np.ndarray,
    labels: np.ndarray,
    starts: list[int],
    window_bars: int,
    spacing: float,
    order_size_pct: float,
) -> pd.DataFrame:
    """Run all windows for a single (spacing, order_size_pct) combo."""
    rows = []
    for s in starts:
        wc = closes[s: s + window_bars]
        wl = labels[s: s + window_bars]
        rows.append(_run_one(wc, wl, spacing, order_size_pct))
    return pd.DataFrame(rows)


def _run_one(closes: np.ndarray, labels: np.ndarray,
             spacing: float, order_size_pct: float) -> dict:
    # Inline grid state — avoids class overhead for tight loop
    active        = False
    center        = 0.0
    levels: dict  = {}
    pending_buys: dict  = {}
    pending_sells: dict = {}
    net_qty       = 0.0
    pnl           = 0.0
    n_trips       = 0
    open_events   = 0
    active_bars   = 0
    equity_curve  = [INITIAL_EQUITY]

    for price, regime in zip(closes, labels):
        # Stability filter (simplified — uses pre-confirmed labels directly
        # since we run full confirm_n logic in the labels array from MC)
        pass

    # Re-run with stability filter inline
    active        = False
    center        = 0.0
    levels        = {}
    pending_buys  = {}
    pending_sells = {}
    pnl           = 0.0
    n_trips       = 0
    open_events   = 0
    active_bars   = 0
    regime_hist: list[int] = []
    stable_regime = 0
    equity        = INITIAL_EQUITY
    equity_curve  = [INITIAL_EQUITY]

    for price, regime in zip(closes, labels):
        regime_hist.append(int(regime))
        if len(regime_hist) > CONFIRM_N:
            regime_hist.pop(0)
        if len(regime_hist) == CONFIRM_N and len(set(regime_hist)) == 1:
            stable_regime = regime_hist[-1]

        if stable_regime == 1:
            if not active:
                # Open grid
                center = price
                active = True
                levels = {}
                pending_buys  = {}
                pending_sells = {}
                qty = equity * order_size_pct * MAX_LEVERAGE / price
                for i in range(1, N_LEVELS + 1):
                    levels[-i] = {"side": "buy",  "price": price * (1 - spacing * i), "qty": qty, "filled": False}
                    levels[+i] = {"side": "sell", "price": price * (1 + spacing * i), "qty": qty, "filled": False}
                open_events += 1
            else:
                # Recenter if drifted
                if abs(price - center) / center > DRIFT_PCT:
                    active = False
                    levels = {}
                    center = price
                    active = True
                    qty = equity * order_size_pct * MAX_LEVERAGE / price
                    for i in range(1, N_LEVELS + 1):
                        levels[-i] = {"side": "buy",  "price": price * (1 - spacing * i), "qty": qty, "filled": False}
                        levels[+i] = {"side": "sell", "price": price * (1 + spacing * i), "qty": qty, "filled": False}
                    open_events += 1
                else:
                    for idx in list(levels.keys()):
                        lvl = levels[idx]
                        if lvl["filled"]:
                            continue
                        filled = (lvl["side"] == "buy"  and price <= lvl["price"]) or \
                                 (lvl["side"] == "sell" and price >= lvl["price"])
                        if not filled:
                            continue
                        lvl["filled"] = True
                        fill_px = lvl["price"]
                        qty_f   = lvl["qty"]
                        if lvl["side"] == "buy":
                            new_idx   = idx + 1
                            new_price = center * (1 + spacing * new_idx)
                            if new_idx in pending_sells:
                                sp = pending_sells.pop(new_idx)
                                pnl += (sp - fill_px) * qty_f - FEE * 2 * qty_f * fill_px
                                n_trips += 1
                            else:
                                pending_buys[new_idx] = fill_px
                            levels[new_idx] = {"side": "sell", "price": new_price, "qty": qty_f, "filled": False}
                        else:
                            new_idx   = idx - 1
                            new_price = center * (1 - spacing * abs(new_idx))
                            if idx in pending_buys:
                                bp = pending_buys.pop(idx)
                                pnl += (fill_px - bp) * qty_f - FEE * 2 * qty_f * bp
                                n_trips += 1
                            else:
                                pending_sells[idx] = fill_px
                            levels[new_idx] = {"side": "buy", "price": new_price, "qty": qty_f, "filled": False}
            active_bars += 1
        else:
            if active:
                active = False
                levels = {}

        equity = INITIAL_EQUITY + pnl
        equity_curve.append(equity)

    eq      = np.array(equity_curve)
    rets    = np.diff(eq) / (eq[:-1] + 1e-9)
    max_dd  = float(((np.maximum.accumulate(eq) - eq) / (np.maximum.accumulate(eq) + 1e-9)).max())
    sharpe  = (float(np.mean(rets)) / (float(np.std(rets)) + 1e-9) * np.sqrt(4 * 365)
               if len(rets) > 1 else 0.0)

    return {
        "pnl":          pnl,
        "pnl_pct":      pnl / INITIAL_EQUITY * 100,
        "sharpe":       sharpe,
        "max_dd_pct":   max_dd * 100,
        "round_trips":  n_trips,
        "uptime_pct":   active_bars / max(len(closes), 1) * 100,
        "open_events":  open_events,
    }


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--n",        type=int,  default=200)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-write", action="store_true",
                        help="Skip writing best config back to scripts/run.py")
    args = parser.parse_args()

    rng          = np.random.default_rng(args.seed)
    window_bars  = WINDOW_DAYS * 96
    min_gap_bars = MIN_GAP_DAYS * 96
    best_configs: dict[str, dict] = {}
    all_rows     = []

    for sym in args.symbols:
        t0 = time.time()
        df = load_symbol(sym, args.data_dir)
        if df is None:
            continue

        closes  = df["close"].values
        funding = df["funding_rate"].values

        logger.info("Fitting regime for %s (%d bars)…", sym, len(closes))
        labels = compute_regime_labels(closes, funding)
        starts = sample_windows(len(closes), window_bars, args.n, min_gap_bars, rng)
        logger.info("  %d windows — sweeping %d spacings × %d order sizes…",
                    len(starts),
                    len(SPACINGS_TO_TEST[sym]),
                    len(ORDER_SIZES_TO_TEST))

        results = []
        for spacing in SPACINGS_TO_TEST[sym]:
            for os_pct in ORDER_SIZES_TO_TEST:
                df_w = simulate_windows(closes, labels, starts, window_bars, spacing, os_pct)
                win_rate  = (df_w["pnl"] > 0).mean()
                med_sharpe = df_w["sharpe"].median()
                score     = med_sharpe * win_rate
                results.append({
                    "symbol":         sym,
                    "spacing":        spacing,
                    "order_size_pct": os_pct,
                    "score":          score,
                    "median_pnl":     df_w["pnl"].median(),
                    "median_pnl_pct": df_w["pnl_pct"].median(),
                    "median_sharpe":  med_sharpe,
                    "p5_pnl":         df_w["pnl"].quantile(0.05),
                    "p95_pnl":        df_w["pnl"].quantile(0.95),
                    "win_rate":       win_rate,
                    "median_dd":      df_w["max_dd_pct"].median(),
                    "median_uptime":  df_w["uptime_pct"].median(),
                    "n_windows":      len(df_w),
                })
                all_rows.append(results[-1])

        df_r = pd.DataFrame(results).sort_values("score", ascending=False)
        best = df_r.iloc[0]
        best_configs[sym] = {"spacing": best["spacing"], "order_size_pct": best["order_size_pct"]}

        logger.info("  Done in %.1fs", time.time() - t0)

        # Print ranked table for this symbol
        W = 95
        print(f"\n{'='*W}")
        print(f"  {sym}  —  {len(starts)} windows × {len(SPACINGS_TO_TEST[sym])}×{len(ORDER_SIZES_TO_TEST)} configs  (scored: median_sharpe × win_rate)")
        print(f"{'='*W}")
        print(f"  {'Spacing':>9}  {'OrdSz':>6}  {'Score':>7}  {'MedPnL$':>9}  {'MedPnL%':>8}  "
              f"{'MedSharpe':>10}  {'WinRate':>8}  {'p5PnL':>8}  {'Uptime%':>8}")
        print(f"  {'-'*88}")
        for _, row in df_r.iterrows():
            marker = " ◀ best" if row["spacing"] == best["spacing"] and row["order_size_pct"] == best["order_size_pct"] else ""
            print(f"  {row['spacing']:>9.3f}  {row['order_size_pct']:>6.2f}  "
                  f"{row['score']:>7.3f}  "
                  f"{row['median_pnl']:>+9.2f}  "
                  f"{row['median_pnl_pct']:>+7.2f}%  "
                  f"{row['median_sharpe']:>10.2f}  "
                  f"{row['win_rate']:>7.1%}  "
                  f"{row['p5_pnl']:>+8.2f}  "
                  f"{row['median_uptime']:>7.1f}%"
                  f"{marker}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  Optimal configs:")
    for sym, cfg in best_configs.items():
        print(f"    {sym:<5}  spacing={cfg['spacing']:.3f}  order_size_pct={cfg['order_size_pct']:.2f}")
    print()

    # ---- Save CSV ----
    out = Path("models/tune_grid.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    logger.info("Full sweep → %s", out)

    # ---- Write back to run.py ----
    if not args.no_write and best_configs:
        _patch_run_py(best_configs)


def _patch_run_py(best_configs: dict[str, dict]) -> None:
    """Update SPACINGS and order_size_pct in scripts/run.py with tuned values."""
    run_path = Path("scripts/run.py")
    if not run_path.exists():
        logger.warning("scripts/run.py not found — skipping patch")
        return

    text = run_path.read_text()
    original = text

    for sym, cfg in best_configs.items():
        spacing  = cfg["spacing"]
        os_pct   = cfg["order_size_pct"]

        # Patch GridConfig for this symbol: spacing_pct=X.XXX
        text = re.sub(
            rf'(GridConfig\(symbol="{sym}"[^)]*spacing_pct=)[0-9.]+',
            lambda m: m.group(0).rsplit("=", 1)[0] + f"={spacing:.3f}",
            text,
        )
        # Patch order_size_pct for this symbol's GridConfig
        text = re.sub(
            rf'(GridConfig\(symbol="{sym}"[^)]*order_size_pct=)[0-9.]+',
            lambda m: m.group(0).rsplit("=", 1)[0] + f"={os_pct:.2f}",
            text,
        )

    if text == original:
        logger.warning("No GridConfig entries patched — check run.py format")
        return

    run_path.write_text(text)
    logger.info("Patched scripts/run.py with tuned grid configs")


if __name__ == "__main__":
    main()
