"""Monte Carlo backtest — random 30-day window sampling across BTC, ETH, SOL.

Regime labels are computed ONCE per symbol on the full dataset, then N random
30-day windows are sliced and simulated. This keeps runtime under ~2 minutes.

Output:
  - Percentile table printed to stdout (5/25/50/75/95th)
  - models/monte_carlo.csv — full per-window results

Usage:
    python scripts/monte_carlo.py
    python scripts/monte_carlo.py --symbols BTC ETH SOL --n 500 --window-days 30
    python scripts/monte_carlo.py --seed 99 --n 200
"""
from __future__ import annotations

import argparse
import json
import logging
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
# Shared constants (mirror backtest_grid.py)                          #
# ------------------------------------------------------------------ #

FEE            = 0.00015
N_LEVELS       = 10
ORDER_SIZE_PCT = 0.04
MAX_LEVERAGE   = 2.0
INITIAL_EQUITY = 10_000.0
DRIFT_PCT      = 0.04
CONFIRM_N      = 3          # live config
SPACINGS       = {"BTC": 0.005, "ETH": 0.006, "SOL": 0.008}
MIN_GAP_DAYS   = 15         # minimum days between window start points


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


# ------------------------------------------------------------------ #
# Regime labels — computed once on the full series                    #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Grid simulator (minimal, mirrors GridStrategy paper logic)          #
# ------------------------------------------------------------------ #

class PaperGrid:
    def __init__(self, spacing: float):
        self.spacing = spacing
        self.active  = False
        self.center  = 0.0
        self.levels: dict[int, dict] = {}
        self.net_qty = 0.0
        self.realized_pnl = 0.0
        self.n_round_trips = 0
        self._pending_buys:  dict[int, float] = {}
        self._pending_sells: dict[int, float] = {}
        self.open_events  = 0
        self.close_events = 0

    def open(self, price: float, equity: float) -> None:
        if self.active:
            return
        self.center = price
        self.active = True
        self.levels.clear()
        self._pending_buys.clear()
        self._pending_sells.clear()
        qty = equity * ORDER_SIZE_PCT * MAX_LEVERAGE / price
        for i in range(1, N_LEVELS + 1):
            self.levels[-i] = {"side": "buy",  "price": price * (1 - self.spacing * i), "qty": qty, "filled": False}
            self.levels[+i] = {"side": "sell", "price": price * (1 + self.spacing * i), "qty": qty, "filled": False}
        self.open_events += 1

    def close(self, price: float) -> None:
        if not self.active:
            return
        self.active = False
        self.levels.clear()
        self.close_events += 1

    def step(self, price: float, equity: float) -> None:
        if not self.active:
            return
        if abs(price - self.center) / self.center > DRIFT_PCT:
            self.close(price)
            self.open(price, equity)
            return
        for idx in list(self.levels.keys()):
            lvl = self.levels[idx]
            if lvl["filled"]:
                continue
            filled = (lvl["side"] == "buy"  and price <= lvl["price"]) or \
                     (lvl["side"] == "sell" and price >= lvl["price"])
            if not filled:
                continue
            lvl["filled"] = True
            fill_px = lvl["price"]
            qty     = lvl["qty"]
            if lvl["side"] == "buy":
                self.net_qty += qty
                new_idx   = idx + 1
                new_price = self.center * (1 + self.spacing * new_idx)
                if new_idx in self._pending_sells:
                    sell_px = self._pending_sells.pop(new_idx)
                    self.realized_pnl += (sell_px - fill_px) * qty - FEE * 2 * qty * fill_px
                    self.n_round_trips += 1
                else:
                    self._pending_buys[new_idx] = fill_px
                self.levels[new_idx] = {"side": "sell", "price": new_price, "qty": qty, "filled": False}
            else:
                self.net_qty -= qty
                new_idx   = idx - 1
                new_price = self.center * (1 - self.spacing * abs(new_idx))
                if idx in self._pending_buys:
                    buy_px = self._pending_buys.pop(idx)
                    self.realized_pnl += (fill_px - buy_px) * qty - FEE * 2 * qty * buy_px
                    self.n_round_trips += 1
                else:
                    self._pending_sells[idx] = fill_px
                self.levels[new_idx] = {"side": "buy", "price": new_price, "qty": qty, "filled": False}


# ------------------------------------------------------------------ #
# Single window simulation                                            #
# ------------------------------------------------------------------ #

def run_window(closes: np.ndarray, labels: np.ndarray, spacing: float) -> dict:
    grid   = PaperGrid(spacing=spacing)
    equity = INITIAL_EQUITY

    regime_history: list[int] = []
    stable_regime  = 0
    active_bars    = 0
    equity_curve   = [INITIAL_EQUITY]

    for price, regime in zip(closes, labels):
        regime_history.append(int(regime))
        if len(regime_history) > CONFIRM_N:
            regime_history.pop(0)
        if len(regime_history) == CONFIRM_N and len(set(regime_history)) == 1:
            stable_regime = regime_history[-1]

        if stable_regime == 1:
            if not grid.active:
                grid.open(price, equity)
            else:
                grid.step(price, equity)
            active_bars += 1
        else:
            if grid.active:
                grid.close(price)

        equity = INITIAL_EQUITY + grid.realized_pnl
        equity_curve.append(equity)

    if grid.active:
        grid.close(closes[-1])

    eq = np.array(equity_curve)
    returns = np.diff(eq) / (eq[:-1] + 1e-9)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / (peak + 1e-9)).max())
    hours  = len(closes) * 15 / 60
    pnl    = grid.realized_pnl
    sharpe = (float(np.mean(returns)) / (float(np.std(returns)) + 1e-9) * np.sqrt(4 * 365)
              if len(returns) > 1 else 0.0)

    return {
        "realized_pnl":  pnl,
        "pnl_pct":       pnl / INITIAL_EQUITY * 100,
        "round_trips":   grid.n_round_trips,
        "open_events":   grid.open_events,
        "uptime_pct":    active_bars / max(len(closes), 1) * 100,
        "pnl_per_hour":  pnl / max(hours, 1),
        "max_dd_pct":    max_dd * 100,
        "sharpe":        sharpe,
    }


# ------------------------------------------------------------------ #
# Random window sampler                                               #
# ------------------------------------------------------------------ #

def sample_windows(n_bars_total: int, window_bars: int, n_samples: int,
                   min_gap_bars: int, rng: np.random.Generator) -> list[int]:
    """Return up to n_samples random start indices with minimum gap between them."""
    max_start = n_bars_total - window_bars
    if max_start <= 0:
        return []

    starts: list[int] = []
    attempts = 0
    while len(starts) < n_samples and attempts < n_samples * 20:
        s = int(rng.integers(0, max_start))
        if all(abs(s - existing) >= min_gap_bars for existing in starts):
            starts.append(s)
        attempts += 1

    return sorted(starts)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",      nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--n",            type=int,  default=300,  help="Windows to sample per symbol")
    parser.add_argument("--window-days",  type=int,  default=30,   help="Window size in days")
    parser.add_argument("--min-gap-days", type=int,  default=MIN_GAP_DAYS)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--data-dir",     default="data")
    args = parser.parse_args()

    rng          = np.random.default_rng(args.seed)
    window_bars  = args.window_days * 96   # 15m bars per day
    min_gap_bars = args.min_gap_days * 96
    all_results  = []

    for sym in args.symbols:
        t0 = time.time()
        df = load_symbol(sym, args.data_dir)
        if df is None:
            continue

        closes  = df["close"].values
        funding = df["funding_rate"].values
        spacing = SPACINGS.get(sym, 0.005)

        logger.info("Fitting regime detector for %s (%d bars)…", sym, len(closes))
        labels = compute_regime_labels(closes, funding)
        range_pct = (labels == 1).mean() * 100
        logger.info("  RANGE: %.1f%% of bars  (%.1fs)", range_pct, time.time() - t0)

        starts = sample_windows(len(closes), window_bars, args.n, min_gap_bars, rng)
        logger.info("  Sampled %d windows (requested %d)", len(starts), args.n)

        for i, s in enumerate(starts):
            w_closes = closes[s: s + window_bars]
            w_labels = labels[s: s + window_bars]
            start_date = df.index[s].strftime("%Y-%m-%d")

            result = run_window(w_closes, w_labels, spacing)
            result["symbol"]     = sym
            result["start"]      = start_date
            result["window_idx"] = i
            all_results.append(result)

        logger.info("  Done — %d windows in %.1fs", len(starts), time.time() - t0)

    if not all_results:
        logger.error("No results — check data files in %s", args.data_dir)
        return

    df_res = pd.DataFrame(all_results)

    # ---- Percentile table ----
    W = 90
    PCTS = [5, 25, 50, 75, 95]
    METRICS = [
        ("realized_pnl",  "PnL ($)",    "{:>+8.2f}"),
        ("pnl_pct",       "PnL (%)",    "{:>+7.2f}%"),
        ("sharpe",        "Sharpe",     "{:>7.2f}"),
        ("max_dd_pct",    "MaxDD (%)",  "{:>7.2f}%"),
        ("round_trips",   "Trips",      "{:>6.1f}"),
        ("uptime_pct",    "Uptime%",    "{:>7.1f}%"),
    ]

    print("\n" + "=" * W)
    print(f"  Monte Carlo — {args.window_days}-day windows  |  N={len(all_results)}  |  seed={args.seed}")
    print("=" * W)

    for sym in args.symbols:
        sub = df_res[df_res["symbol"] == sym]
        if sub.empty:
            continue
        n = len(sub)
        win_rate = (sub["realized_pnl"] > 0).mean() * 100
        print(f"\n  {sym}  ({n} windows,  win rate: {win_rate:.1f}%)")
        print(f"  {'Metric':<14}", end="")
        for p in PCTS:
            print(f"  {'p'+str(p):>9}", end="")
        print(f"  {'mean':>9}")
        print(f"  {'-'*75}")
        for col, label, fmt in METRICS:
            print(f"  {label:<14}", end="")
            for p in PCTS:
                val = sub[col].quantile(p / 100)
                print(f"  {fmt.format(val):>9}", end="")
            print(f"  {fmt.format(sub[col].mean()):>9}")

    # ---- Combined across all symbols ----
    print(f"\n  {'─'*75}")
    print(f"  ALL SYMBOLS combined  ({len(df_res)} windows)")
    print(f"  {'Metric':<14}", end="")
    for p in PCTS:
        print(f"  {'p'+str(p):>9}", end="")
    print(f"  {'mean':>9}")
    print(f"  {'-'*75}")
    for col, label, fmt in METRICS:
        print(f"  {label:<14}", end="")
        for p in PCTS:
            val = df_res[col].quantile(p / 100)
            print(f"  {fmt.format(val):>9}", end="")
        print(f"  {fmt.format(df_res[col].mean()):>9}")

    win_rate_all = (df_res["realized_pnl"] > 0).mean() * 100
    print(f"\n  Overall win rate: {win_rate_all:.1f}%")
    print()

    out = Path("models/monte_carlo.csv")
    out.parent.mkdir(exist_ok=True)
    df_res.to_csv(out, index=False)
    logger.info("Full results → %s", out)


if __name__ == "__main__":
    main()
