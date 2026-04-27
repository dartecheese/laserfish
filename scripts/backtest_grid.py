"""Grid + regime stability backtest across multiple assets and time windows.

Tests two configurations side by side on each window:
  A) No stability filter (regime_confirm_n=1) — current naive behavior
  B) Stability filter    (regime_confirm_n=3) — proposed fix

Metrics per run:
  - Grid uptime %      (fraction of bars where grid was active)
  - Round trips        (completed buy-sell pairs)
  - Realized PnL       ($)
  - Net PnL / hour     (quality-adjusted throughput)
  - Open/close events  (how often the grid was toggled — lower = better stability)

Usage:
    python scripts/backtest_grid.py
    python scripts/backtest_grid.py --symbols BTC ETH SOL --confirm 1 3 5
    python scripts/backtest_grid.py --windows 30 60 90
"""
from __future__ import annotations

import argparse
import json
import logging
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

FEE = 0.00015     # 1.5bps maker — grid uses limit orders
GRID_SPACING = 0.005   # 0.5% per level
N_LEVELS     = 10
ORDER_SIZE_PCT = 0.04  # 4% equity per level
MAX_LEVERAGE   = 2.0
INITIAL_EQUITY = 10_000.0
DRIFT_PCT      = 0.04   # recenter threshold


# ------------------------------------------------------------------ #
# Regime detection (using src.regime.detector on historical data)     #
# ------------------------------------------------------------------ #

def compute_regime_labels(closes: np.ndarray, funding: np.ndarray) -> np.ndarray:
    """Return per-bar regime labels (0=BEAR,1=RANGING,2=BULL,3=VOLATILE)."""
    from src.regime.detector import RegimeDetector
    returns = np.diff(np.log(closes + 1e-9))
    fund    = funding[1:]
    det = RegimeDetector()
    if len(returns) < 200:
        return np.ones(len(closes), dtype=int)
    det.fit(returns, fund)
    labels, _ = det.predict(returns, fund)
    # Pad first bar (no return)
    return np.concatenate([[labels[0]], labels])


# ------------------------------------------------------------------ #
# Grid simulation                                                     #
# ------------------------------------------------------------------ #

class PaperGrid:
    """Minimal grid simulator — mirrors GridStrategy paper logic without the exchange layer."""

    def __init__(self, spacing=GRID_SPACING, n_levels=N_LEVELS,
                 order_size_pct=ORDER_SIZE_PCT, max_leverage=MAX_LEVERAGE):
        self.spacing = spacing
        self.n_levels = n_levels
        self.order_size_pct = order_size_pct
        self.max_leverage = max_leverage

        self.active = False
        self.center = 0.0
        self.levels: dict[int, dict] = {}     # idx → {side, price, qty, filled}
        self.net_qty = 0.0
        self.realized_pnl = 0.0
        self.n_round_trips = 0
        self._pending_buys: dict[int, float] = {}
        self._pending_sells: dict[int, float] = {}
        self.open_events = 0
        self.close_events = 0

    def open(self, price: float, equity: float) -> None:
        if self.active:
            return
        self.center = price
        self.active = True
        self.levels.clear()
        self.net_qty = 0.0
        qty = equity * self.order_size_pct * self.max_leverage / price
        for i in range(1, self.n_levels + 1):
            self.levels[-i] = {"side": "buy",  "price": price * (1 - self.spacing * i), "qty": qty, "filled": False}
            self.levels[+i] = {"side": "sell", "price": price * (1 + self.spacing * i), "qty": qty, "filled": False}
        self.open_events += 1

    def close(self, price: float) -> None:
        if not self.active:
            return
        self.active = False
        self.levels.clear()
        self.net_qty = 0.0
        self.close_events += 1

    def step(self, price: float, equity: float) -> None:
        if not self.active:
            return

        # Recenter if price drifted too far
        if abs(price - self.center) / self.center > DRIFT_PCT:
            self.close(price)
            self.open(price, equity)
            return

        for idx in list(self.levels.keys()):
            lvl = self.levels[idx]
            if lvl["filled"]:
                continue

            # Simulate fill
            filled = (lvl["side"] == "buy"  and price <= lvl["price"]) or \
                     (lvl["side"] == "sell" and price >= lvl["price"])
            if not filled:
                continue

            lvl["filled"] = True
            fill_px = lvl["price"]
            qty = lvl["qty"]

            if lvl["side"] == "buy":
                self.net_qty += qty
                new_idx = idx + 1
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
                new_idx = idx - 1
                new_price = self.center * (1 - self.spacing * abs(new_idx))
                if idx in self._pending_buys:
                    buy_px = self._pending_buys.pop(idx)
                    self.realized_pnl += (fill_px - buy_px) * qty - FEE * 2 * qty * buy_px
                    self.n_round_trips += 1
                else:
                    self._pending_sells[idx] = fill_px
                self.levels[new_idx] = {"side": "buy", "price": new_price, "qty": qty, "filled": False}


# ------------------------------------------------------------------ #
# Full simulation for one asset / window / confirm-N                  #
# ------------------------------------------------------------------ #

CHOP_WINDOW     = 14
CHOP_THRESHOLD  = 50.0


def _choppiness(closes: np.ndarray) -> float:
    """Choppiness index: 100 * log10(sum_ATR / range) / log10(N). >50 = ranging."""
    if len(closes) < CHOP_WINDOW + 1:
        return 100.0  # default ranging when insufficient data
    w = closes[-(CHOP_WINDOW + 1):]
    atr_sum = np.sum(np.abs(np.diff(w)))
    price_range = w.max() - w.min() + 1e-9
    return 100.0 * np.log10(atr_sum / price_range) / np.log10(CHOP_WINDOW)


def run_simulation(
    closes: np.ndarray,
    regime_labels: np.ndarray,
    confirm_n: int,
    use_chop: bool = False,
    label: str = "",
    spacing: float = GRID_SPACING,
) -> dict:
    """Simulate grid trading with stability filter and optional choppiness gate."""
    grid = PaperGrid(spacing=spacing)
    equity = INITIAL_EQUITY

    regime_history: list[int] = []
    stable_regime = 0
    active_bars = 0

    for i, (price, regime) in enumerate(zip(closes, regime_labels)):
        # Stability filter
        regime_history.append(int(regime))
        if len(regime_history) > confirm_n:
            regime_history.pop(0)
        if len(regime_history) == confirm_n and len(set(regime_history)) == 1:
            stable_regime = regime_history[-1]

        # Choppiness gate (per-symbol)
        if use_chop:
            ci = _choppiness(closes[max(0, i - CHOP_WINDOW - 1): i + 1])
            chop_ok = ci > CHOP_THRESHOLD
        else:
            chop_ok = True

        if stable_regime == 1 and chop_ok:
            if not grid.active:
                grid.open(price, equity)
            else:
                grid.step(price, equity)
            active_bars += 1
        else:
            if grid.active:
                grid.close(price)

    if grid.active:
        grid.close(closes[-1])

    hours = len(closes) * 15 / 60
    uptime_pct = active_bars / max(len(closes), 1) * 100
    pnl_per_hour = grid.realized_pnl / max(hours, 1)

    return {
        "label":        label,
        "confirm_n":    confirm_n,
        "use_chop":     use_chop,
        "bars":         len(closes),
        "hours":        hours,
        "realized_pnl": grid.realized_pnl,
        "round_trips":  grid.n_round_trips,
        "open_events":  grid.open_events,
        "close_events": grid.close_events,
        "uptime_pct":   uptime_pct,
        "pnl_per_hour": pnl_per_hour,
        "pnl_pct":      grid.realized_pnl / INITIAL_EQUITY * 100,
    }


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
    # Resample 5m → 15m
    df = df.resample("15min").agg({"open": "first", "high": "max",
                                   "low": "min", "close": "last", "volume": "sum"}).dropna()
    df["funding_rate"] = 0.0
    return df


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",  nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--windows",  nargs="+", type=int, default=[30, 60, 90],
                        help="Lookback windows in days")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    # Per-symbol spacing matching run.py config
    SPACINGS = {"BTC": 0.005, "ETH": 0.006, "SOL": 0.008}

    # Three configs to compare
    CONFIGS = [
        {"confirm_n": 1, "use_chop": False, "tag": "naive (N=1)"},
        {"confirm_n": 3, "use_chop": False, "tag": "stability N=3  ◀ live"},
        {"confirm_n": 5, "use_chop": False, "tag": "stability N=5"},
    ]

    all_results = []

    for sym in args.symbols:
        df = load_symbol(sym, args.data_dir)
        if df is None:
            continue

        logger.info("Computing regime labels for %s (%d bars)…", sym, len(df))
        closes  = df["close"].values
        funding = df["funding_rate"].values
        labels  = compute_regime_labels(closes, funding)
        spacing = SPACINGS.get(sym, 0.005)

        range_pct = (labels == 1).mean() * 100
        logger.info("  Regime=RANGE: %.1f%% of bars", range_pct)

        for window_days in args.windows:
            n_bars = window_days * 96
            if n_bars > len(df):
                continue

            w_closes = closes[-n_bars:]
            w_labels = labels[-n_bars:]
            window_start = df.index[-n_bars].strftime("%Y-%m-%d")
            window_end   = df.index[-1].strftime("%Y-%m-%d")

            for cfg in CONFIGS:
                result = run_simulation(
                    w_closes, w_labels,
                    confirm_n=cfg["confirm_n"],
                    use_chop=cfg["use_chop"],
                    spacing=spacing,
                    label=f"{sym}_{window_days}d_{cfg['tag']}",
                )
                result["symbol"]      = sym
                result["window_days"] = window_days
                result["start"]       = window_start
                result["end"]         = window_end
                result["config"]      = cfg["tag"]
                all_results.append(result)

    if not all_results:
        logger.error("No results — check data files in %s", args.data_dir)
        return

    df_res = pd.DataFrame(all_results)

    # ---- Print comparison table ----
    W = 95
    print("\n" + "=" * W)
    print("  Grid Backtest — Naive vs Stability Filter vs Stability+Chop")
    print("=" * W)

    for sym in args.symbols:
        sub = df_res[df_res["symbol"] == sym]
        if sub.empty:
            continue
        print(f"\n  {'─'*60}")
        print(f"  {sym}")
        print(f"  {'─'*60}")
        print(f"  {'Window':<10} {'Config':<35} {'PnL$':>8} {'PnL%':>7} {'Trips':>6} {'Opens':>6} {'Uptime%':>8} {'$/hr':>8}")
        print(f"  {'-'*90}")
        for wd in sorted(sub["window_days"].unique()):
            wsub = sub[sub["window_days"] == wd]
            for _, row in wsub.iterrows():
                print(f"  {str(wd)+'d':<10} {row['config']:<35} "
                      f"{row['realized_pnl']:>+8.2f} "
                      f"{row['pnl_pct']:>+6.2f}% "
                      f"{int(row['round_trips']):>6} "
                      f"{int(row['open_events']):>6} "
                      f"{row['uptime_pct']:>7.1f}% "
                      f"{row['pnl_per_hour']:>+8.4f}")
            print()

    # ---- Summary across all assets ----
    print("=" * W)
    print("\n  Summary — mean across all symbols and windows:")
    for cfg in CONFIGS:
        sub = df_res[df_res["config"] == cfg["tag"]]
        print(f"  {cfg['tag']:<35}  "
              f"PnL=${sub['realized_pnl'].mean():>+7.2f}  "
              f"trips={sub['round_trips'].mean():>5.1f}  "
              f"opens={sub['open_events'].mean():>5.1f}  "
              f"uptime={sub['uptime_pct'].mean():>5.1f}%  "
              f"$/hr={sub['pnl_per_hour'].mean():>+7.4f}")

    print()
    out = Path("models/backtest_grid.csv")
    df_res.to_csv(out, index=False)
    logger.info("Results → %s", out)


if __name__ == "__main__":
    main()
