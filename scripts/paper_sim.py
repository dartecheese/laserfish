"""Historical paper-trading simulation for funding_arb and ESN-RL strategies.

Replays up to 90 periods of real 8-hour funding rate history + price candles
through the strategy signal logic, executing simulated trades on a paper
portfolio. No ML training needed — works offline from live exchange data.

Usage:
    python scripts/paper_sim.py                         # all symbols, both strategies
    python scripts/paper_sim.py --strategy funding_arb  # just funding arb
    python scripts/paper_sim.py --strategy esn_rl --symbol ETH
    python scripts/paper_sim.py --capital 50000 --leverage 5
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.exchanges.hyperliquid import HyperliquidExchange
from trader.strategies.funding_arb import FundingArbConfig
from trader.strategies.esn_rl import ESNRLAgent, ESNRLConfig, make_feature_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_sim")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical paper-trading sim")
    p.add_argument("--strategy", choices=["funding_arb", "esn_rl", "both"], default="both")
    p.add_argument("--symbol",   default=None, help="Single symbol for esn_rl (default: ETH)")
    p.add_argument("--capital",  type=float, default=10_000.0)
    p.add_argument("--leverage", type=int,   default=3)
    p.add_argument("--periods",  type=int,   default=90, help="8h funding periods to replay (~30 days)")
    p.add_argument("--alloc",    type=float, default=0.12,
                   help="Fraction of initial capital per trade (default 12%%)")
    p.add_argument("--fee-bps",  type=float, default=3.5, help="Round-trip fee basis points")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Simple trade ledger (no PaperPortfolio — avoids cash-compounding bugs)
# ─────────────────────────────────────────────────────────────────────────────

class _Ledger:
    """Fixed-allocation trade ledger. Position size always = alloc * capital."""

    def __init__(self, capital: float, alloc: float, leverage: int, fee_rate: float):
        self.capital   = capital
        self.alloc     = alloc
        self.leverage  = leverage
        self.fee_rate  = fee_rate
        self.cash      = capital
        self.realized  = 0.0
        self.funding   = 0.0
        self._positions: dict[str, dict] = {}   # sym → {side, qty, entry}
        self.equity_curve: list[float]   = [capital]
        self.trades: list[dict]          = []

    def _notional(self, price: float) -> float:
        return self.capital * self.alloc * self.leverage

    def open(self, sym: str, side: str, price: float) -> bool:
        if sym in self._positions or price <= 0:
            return False
        notional = self._notional(price)
        margin   = notional / self.leverage
        if margin > self.cash:
            return False
        qty = notional / price
        fee = notional * self.fee_rate
        self.cash -= margin + fee
        self._positions[sym] = {"side": side, "qty": qty, "entry": price, "fund_pnl": 0.0}
        return True

    def close(self, sym: str, price: float) -> float:
        pos = self._positions.pop(sym, None)
        if pos is None:
            return 0.0
        sign = 1 if pos["side"] == "long" else -1
        pnl  = sign * pos["qty"] * (price - pos["entry"])
        fee  = pos["qty"] * price * self.fee_rate
        margin = (pos["qty"] * pos["entry"]) / self.leverage
        total_pnl = pnl + pos["fund_pnl"] - fee
        self.cash     += margin + total_pnl
        self.realized += total_pnl
        self.trades.append({
            "sym": sym, "side": pos["side"],
            "entry": pos["entry"], "exit": price,
            "pnl_net": total_pnl, "qty": pos["qty"],
        })
        return total_pnl

    def accrue_funding(self, sym: str, rate_8h: float, price: float) -> None:
        pos = self._positions.get(sym)
        if pos is None:
            return
        direction = 1 if pos["side"] == "long" else -1
        notional  = pos["qty"] * price          # USD value of position
        payment   = -direction * rate_8h * notional  # longs pay when funding > 0
        pos["fund_pnl"] += payment
        self.funding    += payment

    def mark_equity(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for sym, pos in self._positions.items():
            p = prices.get(sym, pos["entry"])
            sign = 1 if pos["side"] == "long" else -1
            upnl += sign * pos["qty"] * (p - pos["entry"]) + pos["fund_pnl"]
        margin_held = sum(
            (pos["qty"] * pos["entry"]) / self.leverage
            for pos in self._positions.values()
        )
        eq = self.cash + margin_held + upnl
        self.equity_curve.append(eq)
        return eq


# ─────────────────────────────────────────────────────────────────────────────
# Funding-arb historical sim
# ─────────────────────────────────────────────────────────────────────────────

def run_funding_arb_sim(
    ex: HyperliquidExchange,
    capital: float,
    leverage: int,
    periods: int,
    alloc: float,
    fee_bps: float,
) -> dict:
    logger.info("=== Funding Arb Sim (%d 8h periods ≈ %d days) ===", periods, periods // 3)

    cfg = FundingArbConfig(max_leverage=leverage)
    symbols = cfg.symbols
    fee_rate = fee_bps / 10_000

    logger.info("Fetching funding rate history for %d symbols…", len(symbols))
    history:  dict[str, list[tuple[int, float]]] = {}
    prices_h: dict[str, list[float]]             = {}

    for sym in symbols:
        try:
            history[sym]  = ex.get_funding_rate_history(sym, limit=periods)
            candles       = ex.get_candles(sym, "8h", periods)
            prices_h[sym] = [c.close for c in candles]
        except Exception as exc:
            logger.warning("  %s: skipped (%s)", sym, exc)
            history[sym]  = []
            prices_h[sym] = []

    n_periods = min(periods, min((len(v) for v in history.values() if v), default=0))
    if n_periods == 0:
        return {}
    logger.info("Replaying %d periods…", n_periods)

    ledger  = _Ledger(capital, alloc, leverage, fee_rate)
    rolling = {s: deque(maxlen=cfg.history_window) for s in symbols}
    consec  = {s: 0 for s in symbols}

    def _z(sym: str, rate: float) -> float | None:
        h = list(rolling[sym])
        if len(h) < 10:
            return None
        mu, sd = float(np.mean(h)), float(np.std(h))
        return (rate - mu) / sd if sd > 1e-9 else 0.0

    for i in range(n_periods):
        period_prices: dict[str, float] = {}

        for sym in symbols:
            hist = history.get(sym, [])
            if i >= len(hist):
                continue
            _, rate  = hist[i]
            ph       = prices_h.get(sym, [])
            price    = ph[i] if i < len(ph) else 0.0
            if price <= 0:
                continue
            period_prices[sym] = price

            rolling[sym].append(rate)
            ledger.accrue_funding(sym, rate, price)

            # Exit check
            if sym in ledger._positions:
                z = _z(sym, rate)
                if z is not None and abs(z) <= cfg.z_exit:
                    ledger.close(sym, price)
                    logger.debug("  CLOSE %s @ %.4f", sym, price)

        # Entry signals (skip if position already open)
        for sym in symbols:
            if sym in ledger._positions:
                continue
            hist = history.get(sym, [])
            if i >= len(hist):
                continue
            _, rate = hist[i]
            price   = period_prices.get(sym, 0.0)
            if price <= 0:
                continue

            ann          = rate * 1095
            per_min      = cfg.per_asset_min_funding.get(sym, cfg.min_funding_annualized)
            if abs(ann) < per_min:
                consec[sym] = 0
                continue

            consec[sym] += 1
            if consec[sym] < cfg.require_consecutive:
                continue

            z = _z(sym, rate)
            if z is None or abs(z) < cfg.z_entry:
                continue

            side = "short" if z > 0 else "long"
            if ledger.open(sym, side, price):
                logger.debug("  OPEN  %s %s @ %.4f  z=%.2f", sym, side.upper(), price, z)

        ledger.mark_equity(period_prices)

    # Force-close all at last known price
    for sym in list(ledger._positions.keys()):
        price = period_prices.get(sym, ledger._positions[sym]["entry"])
        ledger.close(sym, price)

    final_equity = ledger.equity_curve[-1]
    return {
        "strategy": "funding_arb",
        "capital": capital, "final_equity": final_equity,
        "total_return_pct": (final_equity - capital) / capital * 100,
        "n_trades": len(ledger.trades), "n_periods": n_periods,
        "equity_curve": ledger.equity_curve, "trades": ledger.trades,
        "funding_total": ledger.funding, "realized_pnl": ledger.realized,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ESN-RL historical sim
# ─────────────────────────────────────────────────────────────────────────────

def run_esn_rl_sim(
    ex: HyperliquidExchange,
    symbol: str,
    capital: float,
    leverage: int,
    periods: int,
    alloc: float,
    fee_bps: float,
) -> dict:
    logger.info("=== ESN-RL Sim: %s (%d 8h periods) ===", symbol, periods)

    fee_rate     = fee_bps / 10_000
    candle_count = min(periods * 32, 1000)   # ccxt limit

    logger.info("Fetching %d 15m candles for %s…", candle_count, symbol)
    candles     = ex.get_candles(symbol, "15m", candle_count)
    fund_hist   = ex.get_funding_rate_history(symbol, limit=periods)
    logger.info("  Got %d candles, %d funding records.", len(candles), len(fund_hist))

    if len(candles) < 50:
        return {"strategy": "esn_rl", "symbol": symbol, "error": "insufficient data"}

    agent = ESNRLAgent(ESNRLConfig(
        n_reservoir=200, spectral_radius=0.95, sparsity=0.05,
        leaking_rate=0.30, learning_rate=5e-5, risk_aversion=1.0,
        transaction_cost=fee_rate, update_window=50,
        warmup_steps=100, max_leverage=leverage, checkpoint_path=None,
    ))

    ledger          = _Ledger(capital, alloc, leverage, fee_rate)
    vol_window      = deque(maxlen=20)
    current_qty     = 0.0   # signed base units
    prev_close      = candles[0].close

    def _fund_at(idx: int) -> float:
        fi = idx // 32
        return fund_hist[fi][1] if fi < len(fund_hist) else 0.0

    for idx, c in enumerate(candles[1:], 1):
        log_ret   = math.log(c.close / prev_close) if prev_close > 0 else 0.0
        vol_window.append(c.volume)
        vol_ratio = c.volume / (float(np.mean(vol_window)) + 1e-9)
        fund_8h   = _fund_at(idx)

        features = make_feature_vector(
            log_return=log_ret, funding_rate_8h=fund_8h,
            basis_pct=0.0, volume_ratio=vol_ratio, timestamp_ms=c.timestamp,
        )
        result = agent.step(features=features, log_return=log_ret, funding_rate_8h=fund_8h)
        prev_close = c.close

        diag = agent.diagnostics()
        if not diag["warmed_up"]:
            ledger.equity_curve.append(capital)
            continue

        eq   = ledger.equity_curve[-1]
        side, delta_qty = agent.position_to_order(
            current_qty=current_qty,
            equity_usd=eq,
            asset_price=c.close,
        )

        if side is not None and delta_qty > 0:
            # Close existing opposite position first
            if symbol in ledger._positions:
                existing = ledger._positions[symbol]["side"]
                target   = "long" if side == "buy" else "short"
                if existing != target:
                    ledger.close(symbol, c.close)
                    current_qty = 0.0

            if symbol not in ledger._positions:
                target_side = "long" if side == "buy" else "short"
                if ledger.open(symbol, target_side, c.close):
                    current_qty = delta_qty if side == "buy" else -delta_qty

        # Accrue funding every 32 candles (= 8h)
        if idx % 32 == 0:
            ledger.accrue_funding(symbol, fund_8h, c.close)

        ledger.mark_equity({symbol: c.close})

    # Close final position
    if symbol in ledger._positions:
        ledger.close(symbol, candles[-1].close)

    final_equity = ledger.equity_curve[-1]
    return {
        "strategy": "esn_rl", "symbol": symbol,
        "capital": capital, "final_equity": final_equity,
        "total_return_pct": (final_equity - capital) / capital * 100,
        "n_trades": len(ledger.trades), "n_candles": len(candles),
        "equity_curve": ledger.equity_curve, "trades": ledger.trades,
        "funding_total": ledger.funding,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    rets = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
    return float(np.mean(rets) / (np.std(rets) + 1e-9) * math.sqrt(len(rets)))


def _max_dd(equity_curve: list[float]) -> float:
    eq   = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / (peak + 1e-9)
    return float(dd.min() * 100)


def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.get("pnl_net", 0) > 0) / len(trades) * 100


def print_report(r: dict) -> None:
    if not r:
        print("  (no data)")
        return
    print()
    strat = r.get("strategy", "?")
    if strat == "funding_arb":
        header = "FUNDING ARB — BACKTEST RESULTS"
    else:
        header = f"ESN-RL ({r.get('symbol','?')}) — BACKTEST RESULTS"
    pad = (57 - len(header)) // 2
    print("┌" + "─" * 57 + "┐")
    print(f"│{' ' * pad}{header}{' ' * (57 - pad - len(header))}│")
    print("└" + "─" * 57 + "┘")

    if "error" in r:
        print(f"  Error: {r['error']}")
        return

    if strat == "funding_arb":
        print(f"  Periods replayed : {r['n_periods']} × 8h  (~{r['n_periods']//3} days)")
    else:
        print(f"  Candles replayed : {r['n_candles']} × 15m")

    print(f"  Trades           : {r['n_trades']}")
    print(f"  Starting capital : ${r['capital']:>10,.2f}")
    print(f"  Final equity     : ${r['final_equity']:>10,.2f}")
    ret = r['total_return_pct']
    print(f"  Total return     : {ret:>+.2f}%")
    print(f"  Funding PnL      : ${r.get('funding_total', 0):>+,.2f}")
    if r['n_trades']:
        print(f"  Win rate         : {_win_rate(r['trades']):.1f}%")
    print(f"  Sharpe (in-sim)  : {_sharpe(r['equity_curve']):.2f}")
    print(f"  Max drawdown     : {_max_dd(r['equity_curve']):.2f}%")

    trades = r.get("trades", [])
    if trades:
        print()
        print("  Top 5 trades by |PnL|:")
        for t in sorted(trades, key=lambda x: abs(x.get("pnl_net", 0)), reverse=True)[:5]:
            sym  = t.get("sym", t.get("symbol", "?"))
            side = t.get("side", "?").upper()
            print(f"    {sym:8s} {side:5s}  "
                  f"entry={t['entry']:.4f}  exit={t['exit']:.4f}  "
                  f"pnl=${t['pnl_net']:>+8.2f}")
    print()


def main() -> None:
    args = build_args()
    ex   = HyperliquidExchange(paper=True)

    if args.strategy in ("funding_arb", "both"):
        r = run_funding_arb_sim(
            ex=ex, capital=args.capital, leverage=args.leverage,
            periods=args.periods, alloc=args.alloc, fee_bps=args.fee_bps,
        )
        print_report(r)

    if args.strategy in ("esn_rl", "both"):
        sym = args.symbol or "ETH"
        r = run_esn_rl_sim(
            ex=ex, symbol=sym, capital=args.capital, leverage=args.leverage,
            periods=args.periods, alloc=args.alloc, fee_bps=args.fee_bps,
        )
        print_report(r)


if __name__ == "__main__":
    main()
