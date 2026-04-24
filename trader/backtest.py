"""Walk-forward backtesting engine.

Simulates live trading using the ONNX model's signals with triple-barrier exits.
Accounts for maker/taker fees and slippage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bars import Bar, cusum_filter
from .features import live_sequence, SEQ_LEN
from .model import infer_alpha, load_onnx


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_bps: float = 3.5        # Hyperliquid taker fee (0.035%)
    slippage_bps: float = 5.0   # conservative estimate
    max_position_pct: float = 0.25
    alpha_threshold: float = 0.15  # min |alpha| to open a position
    pt: float = 0.025
    sl: float = 0.025
    t_max: int = 24


@dataclass
class Trade:
    bar_in: int
    bar_out: int
    side: str           # "long" or "short"
    entry: float
    exit_price: float
    pnl_pct: float
    label: int          # +1=TP, -1=SL, 0=time


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        return (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100

    @property
    def sharpe(self) -> float:
        curve = np.array(self.equity_curve)
        if len(curve) < 2:
            return 0.0
        daily_rets = np.diff(curve) / curve[:-1]
        if daily_rets.std() < 1e-9:
            return 0.0
        return float(daily_rets.mean() / daily_rets.std() * np.sqrt(365))

    @property
    def max_drawdown_pct(self) -> float:
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / np.maximum(peak, 1e-9)
        return float(dd.min() * 100)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades)

    def summary(self) -> str:
        return (
            f"Trades: {len(self.trades)} | "
            f"Return: {self.total_return_pct:.1f}% | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"MaxDD: {self.max_drawdown_pct:.1f}% | "
            f"WinRate: {self.win_rate*100:.1f}%"
        )


def run_backtest(
    bars: list[Bar],
    onnx_path: str,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    if cfg is None:
        cfg = BacktestConfig()

    session = load_onnx(onnx_path)
    fee_frac = (cfg.fee_bps + cfg.slippage_bps) / 10_000

    capital = cfg.initial_capital
    result = BacktestResult(equity_curve=[capital])

    position_bar: int | None = None
    position_side: str | None = None
    entry_price: float = 0.0

    for i in range(SEQ_LEN, len(bars)):
        bar = bars[i]
        close = bar.close

        # Check exit conditions if in a position
        if position_bar is not None:
            held = i - position_bar
            price_ret = (close - entry_price) / entry_price
            if position_side == "short":
                price_ret = -price_ret

            tp_hit = price_ret >= cfg.pt
            sl_hit = price_ret <= -cfg.sl
            time_hit = held >= cfg.t_max

            if tp_hit or sl_hit or time_hit:
                label = 1 if tp_hit else (-1 if sl_hit else 0)
                net_pnl_pct = price_ret - fee_frac * 2   # round-trip fee
                capital *= (1 + net_pnl_pct * cfg.max_position_pct)
                result.trades.append(Trade(
                    bar_in=position_bar, bar_out=i, side=position_side,
                    entry=entry_price, exit_price=close,
                    pnl_pct=net_pnl_pct, label=label,
                ))
                position_bar = None
                position_side = None

        # Generate signal from model
        if position_bar is None:
            seq = live_sequence(bars[:i + 1])
            alpha = infer_alpha(session, seq)

            if abs(alpha) >= cfg.alpha_threshold:
                side = "long" if alpha > 0 else "short"
                entry_price = close * (1 + fee_frac * (1 if side == "long" else -1))
                position_bar = i
                position_side = side

        result.equity_curve.append(capital)

    return result
