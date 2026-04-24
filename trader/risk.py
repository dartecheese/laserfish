"""Position sizing and risk management."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exchanges.base import Balance, Position


@dataclass
class RiskConfig:
    max_position_pct: float = 0.25   # max notional per position as fraction of equity
    max_total_exposure: float = 0.75 # max sum of all position notionals
    max_positions: int = 5
    min_alpha: float = 0.15          # signal must exceed this to open
    max_drawdown_pct: float = 20.0   # halt trading if drawdown exceeds this


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self._peak_equity: float | None = None

    def size_position(
        self,
        alpha: float,
        equity: float,
        asset_price: float,
        leverage: int = 1,
    ) -> float:
        """Return quantity (in base units) to trade.

        Uses fractional Kelly scaled by |alpha|:
            notional = equity × max_position_pct × |alpha|
            quantity  = notional × leverage / price
        """
        if abs(alpha) < self.cfg.min_alpha:
            return 0.0
        notional = equity * self.cfg.max_position_pct * abs(alpha)
        return notional * leverage / asset_price

    def check_drawdown(self, current_equity: float) -> bool:
        """Returns True if drawdown limit breached (halt trading)."""
        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity
        dd_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
        return dd_pct >= self.cfg.max_drawdown_pct

    def can_open_position(
        self,
        symbol: str,
        open_positions: list[Position],
        balance: Balance,
        equity: float,
    ) -> bool:
        """Check position count and gross exposure limits."""
        open_syms = {p.symbol for p in open_positions}
        if symbol in open_syms:
            return False   # already in this position
        if len(open_positions) >= self.cfg.max_positions:
            return False
        total_notional = sum(
            p.quantity * p.entry_price for p in open_positions
        )
        if total_notional / max(equity, 1) >= self.cfg.max_total_exposure:
            return False
        return True
