"""Dynamic leverage engine — Kelly vol-targeting + regime multiplier + circuit breaker.

Formula:
    raw_leverage = target_annual_vol / realized_vol_24h
    regime_leverage = raw_leverage × regime_multiplier
    final_leverage = clip(regime_leverage, min_lev, max_lev)

Circuit breakers:
    - Daily drawdown > dd_hard_pct  → force to min_lev
    - Daily drawdown > dd_soft_pct  → halve current leverage
    - Funding rate in extreme territory → funding penalty applied

The engine is stateless per-call; pass in the current vol/regime/drawdown.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeverageConfig:
    # Annualized vol target — we size so portfolio vol matches this
    target_vol: float = 0.50      # 50% annualized (conservative for crypto)

    # Absolute leverage bounds regardless of vol signal
    min_leverage: float = 1.0
    max_leverage: float = 5.0

    # Soft circuit breaker: halve leverage if daily drawdown exceeds this
    dd_soft_pct: float = 0.04     # 4%

    # Hard circuit breaker: floor to min_leverage if daily drawdown exceeds this
    dd_hard_pct: float = 0.08     # 8%

    # Funding penalty: if |funding_z| > this, scale down leverage
    funding_z_penalty: float = 2.0

    # Minimum realized vol to avoid division by near-zero
    min_realized_vol: float = 0.10   # 10% annualized floor


class DynamicLeverage:
    """
    Computes position leverage dynamically at each entry signal.

    Usage:
        engine = DynamicLeverage(LeverageConfig())
        lev = engine.compute(realized_vol, regime_mult, drawdown_pct, funding_z)
    """

    def __init__(self, cfg: LeverageConfig | None = None):
        self.cfg = cfg or LeverageConfig()
        self._equity_high: float = 0.0   # rolling high-water mark for drawdown

    def update_hwm(self, equity: float) -> None:
        """Update high-water mark. Call at each scan with current equity."""
        if equity > self._equity_high:
            self._equity_high = equity

    def daily_drawdown(self, equity: float) -> float:
        """Current drawdown fraction from high-water mark."""
        if self._equity_high <= 0:
            return 0.0
        return max(0.0, (self._equity_high - equity) / self._equity_high)

    def compute(
        self,
        realized_vol_24h: float,       # annualized realized vol (e.g. 0.60 = 60%)
        regime_multiplier: float = 1.0,
        drawdown_pct: float = 0.0,     # current drawdown from HWM
        funding_z: float = 0.0,        # |funding z-score| for penalty
    ) -> float:
        """
        Return the recommended leverage for the next position.
        """
        cfg = self.cfg

        # Vol targeting: size so portfolio vol = target_vol
        eff_vol = max(realized_vol_24h, cfg.min_realized_vol)
        vol_leverage = cfg.target_vol / eff_vol

        # Regime multiplier from HMM state
        lev = vol_leverage * regime_multiplier

        # Funding penalty — scale down if funding is in extreme territory
        if abs(funding_z) > cfg.funding_z_penalty:
            excess = abs(funding_z) - cfg.funding_z_penalty
            penalty = 1.0 / (1.0 + 0.3 * excess)   # soft decay
            lev *= penalty

        # Circuit breakers
        if drawdown_pct >= cfg.dd_hard_pct:
            logger.warning(
                "Hard circuit breaker: drawdown=%.1f%% — forcing min leverage %.1fx",
                drawdown_pct * 100, cfg.min_leverage,
            )
            lev = cfg.min_leverage
        elif drawdown_pct >= cfg.dd_soft_pct:
            logger.info(
                "Soft circuit breaker: drawdown=%.1f%% — halving leverage",
                drawdown_pct * 100,
            )
            lev = lev * 0.5

        # Final clip
        lev = float(np.clip(lev, cfg.min_leverage, cfg.max_leverage))
        return lev

    def size_position(
        self,
        equity: float,
        mark_price: float,
        leverage: float,
        alpha: float = 1.0,             # 0-1 conviction from momentum z-score
        max_position_pct: float = 0.20, # max fraction of equity per position
    ) -> float:
        """
        Convert leverage into a position quantity.

        notional = equity × leverage × alpha (alpha scales within leverage)
        qty = notional / mark_price
        capped at max_position_pct of equity.
        """
        notional = equity * leverage * alpha
        max_notional = equity * max_position_pct * leverage
        notional = min(notional, max_notional)
        if mark_price <= 0:
            return 0.0
        return notional / mark_price
