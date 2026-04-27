"""RL risk manager — wraps the trained PPO agent with circuit breakers.

Sits between the PPO model output and live order execution:
  1. Queries raw leverage action from model
  2. Scales by regime multiplier (from RegimeDetector)
  3. Applies circuit breakers (drawdown, funding, volatility)
  4. Returns final target leverage in [-MAX_LEVERAGE, +MAX_LEVERAGE]

Circuit breakers (hard stops — override model):
  - Daily drawdown > 8%  → flatten (leverage = 0)
  - Session drawdown > 5% → halve model output
  - Realized vol > 3× target → scale down proportionally
  - Funding |z| > 3.0 → reduce leverage by 40%
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MAX_LEVERAGE = 10.0


@dataclass
class RiskConfig:
    target_vol: float = 0.50          # annualized target vol for position sizing
    max_leverage: float = MAX_LEVERAGE
    dd_soft: float = 0.05             # session drawdown → halve leverage
    dd_hard: float = 0.08             # daily drawdown → flatten
    vol_scale_cap: float = 3.0        # vol multiples before scaling kicks in
    funding_z_cap: float = 3.0        # |funding z| above which we cut leverage 40%
    min_regime_prob: float = 0.50     # below this confidence, use conservative leverage


@dataclass
class RiskState:
    peak_equity: float = 0.0
    session_start_equity: float = 0.0
    daily_start_equity: float = 0.0
    is_hard_stopped: bool = False
    soft_stop_active: bool = False
    _realized_vol_window: list = field(default_factory=list)


class RiskManager:
    """
    Stateful risk wrapper around the PPO model.

    Usage:
        rm = RiskManager(initial_equity=10_000)
        lev = rm.compute(
            raw_action=model.predict(obs)[0],
            equity=current_equity,
            realized_vol=rolling_24h_vol,
            regime=regime_label,
            regime_prob=regime_confidence,
            funding_z=funding_zscore,
        )
    """

    def __init__(self, initial_equity: float, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self._state = RiskState(
            peak_equity=initial_equity,
            session_start_equity=initial_equity,
            daily_start_equity=initial_equity,
        )
        from src.regime.detector import LEVERAGE_MULTIPLIERS
        self._regime_mult = LEVERAGE_MULTIPLIERS

    def reset_daily(self, equity: float) -> None:
        self._state.daily_start_equity = equity
        self._state.is_hard_stopped = False
        logger.info("RiskManager: daily reset at equity=%.2f", equity)

    def reset_session(self, equity: float) -> None:
        self._state.session_start_equity = equity
        self._state.soft_stop_active = False

    def compute(
        self,
        raw_action: float,
        equity: float,
        realized_vol: float,
        regime: int,
        regime_prob: float,
        funding_z: float = 0.0,
    ) -> float:
        """Return final target leverage scalar in [-max_leverage, +max_leverage]."""
        # Update peak
        self._state.peak_equity = max(self._state.peak_equity, equity)

        # 1. Hard circuit breaker: daily drawdown
        daily_dd = (self._state.daily_start_equity - equity) / (self._state.daily_start_equity + 1e-9)
        if daily_dd > self.cfg.dd_hard or self._state.is_hard_stopped:
            self._state.is_hard_stopped = True
            logger.warning("RiskManager: HARD STOP | daily_dd=%.2f%%", daily_dd * 100)
            return 0.0

        # 2. Soft circuit breaker: session drawdown
        session_dd = (self._state.session_start_equity - equity) / (self._state.session_start_equity + 1e-9)
        soft_scale = 0.5 if session_dd > self.cfg.dd_soft else 1.0
        if session_dd > self.cfg.dd_soft:
            logger.info("RiskManager: soft stop active (session_dd=%.2f%%)", session_dd * 100)

        # 3. Regime multiplier
        base_regime_mult = self._regime_mult.get(regime, 0.5)
        confidence_scale = max(self.cfg.min_regime_prob, float(regime_prob))
        regime_scale = base_regime_mult * confidence_scale

        # 4. Volatility scaling: target_vol / realized_vol, capped
        if realized_vol > 1e-6:
            bars_per_year = 365 * 24 * 4  # 15m bars
            realized_vol_ann = realized_vol * np.sqrt(bars_per_year)
            vol_scale = min(self.cfg.target_vol / realized_vol_ann, self.cfg.vol_scale_cap)
        else:
            vol_scale = 1.0

        # 5. Funding penalty
        if abs(funding_z) > self.cfg.funding_z_cap:
            excess = abs(funding_z) - self.cfg.funding_z_cap
            funding_scale = max(0.4, 1.0 - 0.1 * excess)
        else:
            funding_scale = 1.0

        # Compose
        scale = soft_scale * regime_scale * vol_scale * funding_scale
        target_lev = float(raw_action) * self.cfg.max_leverage * scale
        target_lev = float(np.clip(target_lev, -self.cfg.max_leverage, self.cfg.max_leverage))

        return target_lev

    def is_stopped(self) -> bool:
        return self._state.is_hard_stopped
