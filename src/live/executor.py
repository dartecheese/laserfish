"""Live order executor for the PPO RL agent.

Bridges the trained model to real Hyperliquid orders:
  1. Fetches latest bars from exchange
  2. Builds features via FeatureEngineer
  3. Runs RegimeDetector for live regime
  4. Queries PPO model for target leverage
  5. Applies RiskManager circuit breakers
  6. Computes target position qty
  7. Places / cancels orders to reach target

Designed to run as a 15-minute cron alongside the momentum/grid bot in run.py.
Can be toggled off via paper_mode=True without changing anything else.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH   = os.getenv("RL_MODEL_PATH", "models/rl_btc_perp.zip")
LOOKBACK     = 800       # bars of history for regime + features
BARS_PER_YEAR = 365 * 24 * 4


class RLExecutor:
    """
    Live inference + execution for the PPO agent.

    Usage in run.py:
        executor = RLExecutor(exchange, paper_mode=True)
        executor.step(mark_price, equity)   # call every 15 minutes
    """

    def __init__(
        self,
        exchange,
        symbol: str = "BTC",
        paper_mode: bool = True,
        initial_equity: float = 10_000.0,
    ):
        self.ex = exchange
        self.symbol = symbol
        self.paper_mode = paper_mode
        self._model = None
        self._risk = None
        self._initial_equity = initial_equity

        # Rolling history buffers
        self._closes:   deque[float] = deque(maxlen=LOOKBACK)
        self._fundings: deque[float] = deque(maxlen=LOOKBACK)
        self._volumes:  deque[float] = deque(maxlen=LOOKBACK)

        self._position = 0.0        # current open qty
        self._last_action = 0.0

        self._load_model()
        from src.risk.manager import RiskManager
        self._risk = RiskManager(initial_equity=initial_equity)

    def step(self, mark_price: float, equity: float) -> float:
        """
        Main inference + execution call.
        Returns the target leverage applied this bar.
        """
        if self._model is None:
            return 0.0

        self._closes.append(mark_price)

        try:
            funding = self.ex.get_funding_rate(self.symbol) or 0.0
        except Exception:
            funding = 0.0
        self._fundings.append(funding)

        if len(self._closes) < 50:
            return 0.0

        obs = self._build_obs(mark_price, equity)
        if obs is None:
            return 0.0

        raw_action, _ = self._model.predict(obs, deterministic=True)
        raw_action = float(raw_action[0])

        regime, regime_prob, funding_z, realized_vol = self._compute_regime_stats()

        target_lev = self._risk.compute(
            raw_action=raw_action,
            equity=equity,
            realized_vol=realized_vol,
            regime=regime,
            regime_prob=regime_prob,
            funding_z=funding_z,
        )

        if not self._risk.is_stopped():
            self._execute(mark_price, equity, target_lev)
        else:
            self._flatten(mark_price)

        self._last_action = target_lev
        return target_lev

    def reset_daily(self, equity: float) -> None:
        if self._risk:
            self._risk.reset_daily(equity)

    @property
    def last_action(self) -> float:
        return self._last_action

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if not Path(MODEL_PATH).exists():
            logger.warning("RLExecutor: no model at %s — inference disabled", MODEL_PATH)
            return
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(MODEL_PATH)
            logger.info("RLExecutor: model loaded from %s", MODEL_PATH)
        except Exception as e:
            logger.error("RLExecutor: failed to load model: %s", e)

    def _build_obs(self, mark_price: float, equity: float) -> np.ndarray | None:
        """Build the observation vector matching HyperliquidBTCEnv._obs()."""
        try:
            from src.data.features import FeatureEngineer, FEATURE_COLUMNS
            import pandas as pd

            closes  = np.array(self._closes)
            fundings = np.array(self._fundings)

            df = pd.DataFrame({
                "open":  closes,
                "high":  closes * 1.001,
                "low":   closes * 0.999,
                "close": closes,
                "volume": np.ones(len(closes)),
                "funding_rate": fundings,
            })

            fe = FeatureEngineer()
            df = fe.build(df)
            # Fill regime columns with live values
            regime, regime_prob, _, _ = self._compute_regime_stats()
            df["regime_label"]         = float(regime)
            df["regime_prob"]          = float(regime_prob)
            df["regime_duration_norm"] = 0.5

            from src.data.features import N_FEATURES
            if len(df) == 0:
                return None

            feats = df[FEATURE_COLUMNS].values[-1].astype(np.float32)

            notional = self._position * mark_price
            pos_norm    = notional / (self._initial_equity + 1e-9)
            unreal_norm = 0.0
            time_norm   = 0.5

            obs = np.concatenate([feats, [pos_norm, unreal_norm, time_norm]])
            return obs.astype(np.float32)

        except Exception as e:
            logger.debug("_build_obs error: %s", e)
            return None

    def _compute_regime_stats(self) -> tuple[int, float, float, float]:
        """Returns (regime, regime_prob, funding_z, realized_vol)."""
        closes   = np.array(self._closes)
        fundings = np.array(self._fundings)

        if len(closes) < 24:
            return 1, 0.5, 0.0, 0.01

        returns = np.diff(np.log(closes + 1e-9))

        # Realized vol (24-bar)
        realized_vol = float(np.std(returns[-24:]))

        # Funding z-score
        if len(fundings) >= 10:
            mu  = float(np.mean(fundings))
            std = float(np.std(fundings)) + 1e-9
            funding_z = float((fundings[-1] - mu) / std)
        else:
            funding_z = 0.0

        # Regime
        try:
            from src.regime.detector import RegimeDetector
            det = RegimeDetector()
            if len(returns) >= 200:
                det.fit(returns, fundings[1:])
                regime, prob = det.current_regime(returns, fundings[1:])
            else:
                regime, prob = 1, 0.5
        except Exception:
            regime, prob = 1, 0.5

        return regime, prob, funding_z, realized_vol

    def _execute(self, mark_price: float, equity: float, target_lev: float) -> None:
        """Place orders to reach target leverage."""
        target_notional = target_lev * equity
        target_qty = target_notional / (mark_price + 1e-9)
        delta_qty = target_qty - self._position

        if abs(delta_qty) * mark_price < 10.0:   # < $10 notional change → skip
            return

        side = "buy" if delta_qty > 0 else "sell"
        qty  = abs(delta_qty)

        if self.paper_mode:
            logger.info(
                "RLExecutor [PAPER] | %s %.4f BTC @ ~%.2f | lev=%.1fx",
                side.upper(), qty, mark_price, target_lev,
            )
            self._position = target_qty
            return

        try:
            from trader.exchanges.base import BracketParams
            result = self.ex.place_bracket_order(BracketParams(
                symbol=self.symbol,
                side=side,
                quantity=qty,
                price=None,   # market order
                take_profit_pct=None,
                stop_loss_pct=None,
                leverage=int(abs(target_lev)) + 1,
            ))
            self._position = target_qty
            logger.info(
                "RLExecutor | %s %.4f BTC | order_id=%s | lev=%.1fx",
                side.upper(), qty, result.order_id, target_lev,
            )
        except Exception as e:
            logger.error("RLExecutor execute failed: %s", e)

    def _flatten(self, mark_price: float) -> None:
        if abs(self._position) < 1e-8:
            return
        logger.warning("RLExecutor: flattening position due to hard stop")
        if not self.paper_mode:
            side = "sell" if self._position > 0 else "buy"
            try:
                from trader.exchanges.base import BracketParams
                self.ex.place_bracket_order(BracketParams(
                    symbol=self.symbol, side=side,
                    quantity=abs(self._position), price=None, leverage=1,
                ))
            except Exception as e:
                logger.error("RLExecutor flatten failed: %s", e)
        self._position = 0.0
