"""Hyperliquid BTC perpetual Gymnasium trading environment.

Simulates the full cost structure of Hyperliquid perp trading:
  - Maker/taker fees (0.015% / 0.045%)
  - Hourly funding payments on open positions
  - Liquidation at 85% of theoretical liquidation price
  - Slippage on large orders

Action space:  Box[-1, 1] — target leverage
               -1.0 = 10x short, 0.0 = flat, +1.0 = 10x long

Observation:   N_FEATURES + 3 position-state features

Reward:        Sharpe-adjusted step PnL with drawdown penalty
               and turnover penalty (discourages overtrading)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.data.features import N_FEATURES

# Hyperliquid fee structure
TAKER_FEE = 0.00045    # 0.045%
MAKER_FEE = 0.00015    # 0.015%
MAX_LEVERAGE = 10.0
FUNDING_INTERVAL = 4   # bars between funding payments (4 × 15m = 1h)
LIQ_BUFFER = 0.85      # liquidate at 85% of theoretical liq price


class HyperliquidBTCEnv(gym.Env):
    """
    BTC perp trading environment for PPO/A2C training.

    data:         (N, N_FEATURES) float32 feature array — from FeatureEngineer
    price_col:    index of the close price column in data (default 3 = 4th column)
    funding_col:  index of the funding_rate column (default 9 = 10th column)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: np.ndarray,
        price_col: int = 3,
        funding_col: int = 9,
        initial_balance: float = 10_000.0,
        use_maker: bool = True,
        slippage_bps: float = 2.0,
        reward_scaling: float = 100.0,
    ):
        super().__init__()
        self.data = data.astype(np.float32)
        self.price_col = price_col
        self.funding_col = funding_col
        self.initial_balance = initial_balance
        self.fee_rate = MAKER_FEE if use_maker else TAKER_FEE
        self.slippage = slippage_bps / 10_000
        self.reward_scaling = reward_scaling

        obs_dim = data.shape[1] + 3  # features + [position, unreal_pnl, time]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )

        self._reset_state()

    # ------------------------------------------------------------------ #
    # Gymnasium interface                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        target_lev = float(np.clip(action[0], -1.0, 1.0)) * MAX_LEVERAGE
        price = self._price()
        funding = self._funding()

        # 1. Funding payment on open positions
        if self.step_idx % FUNDING_INTERVAL == 0 and self.position != 0:
            notional = abs(self.position) * price
            payment = notional * funding
            self.balance -= payment if self.position > 0 else -payment
            self._total_funding += payment

        # 2. Rebalance to target
        target_qty = (target_lev * self.balance) / (price + 1e-9)
        delta = target_qty - self.position
        if abs(delta) > 1e-9:
            cost = abs(delta) * price * (self.fee_rate + self.slippage)
            self.balance -= cost
            self._trade_count += 1
        self.position = target_qty

        # 3. Liquidation check
        liquidated = self._check_liq(price)
        if liquidated:
            self.balance *= 0.05
            self.position = 0.0

        # 4. Step to next bar
        self.step_idx += 1
        next_price = self._price()
        self.unrealized_pnl = self.position * (next_price - price)

        # 5. Reward
        equity = self.balance + self.unrealized_pnl
        self.peak_balance = max(self.peak_balance, equity)
        step_ret = self.position * (next_price - price) / (self.initial_balance + 1e-9)
        dd = (self.peak_balance - equity) / (self.peak_balance + 1e-9)
        dd_penalty = dd ** 2 * 0.5
        liq_penalty = -10.0 if liquidated else 0.0
        turnover_penalty = abs(delta) * price / self.initial_balance * 0.001

        reward = (step_ret - dd_penalty + liq_penalty - turnover_penalty) * self.reward_scaling

        done = self.step_idx >= len(self.data) - 1
        truncated = (self.balance + self.unrealized_pnl) < self.initial_balance * 0.10

        return self._obs(), float(reward), done, truncated, {
            "equity": equity, "position": self.position, "trades": self._trade_count,
        }

    def render(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _reset_state(self):
        self.step_idx = 0
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.peak_balance = float(self.initial_balance)
        self._trade_count = 0
        self._total_funding = 0.0

    def _price(self) -> float:
        idx = min(self.step_idx, len(self.data) - 1)
        # Use close price from features — raw close price must be in price_col
        # NOTE: data contains normalized features, but we store raw close separately
        # The env expects data[:,price_col] to be the actual price (not normalized)
        return float(self.data[idx, self.price_col])

    def _funding(self) -> float:
        idx = min(self.step_idx, len(self.data) - 1)
        return float(self.data[idx, self.funding_col])

    def _check_liq(self, price: float) -> bool:
        if self.position == 0:
            return False
        notional = abs(self.position) * price
        if notional < 1e-9:
            return False
        margin_ratio = self.balance / notional
        return margin_ratio < (1.0 / MAX_LEVERAGE) * (1.0 - LIQ_BUFFER)

    def _obs(self) -> np.ndarray:
        idx = min(self.step_idx, len(self.data) - 1)
        feats = self.data[idx]
        price = self._price()
        pos_norm = self.position * price / (self.initial_balance + 1e-9)
        unreal_norm = self.unrealized_pnl / (self.initial_balance + 1e-9)
        time_norm = min(self.step_idx / (96.0 * 7), 1.0)  # normalized to 1 week
        state = np.array([pos_norm, unreal_norm, time_norm], dtype=np.float32)
        return np.concatenate([feats, state])
