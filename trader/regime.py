"""Market regime detector using a 3-state Gaussian HMM.

States:
  0 — Trending     : low vol, directional momentum, normal funding
  1 — Ranging      : moderate vol, choppy price action
  2 — Crisis       : high vol, funding extremes, potential liquidation cascade

Features (per observation):
  - 24h realized volatility (annualized)
  - 1h price return (log)
  - Funding rate z-score vs recent history
  - OI momentum (% change in open interest over last window)

The HMM is fit during warm-up on BTC anchor data, then updated online.
At runtime, call regime() to get the current state index.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# Annualized vol thresholds that seed the initial HMM means (helps convergence)
# These are calibrated to BTC/crypto realized vol characteristics
_SEED_MEANS = np.array([
    [0.40, 0.00,  0.0,  0.00],   # 0: Trending  — 40% vol, flat funding
    [0.70, 0.00,  0.5,  0.02],   # 1: Ranging   — 70% vol, mild funding
    [1.20, 0.00,  2.0,  0.05],   # 2: Crisis    — 120%+ vol, funding spike
])

# Regime labels and leverage multipliers
REGIME_LABEL  = {0: "TREND", 1: "RANGE", 2: "CRISIS"}
REGIME_MULT   = {0: 1.5,     1: 0.8,    2: 0.2}     # applied to base leverage
REGIME_COLORS = {0: "▲",     1: "─",    2: "▼"}


class RegimeDetector:
    """
    Online 3-state HMM regime classifier.

    Usage:
        detector = RegimeDetector()
        detector.fit(obs_matrix)          # call once after warm-up
        state = detector.regime(new_obs)  # call at each scan
    """

    # Minimum observations needed to fit the HMM
    MIN_FIT_OBS = 50

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self._hmm: GaussianHMM | None = None
        self._obs_buffer: deque[list[float]] = deque(maxlen=5000)
        self._current_state: int = 0  # default to trending until fit
        self._fitted = False
        self._random_state = random_state

    # ------------------------------------------------------------------ #
    # Feature construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_obs(
        prices: list[float],
        funding_rates: list[float],
        open_interests: list[float],
        vol_window: int = 288,   # 288 × 5m = 24h
    ) -> list[float] | None:
        """
        Build one HMM observation vector from raw price/funding/OI history.
        Returns None if insufficient data.
        """
        if len(prices) < vol_window + 2:
            return None

        # 24h realized vol (annualized)
        log_rets = np.diff(np.log(prices[-vol_window:]))
        realized_vol = float(np.std(log_rets)) * np.sqrt(288 * 365)

        # 1h log return
        ret_1h = float(np.log(prices[-1] / prices[-13])) if len(prices) >= 13 else 0.0

        # Funding rate z-score
        if len(funding_rates) >= 5:
            mu = float(np.mean(funding_rates))
            sd = float(np.std(funding_rates)) + 1e-9
            fund_z = (funding_rates[-1] - mu) / sd
        else:
            fund_z = 0.0

        # OI momentum (% change over last 12 bars = 1h)
        if len(open_interests) >= 13:
            oi_prev = open_interests[-13]
            oi_mom = (open_interests[-1] - oi_prev) / (abs(oi_prev) + 1e-9)
        else:
            oi_mom = 0.0

        return [realized_vol, ret_1h, fund_z, oi_mom]

    # ------------------------------------------------------------------ #
    # Fit / update                                                         #
    # ------------------------------------------------------------------ #

    def add_obs(self, obs: list[float]) -> None:
        """Add one observation to the rolling buffer."""
        self._obs_buffer.append(obs)

    def fit(self, obs_matrix: np.ndarray | None = None) -> bool:
        """
        Fit the HMM. Uses obs_matrix if provided, else the internal buffer.
        Returns True on success.
        """
        data = obs_matrix if obs_matrix is not None else np.array(list(self._obs_buffer))
        if len(data) < self.MIN_FIT_OBS:
            logger.warning("RegimeDetector: only %d obs — need %d to fit",
                           len(data), self.MIN_FIT_OBS)
            return False

        hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=self._random_state,
        )
        # Seed means to encourage correct state ordering
        hmm.means_prior = _SEED_MEANS
        hmm.means_weight = 1.0

        try:
            hmm.fit(data)
            self._hmm = hmm
            self._fitted = True
            # Identify which state index maps to crisis (highest vol mean)
            vol_means = hmm.means_[:, 0]
            self._crisis_idx = int(np.argmax(vol_means))
            self._trend_idx  = int(np.argmin(vol_means))
            self._range_idx  = 3 - self._crisis_idx - self._trend_idx
            logger.info(
                "RegimeDetector fit on %d obs | trend=%d(vol=%.0f%%) range=%d(%.0f%%) crisis=%d(%.0f%%)",
                len(data),
                self._trend_idx,  hmm.means_[self._trend_idx, 0]  * 100,
                self._range_idx,  hmm.means_[self._range_idx, 0]  * 100,
                self._crisis_idx, hmm.means_[self._crisis_idx, 0] * 100,
            )
            # Run predict on full history to set current state
            states = hmm.predict(data)
            self._current_state = self._normalize_state(int(states[-1]))
            return True
        except Exception as e:
            logger.warning("RegimeDetector fit failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Online inference                                                     #
    # ------------------------------------------------------------------ #

    def regime(self, obs: list[float] | None = None) -> int:
        """
        Return current regime state (0=trend, 1=range, 2=crisis).
        If obs is provided, adds it and updates state.
        """
        if obs is not None:
            self.add_obs(obs)
            if self._fitted and self._hmm is not None:
                try:
                    data = np.array(list(self._obs_buffer))
                    states = self._hmm.predict(data)
                    self._current_state = self._normalize_state(int(states[-1]))
                except Exception:
                    pass
        return self._current_state

    def regime_label(self) -> str:
        return REGIME_LABEL.get(self._current_state, "UNKNOWN")

    def leverage_multiplier(self) -> float:
        return REGIME_MULT.get(self._current_state, 1.0)

    def _normalize_state(self, raw: int) -> int:
        """Map raw HMM state index to canonical {0=trend, 1=range, 2=crisis}."""
        if not self._fitted:
            return 0
        if raw == self._trend_idx:
            return 0
        if raw == self._crisis_idx:
            return 2
        return 1
