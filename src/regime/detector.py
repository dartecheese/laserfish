"""4-state regime detector: HMM + GARCH.

States:
  0 = BEAR     : trending down, elevated vol, negative funding
  1 = RANGING  : sideways, mean-reverting, low vol
  2 = BULL     : trending up, moderate vol, positive funding
  3 = VOLATILE : explosive/crash, very high vol, funding extremes

Architecture:
  - GARCH(1,1) estimates conditional volatility series
  - GaussianHMM clusters (return, cond_vol, funding) into 4 states
  - States sorted by mean return so labels are consistent across fits

Extends the 3-state trader/regime.py with GARCH vol and 4 states.
Used by the RL training pipeline. For live inference, trader/regime.py
(lighter, online-updatable) is preferred.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: "BEAR", 1: "RANGING", 2: "BULL", 3: "VOLATILE"}
LEVERAGE_MULTIPLIERS = {0: 0.3, 1: 0.7, 2: 1.0, 3: 0.2}


class RegimeDetector:
    """
    4-state HMM + GARCH regime classifier.

    Usage:
        det = RegimeDetector()
        det.fit(returns, funding_rates)
        labels, probs = det.predict(returns, funding_rates)
        lev = det.leverage_multiplier(labels[-1], probs[-1])
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._hmm: GaussianHMM | None = None
        self._garch_params: dict | None = None
        self._fitted = False
        self._state_map: dict[int, int] = {}
        self._obs_mean: np.ndarray | None = None
        self._obs_std:  np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, returns: np.ndarray, funding_rates: np.ndarray | None = None) -> None:
        """
        Fit the GARCH + HMM on historical data.

        returns:       1-D array of bar log-returns
        funding_rates: 1-D array aligned with returns (optional)
        """
        if len(returns) < 200:
            logger.warning("RegimeDetector: need ≥200 obs to fit, got %d", len(returns))
            return

        # Step 1: GARCH conditional vol
        cond_vol = self._fit_garch(returns)

        # Step 2: Build observation matrix
        if funding_rates is not None and len(funding_rates) == len(returns):
            obs = np.column_stack([returns, cond_vol, funding_rates])
        else:
            obs = np.column_stack([returns, cond_vol])

        # Step 3: Normalize obs to prevent ill-conditioned covariance
        self._obs_mean = obs.mean(axis=0)
        self._obs_std  = obs.std(axis=0) + 1e-8
        obs_norm = (obs - self._obs_mean) / self._obs_std

        # Step 4: Fit HMM — use "diag" covariance for numerical stability
        hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=self.random_state,
        )
        hmm.fit(obs_norm)
        obs = obs_norm
        self._hmm = hmm

        # Step 4: Sort states by mean return (ascending → 0=bear, n-1=bull)
        #         Highest-vol state → VOLATILE (state 3)
        mean_returns = hmm.means_[:, 0]
        mean_vols    = hmm.means_[:, 1]
        vol_order    = int(np.argmax(mean_vols))

        # Sort non-volatile states by return
        non_vol = [i for i in range(self.n_regimes) if i != vol_order]
        sorted_by_ret = sorted(non_vol, key=lambda i: mean_returns[i])

        # Map: sorted[0]=BEAR(0), sorted[1]=RANGING(1), sorted[2]=BULL(2), vol=VOLATILE(3)
        self._state_map = {}
        for canonical, raw in enumerate(sorted_by_ret):
            self._state_map[raw] = canonical
        self._state_map[vol_order] = 3

        self._fitted = True
        self._log_fit_summary(hmm)

    def _fit_garch(self, returns: np.ndarray) -> np.ndarray:
        """Returns conditional volatility series via GARCH(1,1) or rolling std fallback."""
        if ARCH_AVAILABLE:
            try:
                model = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
                res = model.fit(disp="off", show_warning=False)
                cond_vol = res.conditional_volatility / 100
                # Store params for online update
                self._garch_params = dict(res.params)
                return cond_vol
            except Exception as e:
                logger.debug("GARCH fit failed (%s) — using rolling std", e)

        # Fallback: 24-bar rolling std
        cond_vol = pd.Series(returns).rolling(24, min_periods=5).std().fillna(
            np.std(returns[:24]) if len(returns) >= 24 else np.std(returns)
        ).values
        return cond_vol

    def _log_fit_summary(self, hmm: GaussianHMM) -> None:
        logger.info("RegimeDetector fitted | %d states", self.n_regimes)
        for raw, canonical in self._state_map.items():
            name = REGIME_NAMES[canonical]
            mu  = hmm.means_[raw, 0] * 1e4   # in bps
            vol = hmm.means_[raw, 1] * 1e4
            logger.info("  %s (raw=%d) : mean_ret=%+.1fbps  mean_vol=%.1fbps",
                        name, raw, mu, vol)

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        returns: np.ndarray,
        funding_rates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (labels, probs) arrays of same length as returns.
        labels: canonical regime index (0=BEAR, 1=RANGING, 2=BULL, 3=VOLATILE)
        probs:  max posterior probability for each timestep
        """
        if not self._fitted or self._hmm is None:
            n = len(returns)
            return np.ones(n, dtype=int), np.full(n, 0.5)

        cond_vol = self._fit_garch(returns)
        if funding_rates is not None and len(funding_rates) == len(returns):
            obs = np.column_stack([returns, cond_vol, funding_rates])
        else:
            obs = np.column_stack([returns, cond_vol])

        # Normalize with training stats if available
        if hasattr(self, "_obs_mean") and self._obs_mean is not None:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)

        try:
            raw_states = self._hmm.predict(obs)
            posteriors  = self._hmm.predict_proba(obs)
            labels = np.array([self._state_map.get(int(s), 1) for s in raw_states])
            probs  = posteriors.max(axis=1)
            return labels, probs
        except Exception as e:
            logger.warning("RegimeDetector predict failed: %s", e)
            n = len(returns)
            return np.ones(n, dtype=int), np.full(n, 0.5)

    def current_regime(
        self,
        returns: np.ndarray,
        funding_rates: np.ndarray | None = None,
    ) -> tuple[int, float]:
        """Convenience: return (regime_label, prob) for the most recent bar."""
        labels, probs = self.predict(returns, funding_rates)
        return int(labels[-1]), float(probs[-1])

    # ------------------------------------------------------------------ #
    # Leverage                                                             #
    # ------------------------------------------------------------------ #

    def leverage_multiplier(self, regime: int, regime_prob: float) -> float:
        base = LEVERAGE_MULTIPLIERS.get(regime, 0.5)
        # Scale by confidence: uncertain regime → conservative
        confidence_scale = max(0.5, float(regime_prob))
        return base * confidence_scale

    def regime_name(self, label: int) -> str:
        return REGIME_NAMES.get(label, "UNKNOWN")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"hmm": self._hmm, "state_map": self._state_map,
                         "garch_params": self._garch_params, "fitted": self._fitted,
                         "obs_mean": self._obs_mean, "obs_std": self._obs_std}, f)
        logger.info("RegimeDetector saved → %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._hmm = d["hmm"]
        self._state_map = d["state_map"]
        self._garch_params = d.get("garch_params")
        self._fitted = d["fitted"]
        self._obs_mean = d.get("obs_mean")
        self._obs_std  = d.get("obs_std")
        logger.info("RegimeDetector loaded ← %s", path)


# Avoid circular import if arch not installed
try:
    import pandas as pd
except ImportError:
    pass
