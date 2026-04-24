"""Bar-level feature engineering for the Transformer classifier.

Each bar is described by a 10-dim vector. During training, we stack
SEQ_LEN consecutive bar feature vectors to form the model input.
"""
from __future__ import annotations

import numpy as np

from .bars import Bar

SEQ_LEN = 32
BAR_FEATURE_DIM = 10
BAR_FEATURE_NAMES = (
    "log_return",        # bar open→close log-return
    "hl_ratio",          # log(high/low) — intra-bar range
    "close_vwap_dev",    # log(close/vwap) — close vs average fill price
    "vol_zscore",        # volume z-score over trailing 20 bars
    "dollar_vol_zscore", # dollar-volume z-score over trailing 20 bars
    "momentum_5",        # cumulative log-return over last 5 bars
    "momentum_20",       # cumulative log-return over last 20 bars
    "realized_vol_20",   # realized volatility (std of log-returns, 20 bars)
    "rsi_norm",          # RSI(14) rescaled to [-1, 1]
    "log_tick_count",    # log(tick_count) — proxy for bar duration
)


def _rsi(log_rets: np.ndarray, period: int = 14) -> float:
    if len(log_rets) < period:
        return 0.0
    window = log_rets[-period:]
    gains = window[window > 0].sum()
    losses = -window[window < 0].sum()
    if losses < 1e-12:
        return 1.0
    rs = gains / losses
    rsi = 1.0 - 1.0 / (1.0 + rs)  # normalized to [0,1]
    return float(rsi * 2 - 1)      # rescale to [-1, 1]


def compute_bar_features(bars: list[Bar], idx: int) -> np.ndarray:
    """10-dim feature vector for bar at position `idx` in `bars`."""
    b = bars[idx]

    # Trailing log-return series
    start = max(0, idx - 20)
    closes = np.array([bars[i].close for i in range(start, idx + 1)])
    log_rets = np.diff(np.log(np.maximum(closes, 1e-12)))

    log_return = float(np.log(b.close / b.open) if b.open > 0 else 0.0)
    hl_ratio = float(np.log(b.high / b.low) if b.low > 0 else 0.0)
    close_vwap = float(np.log(b.close / b.vwap) if b.vwap > 0 else 0.0)

    # Volume z-scores over trailing 20 bars
    look = bars[max(0, idx - 19):idx + 1]
    vols = np.array([x.volume for x in look])
    dvols = np.array([x.dollar_volume for x in look])
    vol_z = float((vols[-1] - vols.mean()) / (vols.std() + 1e-9))
    dvol_z = float((dvols[-1] - dvols.mean()) / (dvols.std() + 1e-9))

    # Momentum
    mom5 = float(log_rets[-5:].sum()) if len(log_rets) >= 5 else float(log_rets.sum())
    mom20 = float(log_rets.sum())

    # Realized vol
    rv20 = float(log_rets.std()) if len(log_rets) >= 2 else 0.0

    rsi_val = _rsi(log_rets)
    log_tick = float(np.log1p(b.tick_count))

    return np.array(
        [log_return, hl_ratio, close_vwap, vol_z, dvol_z,
         mom5, mom20, rv20, rsi_val, log_tick],
        dtype=np.float32,
    )


def build_feature_sequences(
    bars: list[Bar],
    event_t0s: list[int],
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Build input matrix for the Transformer.

    Returns array of shape (n_events, seq_len, BAR_FEATURE_DIM).
    For each event t0, the sequence covers bars [t0-seq_len+1 .. t0].
    Events with insufficient history are padded with zeros on the left.
    """
    # Pre-compute all bar features
    all_feats = np.zeros((len(bars), BAR_FEATURE_DIM), dtype=np.float32)
    for i in range(len(bars)):
        all_feats[i] = compute_bar_features(bars, i)

    X = np.zeros((len(event_t0s), seq_len, BAR_FEATURE_DIM), dtype=np.float32)
    for n, t0 in enumerate(event_t0s):
        start = t0 - seq_len + 1
        if start < 0:
            # left-pad with zeros
            valid = all_feats[0:t0 + 1]
            X[n, seq_len - len(valid):] = valid
        else:
            X[n] = all_feats[start:t0 + 1]

    return X


def live_sequence(bars: list[Bar], seq_len: int = SEQ_LEN) -> np.ndarray:
    """Build a single (1, seq_len, BAR_FEATURE_DIM) tensor from the most recent bars."""
    seq = build_feature_sequences(bars, [len(bars) - 1], seq_len)
    return seq  # shape (1, seq_len, BAR_FEATURE_DIM)
