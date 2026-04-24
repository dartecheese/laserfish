"""Triple-barrier labeling (Lopez de Prado, AFML ch. 3).

For each candidate entry bar, the label is determined by which barrier
is touched first:
  +1  upper barrier hit (profit target)
  -1  lower barrier hit (stop loss)
   0  vertical barrier hit (time limit, close at market)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bars import Bar


@dataclass
class LabeledEvent:
    t0: int     # entry bar index
    t1: int     # exit bar index
    label: int  # +1, -1, or 0
    ret: float  # realized log-return


def triple_barrier_labels(
    bars: list[Bar],
    event_indices: list[int],
    pt: float = 0.025,
    sl: float = 0.025,
    t_max: int = 24,
) -> list[LabeledEvent]:
    """
    Args:
        bars: Full bar series.
        event_indices: Candidate entry indices (from cusum_filter).
        pt: Profit-taking fraction (e.g. 0.025 = 2.5%).
        sl: Stop-loss fraction.
        t_max: Max bars to hold (vertical barrier).
    """
    closes = np.array([b.close for b in bars])
    log_closes = np.log(closes)
    n = len(bars)
    events: list[LabeledEvent] = []

    for idx in event_indices:
        if idx >= n - 1:
            continue
        entry = closes[idx]
        upper = entry * (1 + pt)
        lower = entry * (1 - sl)
        horizon = min(idx + t_max, n - 1)

        label = 0
        exit_idx = horizon
        for j in range(idx + 1, horizon + 1):
            if closes[j] >= upper:
                label = 1
                exit_idx = j
                break
            if closes[j] <= lower:
                label = -1
                exit_idx = j
                break

        events.append(LabeledEvent(
            t0=idx, t1=exit_idx, label=label,
            ret=log_closes[exit_idx] - log_closes[idx],
        ))

    return events


def class_weights(events: list[LabeledEvent]) -> np.ndarray:
    """Inverse-frequency weights for CrossEntropyLoss. Maps {-1,0,+1} → {0,1,2}."""
    mapped = np.array([e.label + 1 for e in events])
    counts = np.bincount(mapped, minlength=3).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return (counts.sum() / (3.0 * counts)).astype(np.float32)
