"""LOB (Limit Order Book) microstructure feature extraction.

Derived from the LOB literature — specifically:
  Schnaubelt (EJOR 2022): "queue imbalances provide the highest value among features;
    order placement becomes more aggressive in anticipation of lower execution
    probabilities, which is indicated by trade and order imbalances"
  TLOB (arXiv 2502.15757): top-10 bid/ask levels as input; spatial + temporal attention

We extract three compact features from the top-N levels of the L2 book:

  spread_bps   — bid-ask spread in basis points; proxy for liquidity cost
  imbalance    — (bid_vol − ask_vol) / total_vol ∈ [−1, 1];
                 positive = buy pressure, negative = sell pressure
  depth_ratio  — log(bid_depth / ask_depth), clipped to [−2, 2];
                 independent of total volume, captures skew

These three features add 3 dimensions to any feature vector and are computable
from a single L2 snapshot with no additional history required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LOBFeatures:
    spread_bps: float    # ≥ 0; clipped at 50bps
    imbalance: float     # ∈ [−1, 1]
    depth_ratio: float   # ∈ [−2, 2]

    def to_array(self) -> list[float]:
        return [self.spread_bps / 50.0,   # normalise to [0, 1]
                self.imbalance,            # already [−1, 1]
                self.depth_ratio / 2.0]   # normalise to [−1, 1]


NULL_LOB = LOBFeatures(spread_bps=0.0, imbalance=0.0, depth_ratio=0.0)


def extract_lob_features(order_book: dict, n_levels: int = 5) -> LOBFeatures:
    """
    Extract LOBFeatures from a ccxt-style order book dict.

    order_book format (from ccxt fetch_order_book):
        {"bids": [[price, size], ...], "asks": [[price, size], ...]}

    Returns NULL_LOB if the book is empty or malformed.
    """
    bids = order_book.get("bids", [])[:n_levels]
    asks = order_book.get("asks", [])[:n_levels]

    if not bids or not asks:
        return NULL_LOB

    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
    except (IndexError, TypeError, ValueError):
        return NULL_LOB

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return NULL_LOB

    mid = (best_bid + best_ask) / 2.0
    spread_bps = float(np.clip((best_ask - best_bid) / mid * 10_000, 0.0, 50.0))

    bid_vol = sum(float(b[1]) for b in bids if len(b) >= 2)
    ask_vol = sum(float(a[1]) for a in asks if len(a) >= 2)
    total = bid_vol + ask_vol
    imbalance = float(np.clip((bid_vol - ask_vol) / total, -1.0, 1.0)) if total > 0 else 0.0

    depth_ratio = float(np.clip(np.log(bid_vol / (ask_vol + 1e-9)), -2.0, 2.0))

    return LOBFeatures(spread_bps=spread_bps, imbalance=imbalance, depth_ratio=depth_ratio)
