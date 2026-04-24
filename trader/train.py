"""Walk-forward Transformer training pipeline.

Paper approach: for each (symbol, bar_type, threshold) combination,
train on a rolling window, evaluate on the next out-of-sample period,
step forward, and repeat. Export the best model as ONNX.

Usage (via scripts/train.py):
    python scripts/train.py --symbols BTC,ETH --interval 1h \
        --bar-type volume --pt 0.025 --sl 0.025 --t-max 24 \
        --out models/transformer.onnx
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .bars import Bar, make_volume_bars, make_dollar_bars, cusum_filter, auto_threshold
from .data import Kline
from .features import build_feature_sequences, SEQ_LEN, BAR_FEATURE_DIM
from .labeling import triple_barrier_labels, LabeledEvent, class_weights
from .model import CryptoTransformer, N_CLASSES, export_onnx


@dataclass
class TrainConfig:
    bar_type: str = "volume"        # "volume" or "dollar"
    target_bars_day: float = 20.0   # for auto_threshold
    cusum_h: float = 0.005          # CUSUM threshold (log-return units)
    pt: float = 0.025               # profit target
    sl: float = 0.025               # stop loss
    t_max: int = 24                 # vertical barrier (bars)
    seq_len: int = SEQ_LEN

    # Model
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 50
    patience: int = 10              # early stopping
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Walk-forward
    train_months: int = 18
    test_months: int = 3
    step_months: int = 3


def _make_bars(klines: list[Kline], cfg: TrainConfig) -> list[Bar]:
    threshold = auto_threshold(klines, cfg.target_bars_day, cfg.bar_type)
    if cfg.bar_type == "volume":
        return make_volume_bars(klines, threshold)
    return make_dollar_bars(klines, threshold)


def _build_dataset(
    bars: list[Bar], cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, list[LabeledEvent]]:
    """Returns X (n, seq_len, feat), y (n,), events."""
    events_idx = cusum_filter(bars, cfg.cusum_h)
    if not events_idx:
        return np.empty((0, cfg.seq_len, BAR_FEATURE_DIM)), np.empty(0), []

    events = triple_barrier_labels(bars, events_idx, cfg.pt, cfg.sl, cfg.t_max)
    if not events:
        return np.empty((0, cfg.seq_len, BAR_FEATURE_DIM)), np.empty(0), []

    t0s = [e.t0 for e in events]
    X = build_feature_sequences(bars, t0s, cfg.seq_len)
    y = np.array([e.label + 1 for e in events], dtype=np.int64)   # {-1,0,1}→{0,1,2}
    return X, y, events


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    events_train: list[LabeledEvent],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[CryptoTransformer, float]:
    """Train one walk-forward fold. Returns (model, best_val_acc)."""
    weights = torch.tensor(class_weights(events_train), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CryptoTransformer(
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def _split_by_timestamp(bars: list[Bar], split_ms: int) -> tuple[list[Bar], list[Bar]]:
    for i, b in enumerate(bars):
        if b.timestamp >= split_ms:
            return bars[:i], bars[i:]
    return bars, []


def train_walk_forward(
    klines_by_symbol: dict[str, list[Kline]],
    cfg: TrainConfig,
    out_path: str | Path,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pool bars from all symbols
    all_bars: list[Bar] = []
    for sym, klines in klines_by_symbol.items():
        b = _make_bars(klines, cfg)
        print(f"  {sym}: {len(klines)} klines → {len(b)} {cfg.bar_type} bars")
        all_bars.extend(b)

    # Sort by timestamp (mixing symbols in time order)
    all_bars.sort(key=lambda b: b.timestamp)

    if len(all_bars) < cfg.seq_len + cfg.t_max + 10:
        raise ValueError("Not enough bars for training. Fetch more data.")

    # Walk-forward splits
    start_ms = all_bars[0].timestamp
    end_ms = all_bars[-1].timestamp
    ms_per_month = 30 * 24 * 3600 * 1000

    best_model: CryptoTransformer | None = None
    best_acc = 0.0
    fold = 0

    cursor = start_ms
    while True:
        train_end = cursor + cfg.train_months * ms_per_month
        test_end = train_end + cfg.test_months * ms_per_month
        if test_end > end_ms:
            break

        train_bars, rest = _split_by_timestamp(all_bars, train_end)
        test_bars, _ = _split_by_timestamp(rest, test_end)

        if len(train_bars) < cfg.seq_len + cfg.t_max + 10:
            cursor += cfg.step_months * ms_per_month
            continue

        X_tr, y_tr, ev_tr = _build_dataset(train_bars, cfg)
        X_te, y_te, ev_te = _build_dataset(test_bars, cfg)

        if len(X_tr) < 50 or len(X_te) < 10:
            cursor += cfg.step_months * ms_per_month
            continue

        fold += 1
        print(f"\nFold {fold}: train={len(X_tr)}, test={len(X_te)} events")
        t0 = time.time()
        model, val_acc = _train_one_fold(X_tr, y_tr, X_te, y_te, ev_tr, cfg, device)
        print(f"  val_acc={val_acc:.3f}  ({time.time()-t0:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

        cursor += cfg.step_months * ms_per_month

    if best_model is None:
        raise RuntimeError("No folds completed — check data coverage.")

    print(f"\nBest fold val_acc={best_acc:.3f}. Exporting ONNX → {out_path}")
    export_onnx(best_model.cpu(), out_path)
