"""Walk-forward Transformer training pipeline.

Precomputes all bars, features, and labels once across the full dataset,
then slices by timestamp index for each fold — eliminating the O(folds)
redundant recomputation that dominated runtime.

Usage (via scripts/train.py):
    python scripts/train.py --symbols BTC,ETH,SOL,AVAX,LINK \
        --out models/transformer_5m.onnx
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .bars import Bar, make_volume_bars, make_dollar_bars, cusum_filter, auto_threshold
from .data import Kline
from .features import build_feature_sequences, compute_bar_features, SEQ_LEN, BAR_FEATURE_DIM
from .labeling import triple_barrier_labels, LabeledEvent, class_weights
from .model import CryptoTransformer, export_onnx


@dataclass
class TrainConfig:
    bar_type: str = "volume"
    target_bars_day: float = 20.0
    cusum_h: float = 0.005
    pt: float = 0.025
    sl: float = 0.025
    t_max: int = 24
    seq_len: int = SEQ_LEN

    # Model
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 50
    patience: int = 10
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


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    events_train: list[LabeledEvent],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[CryptoTransformer, float]:
    weights = torch.tensor(class_weights(events_train), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CryptoTransformer(
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=False)

    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    use_amp = device.type in ("cuda", "mps")

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = criterion(model(xb), yb)
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
            else:
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


def train_walk_forward(
    klines_by_symbol: dict[str, list[Kline]],
    cfg: TrainConfig,
    out_path: str | Path,
    device: torch.device | None = None,
) -> None:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    # Build bars for all symbols
    all_bars: list[Bar] = []
    for sym, klines in klines_by_symbol.items():
        b = _make_bars(klines, cfg)
        print(f"  {sym}: {len(klines)} klines → {len(b)} {cfg.bar_type} bars")
        all_bars.extend(b)
    all_bars.sort(key=lambda b: b.timestamp)

    n = len(all_bars)
    if n < cfg.seq_len + cfg.t_max + 10:
        raise ValueError("Not enough bars for training.")

    # ── Precompute everything once ──────────────────────────────────────
    # Bar features: (n, BAR_FEATURE_DIM)
    print(f"Precomputing features for {n} bars…")
    t0 = time.time()
    all_feats = np.zeros((n, BAR_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        all_feats[i] = compute_bar_features(all_bars, i)

    # CUSUM events + triple-barrier labels on full series
    event_indices = cusum_filter(all_bars, cfg.cusum_h)
    all_events = triple_barrier_labels(all_bars, event_indices, cfg.pt, cfg.sl, cfg.t_max)
    print(f"  {len(all_events)} labeled events  ({time.time()-t0:.0f}s)")

    if not all_events:
        raise RuntimeError("No labeled events — check CUSUM threshold and data length.")

    # Build full X, y once — shape (E, seq_len, feat)
    print("Building feature sequences…")
    t0 = time.time()
    t0s = [e.t0 for e in all_events]
    timestamps = np.array([all_bars[e.t0].timestamp for e in all_events])

    # Build sequences directly from precomputed features (avoids re-calling compute_bar_features)
    E = len(t0s)
    X_all = np.zeros((E, cfg.seq_len, BAR_FEATURE_DIM), dtype=np.float32)
    for idx, t0_bar in enumerate(t0s):
        start = t0_bar - cfg.seq_len + 1
        if start < 0:
            valid = all_feats[0:t0_bar + 1]
            X_all[idx, cfg.seq_len - len(valid):] = valid
        else:
            X_all[idx] = all_feats[start:t0_bar + 1]

    y_all = np.array([e.label + 1 for e in all_events], dtype=np.int64)
    bar_timestamps = np.array([all_bars[i].timestamp for i in range(n)])
    print(f"  X_all shape: {X_all.shape}  ({time.time()-t0:.0f}s)")

    # ── Walk-forward folds (just array slicing now) ─────────────────────
    start_ms = all_bars[0].timestamp
    end_ms = all_bars[-1].timestamp
    ms_per_month = 30 * 24 * 3600 * 1000

    best_model: CryptoTransformer | None = None
    best_acc = 0.0
    fold = 0
    cursor = start_ms

    while True:
        train_end_ms = cursor + cfg.train_months * ms_per_month
        test_end_ms = train_end_ms + cfg.test_months * ms_per_month
        if test_end_ms > end_ms:
            break

        tr_mask = timestamps < train_end_ms
        te_mask = (timestamps >= train_end_ms) & (timestamps < test_end_ms)

        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        ev_tr = [e for e, m in zip(all_events, tr_mask) if m]

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
