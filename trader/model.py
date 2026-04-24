"""Transformer classifier for triple-barrier signal prediction.

Input:  (batch, SEQ_LEN, BAR_FEATURE_DIM) — sequence of bar feature vectors
Output: (batch, 3) logits — classes are SELL=0, HOLD=1, BUY=2

During inference, alpha = softmax(logits)[BUY] - softmax(logits)[SELL] ∈ (-1, 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import SEQ_LEN, BAR_FEATURE_DIM

N_CLASSES = 3   # SELL=0, HOLD=1, BUY=2


class CryptoTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = BAR_FEATURE_DIM,
        seq_len: int = SEQ_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h[:, -1, :])   # classify from last token


def logits_to_alpha(logits: np.ndarray) -> float:
    """Convert 3-class logits to a directional alpha ∈ (-1, 1)."""
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return float(probs[2] - probs[0])   # P(BUY) - P(SELL)


def export_onnx(model: CryptoTransformer, path: str | Path) -> None:
    """Export the model to ONNX with fixed seq_len and dynamic batch size."""
    model.eval()
    dummy = torch.zeros(1, model.seq_len, BAR_FEATURE_DIM)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["bars"],
        output_names=["logits"],
        dynamic_axes={"bars": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX → {path}")


def load_onnx(path: str | Path):
    """Load an ONNX model for inference. Returns an onnxruntime.InferenceSession."""
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def infer_alpha(session, bars_seq: np.ndarray) -> float:
    """Run inference on a (1, seq_len, BAR_FEATURE_DIM) numpy array.
    Returns alpha ∈ (-1, 1).
    """
    logits = session.run(["logits"], {"bars": bars_seq})[0][0]
    return logits_to_alpha(logits)
