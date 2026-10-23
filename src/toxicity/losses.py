"""Loss functions for imbalanced multi-label training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pos_weight_from_labels(
    y: np.ndarray, power: float = 1.0, max_weight: float | None = None
) -> np.ndarray:
    """Per-label positive weight = (negatives / positives) ** power.

    power=1.0 is plain inverse frequency; power=0.5 dampens it, which keeps the
    rarest labels from dominating the loss. Labels with no positives get 1.0.
    """
    positives = y.sum(axis=0).astype(float)
    negatives = len(y) - positives
    weight = np.ones_like(positives)
    has_pos = positives > 0
    weight[has_pos] = (negatives[has_pos] / positives[has_pos]) ** power
    if max_weight is not None:
        weight = np.minimum(weight, max_weight)
    return weight


class FocalLossWithLogits(nn.Module):
    """Binary focal loss applied to each label independently (Lin et al., 2017).

    gamma=0 and alpha=None reduce it to plain binary cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_correct = prob * targets + (1 - prob) * (1 - targets)
        loss = (1 - p_correct) ** self.gamma * bce
        if self.alpha is not None:
            loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * loss
        return loss.mean()


def build_loss(name: str, y_train: np.ndarray, train_cfg: dict) -> nn.Module:
    """Create the loss named in the config: bce, weighted_bce or focal."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        weight = pos_weight_from_labels(
            y_train,
            power=train_cfg.get("pos_weight_power", 1.0),
            max_weight=train_cfg.get("pos_weight_max"),
        )
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    if name == "focal":
        return FocalLossWithLogits(
            gamma=train_cfg.get("focal_gamma", 2.0), alpha=train_cfg.get("focal_alpha")
        )
    raise ValueError(f"Unknown loss '{name}'. Use bce, weighted_bce or focal.")
