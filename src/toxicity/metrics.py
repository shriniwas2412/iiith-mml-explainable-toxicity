"""Per-label evaluation and decision-threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import LABELS


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] = LABELS,
    default: float = 0.5,
) -> dict[str, float]:
    """Pick, for each label, the threshold with the best F1.

    Run this on validation predictions only. A comment is flagged when its
    probability is >= the threshold. Labels with no positive example keep
    `default`.
    """
    thresholds = {}
    for j, label in enumerate(labels):
        if y_true[:, j].sum() == 0:
            thresholds[label] = default
            continue
        precision, recall, candidates = precision_recall_curve(y_true[:, j], y_prob[:, j])
        # The last precision/recall pair has no threshold attached.
        precision, recall = precision[:-1], recall[:-1]
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        thresholds[label] = float(candidates[int(np.argmax(f1))])
    return thresholds


def apply_thresholds(
    y_prob: np.ndarray, thresholds: dict[str, float] | float, labels: list[str] = LABELS
) -> np.ndarray:
    """Turn probabilities into 0/1 flags using one cutoff or one per label."""
    if isinstance(thresholds, dict):
        cutoffs = np.array([thresholds[label] for label in labels])
    else:
        cutoffs = np.full(len(labels), float(thresholds))
    return (y_prob >= cutoffs).astype(int)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float] | float = 0.5,
    labels: list[str] = LABELS,
) -> pd.DataFrame:
    """Per-label precision, recall, F1, ROC-AUC and PR-AUC, with macro and micro rows.

    ROC-AUC and PR-AUC are NaN for a label that has only one class in `y_true`.
    Macro ROC-AUC is the mean column-wise AUC that the Kaggle competition used.
    """
    y_pred = apply_thresholds(y_prob, thresholds, labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    rows = []
    for j, label in enumerate(labels):
        both_classes = 0 < y_true[:, j].sum() < len(y_true)
        rows.append(
            {
                "label": label,
                "support": int(y_true[:, j].sum()),
                "threshold": thresholds[label] if isinstance(thresholds, dict) else float(thresholds),
                "precision": precision[j],
                "recall": recall[j],
                "f1": f1[j],
                "roc_auc": roc_auc_score(y_true[:, j], y_prob[:, j]) if both_classes else np.nan,
                "pr_auc": average_precision_score(y_true[:, j], y_prob[:, j]) if both_classes else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("label")

    macro = table[["precision", "recall", "f1", "roc_auc", "pr_auc"]].mean(skipna=True)
    macro["support"] = table["support"].sum()
    macro["threshold"] = np.nan

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    flat_true, flat_prob = y_true.ravel(), y_prob.ravel()
    both_classes = 0 < flat_true.sum() < len(flat_true)
    micro = {
        "support": table["support"].sum(),
        "threshold": np.nan,
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "roc_auc": roc_auc_score(flat_true, flat_prob) if both_classes else np.nan,
        "pr_auc": average_precision_score(flat_true, flat_prob) if both_classes else np.nan,
    }
    table.loc["macro"] = macro
    table.loc["micro"] = micro
    table["support"] = table["support"].astype(int)
    return table
