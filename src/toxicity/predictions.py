"""One file format for model predictions, shared by every model and script.

A predictions CSV has: id, the six true label columns, and prob_<label> columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, LABELS


def prob_columns(labels: list[str] = LABELS) -> list[str]:
    return [f"prob_{label}" for label in labels]


def save_predictions(
    path: str | Path, df: pd.DataFrame, y_prob: np.ndarray, labels: list[str] = LABELS
) -> None:
    """Write ids, true labels and predicted probabilities for the rows of `df`."""
    out = df[[ID_COL, *labels]].copy()
    out[prob_columns(labels)] = y_prob
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_predictions(
    path: str | Path, labels: list[str] = LABELS
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Return (ids, y_true, y_prob) from a predictions CSV."""
    df = pd.read_csv(path)
    return df[ID_COL], df[labels].to_numpy(), df[prob_columns(labels)].to_numpy()
