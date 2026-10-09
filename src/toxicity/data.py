"""Loading, cleaning and splitting the Jigsaw Toxic Comment data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .config import ID_COL, LABELS, TEXT_COL

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Collapse newlines and repeated whitespace. Casing is left to the tokenizer."""
    return _WHITESPACE.sub(" ", str(text)).strip()


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _read_csv(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False: a comment that is literally "NA" or "null" stays text.
    return pd.read_csv(path, keep_default_na=False)


def read_split(path: str | Path) -> pd.DataFrame:
    """Read a file written by scripts/03_make_splits.py."""
    df = _read_csv(path)
    _require_columns(df, [ID_COL, TEXT_COL, *LABELS], str(path))
    return df


def load_train(path: str | Path) -> pd.DataFrame:
    """Read train.csv (id, comment_text and the six label columns)."""
    df = _read_csv(path)
    _require_columns(df, [ID_COL, TEXT_COL, *LABELS], "train file")
    df[TEXT_COL] = df[TEXT_COL].map(normalize_text)
    return df[[ID_COL, TEXT_COL, *LABELS]]


def load_test(test_path: str | Path, labels_path: str | Path) -> pd.DataFrame:
    """Join test comments with the released labels and keep only scored rows.

    Kaggle marked the rows it did not score with -1 in every label column.
    """
    text = _read_csv(test_path)
    labels = _read_csv(labels_path)
    _require_columns(text, [ID_COL, TEXT_COL], "test file")
    _require_columns(labels, [ID_COL, *LABELS], "test labels file")
    df = text.merge(labels, on=ID_COL, how="inner", validate="one_to_one")
    scored = (df[LABELS] != -1).all(axis=1)
    df = df.loc[scored].reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].map(normalize_text)
    return df[[ID_COL, TEXT_COL, *LABELS]]


def multilabel_split(
    df: pd.DataFrame, held_out: float | int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so each label keeps roughly the same rate on both sides.

    `held_out` is a fraction or an absolute row count. Uses iterative
    stratification (Sechidis et al., 2011), which matters for labels as rare
    as `threat`.
    """
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=held_out, random_state=seed
    )
    y = df[LABELS].to_numpy()
    keep_idx, held_idx = next(splitter.split(np.zeros((len(df), 1)), y))
    return (
        df.iloc[np.sort(keep_idx)].reset_index(drop=True),
        df.iloc[np.sort(held_idx)].reset_index(drop=True),
    )


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw `n` rows while keeping the label rates of `df`."""
    if n >= len(df):
        return df.reset_index(drop=True)
    _, sample = multilabel_split(df, held_out=n, seed=seed)
    return sample


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of positives per label, plus rows with no label at all."""
    counts = df[LABELS].sum().astype(int)
    out = pd.DataFrame({"count": counts, "share": counts / len(df)})
    no_label = int((df[LABELS].sum(axis=1) == 0).sum())
    out.loc["no_label"] = [no_label, no_label / len(df)]
    out["count"] = out["count"].astype(int)
    return out
