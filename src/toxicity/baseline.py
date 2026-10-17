"""TF-IDF + logistic regression baseline (one binary classifier per label)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


def build_baseline(
    word_max_features: int = 100_000,
    char_max_features: int = 100_000,
    use_char: bool = True,
    min_df: int = 2,
    C: float = 4.0,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """Word (1-2 gram) and optional character (3-5 gram) TF-IDF into one-vs-rest LR.

    `class_weight="balanced"` reweights the loss per label instead of resampling.
    """
    shared = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True, min_df=min_df, dtype=np.float32)
    features = [
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=word_max_features, **shared))
    ]
    if use_char:
        features.append(
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=char_max_features, **shared))
        )
    classifier = OneVsRestClassifier(
        LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", random_state=seed)
    )
    return Pipeline([("features", FeatureUnion(features)), ("clf", classifier)])
