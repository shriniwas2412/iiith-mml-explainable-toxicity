"""TF-IDF + logistic regression baseline (one binary classifier per label)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


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


class PerLabel(BaseEstimator, ClassifierMixin):
    """Fit one copy of `estimator` per label; optionally on NB-weighted features.

    Unlike OneVsRestClassifier this passes `sample_weight` to every copy, which
    the identity re-weighting experiment needs. With `nb_ratio=True` each label
    gets the NB-SVM trick (Wang & Manning, 2012): features are scaled by the
    log-count ratio r = log((p / |p|_1) / (q / |q|_1)) before the classifier.
    """

    def __init__(self, estimator, nb_ratio: bool = False, alpha: float = 1.0):
        self.estimator = estimator
        self.nb_ratio = nb_ratio
        self.alpha = alpha

    def _scale(self, X, j):
        return X.multiply(self.ratios_[j]).tocsr() if self.nb_ratio else X

    def fit(self, X, Y, sample_weight=None):
        Y = np.asarray(Y)
        self.estimators_, self.ratios_ = [], []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if self.nb_ratio:
                p = self.alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
                q = self.alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
                self.ratios_.append(np.log((p / p.sum()) / (q / q.sum())).astype(np.float32))
            model = clone(self.estimator)
            model.fit(self._scale(X, j), y, sample_weight=sample_weight)
            self.estimators_.append(model)
        return self

    def predict_proba(self, X):
        return np.column_stack(
            [m.predict_proba(self._scale(X, j))[:, 1] for j, m in enumerate(self.estimators_)]
        )


MODEL_KINDS = ("lr", "svm", "cnb", "nbsvm")


def build_model(
    kind: str = "lr",
    word_max_features: int = 100_000,
    char_max_features: int = 100_000,
    use_char: bool = False,
    min_df: int = 2,
    C: float = 4.0,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """TF-IDF features into one of four per-label classifiers, for model comparison.

    lr: logistic regression · svm: linear SVM with sigmoid-calibrated probabilities
    · cnb: complement naive Bayes · nbsvm: logistic regression on NB-scaled features.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind '{kind}'. Use one of {MODEL_KINDS}.")
    features = build_baseline(word_max_features, char_max_features, use_char, min_df, C, class_weight, seed)
    features = features.named_steps["features"]
    if kind == "lr":
        base = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", random_state=seed)
    elif kind == "svm":
        base = CalibratedClassifierCV(
            LinearSVC(C=C / 4, class_weight=class_weight, random_state=seed), cv=3, method="sigmoid"
        )
    elif kind == "cnb":
        base = ComplementNB(alpha=1.0)
    else:
        base = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", random_state=seed)
    return Pipeline([("features", features), ("clf", PerLabel(base, nb_ratio=kind == "nbsvm"))])
