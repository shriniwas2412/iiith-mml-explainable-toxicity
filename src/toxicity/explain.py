"""Exact explanations for the linear (TF-IDF + logistic regression) models.

A one-vs-rest linear model scores a comment as

    logit = intercept + sum over features of (tf-idf value x coefficient)

so each feature's contribution to the logit is exact: no sampling and no
approximation, unlike LIME or SHAP on a black box. The contributions plus the
intercept add back up to the model's decision function (checked in `explain_text`).

Local explanations are reported per original word. A word feature goes to its
word. A bigram is split equally between its two words. A character n-gram
(`char_wb`, if the model has char features) goes to the word it was cut from.
When a feature occurs more than once in a comment, its contribution is split
equally between the occurrences. Features that cannot be placed on a word
(punctuation-only n-grams) are summed into "(other)".
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from .config import LABELS

# Whole-word profanity and slurs, with a few suffixes. Used only to mask text in
# reports: it is a display filter, not a detector.
_STEMS = (
    "fuck|shit|bitch|bastard|asshole|arse|cunt|crap|piss|cock|pussy|slut|whore|nigg\\w*|fag|faggot|"
    "retard|dumbass|twat|wank|kike|chink|spic|bollocks|motherfuck\\w*"
)
_SUFFIX = "(?:ing|er|ers|s|y|ty|ed|es|head|face)?"
_PROFANE = re.compile(rf"^(?:{_STEMS}){_SUFFIX}$|^dicks?$")
_WORD = re.compile(r"\w+", re.UNICODE)


def mask_word(word: str) -> str:
    """'fucking' -> 'f******'. Only profane words are masked; the first letter stays."""
    return word[0] + "*" * (len(word) - 1) if len(word) > 1 and _PROFANE.match(word.lower()) else word


def mask_text(text: str) -> str:
    """Mask every profane word in `text` (feature names such as 'fuck you' included)."""
    return _WORD.sub(lambda m: mask_word(m.group()), text)


class LinearExplainer:
    """Coefficients and exact contributions of a fitted TF-IDF + per-label linear pipeline."""

    def __init__(self, pipeline):
        self.features = pipeline.named_steps["features"]
        clf = pipeline.named_steps["clf"]
        models = getattr(clf, "estimators_", None)
        if models is None or len(models) != len(LABELS) or not hasattr(models[0], "coef_"):
            raise ValueError("Needs a linear classifier per label (OneVsRestClassifier or PerLabel of LogisticRegression).")
        if getattr(clf, "nb_ratio", False):
            raise ValueError("NB-scaled features (nbsvm) are not supported.")
        self.coef = np.vstack([m.coef_.ravel() for m in models])  # (6, n_features)
        self.intercept = np.array([m.intercept_[0] for m in models])
        self.names = [str(n) for n in self.features.get_feature_names_out()]  # "word__foo bar", "char__ foo"
        self.index = {n: i for i, n in enumerate(self.names)}
        self.vectorizers = dict(self.features.transformer_list)

    def weight(self, label: str, feature: str, kind: str = "word") -> float | None:
        """Coefficient of one feature (e.g. 'gay' or 'african american'), or None if not in the vocabulary."""
        i = self.index.get(f"{kind}__{feature}")
        return None if i is None else float(self.coef[LABELS.index(label), i])

    def top_features(self, label: str, n: int = 20) -> pd.DataFrame:
        """The n features with the largest positive coefficient for `label`."""
        w = self.coef[LABELS.index(label)]
        order = np.argsort(-w, kind="stable")[:n]
        kinds, feats = zip(*(self.names[i].split("__", 1) for i in order))
        return pd.DataFrame({"rank": np.arange(1, len(order) + 1), "feature": feats, "kind": kinds, "coefficient": w[order]})

    def _units(self, text: str) -> list[str]:
        """Original words of the comment (letters and digits), as displayed."""
        return _WORD.findall(text)

    def _place(self, text: str, units: list[str]) -> dict[str, list[list[int]]]:
        """For each feature name, the occurrences of it, each a list of unit positions it covers."""
        places: dict[str, list[list[int]]] = {}
        for kind, vec in self.vectorizers.items():
            pre = vec.build_preprocessor()(text)
            if vec.analyzer == "word":
                tokens = vec.build_tokenizer()(pre)
                # Unit position of each kept token: tokens are a subset of the \w+ units.
                pos, j = [], 0
                lowered = [vec.build_preprocessor()(u) for u in units]
                for tok in tokens:
                    while j < len(units) and lowered[j] != tok:
                        j += 1
                    pos.append(j if j < len(units) else -1)
                    j += 1
                lo, hi = vec.ngram_range
                for n in range(lo, hi + 1):
                    for s in range(len(tokens) - n + 1):
                        gram = " ".join(tokens[s : s + n])
                        cover = [p for p in pos[s : s + n] if p >= 0]
                        places.setdefault(f"{kind}__{gram}", []).append(cover)
            else:  # char / char_wb: n-grams are cut inside whitespace-separated words
                words = pre.split()
                lowered = [vec.build_preprocessor()(u) for u in units]
                cursor = 0
                for word in words:
                    # Units sitting inside this whitespace word, consumed in order.
                    inside = []
                    while cursor < len(units) and lowered[cursor] in word:
                        inside.append(cursor)
                        cursor += 1
                    for gram in vec.build_analyzer()(word):
                        places.setdefault(f"{kind}__{gram}", []).append(inside)
        return places

    def explain_text(self, text: str, label: str) -> dict:
        """Exact attribution of one comment for one label.

        Returns the words with their contributions to the logit, the intercept,
        the logit and probability, and the leftover not placed on any word.
        """
        j = LABELS.index(label)
        row = self.features.transform([text]).tocsr()
        units = self._units(text)
        contrib = np.zeros(len(units))
        other = 0.0
        places = self._place(text, units)
        for i, value in zip(row.indices, row.data):
            c = float(value) * self.coef[j, i]
            occurrences = places.get(self.names[i], [])
            if not occurrences:
                other += c
                continue
            for cover in occurrences:
                if not cover:
                    other += c / len(occurrences)
                    continue
                for k in cover:
                    contrib[k] += c / len(occurrences) / len(cover)
        logit = float(self.intercept[j] + (row @ self.coef[j])[0])
        total = self.intercept[j] + contrib.sum() + other
        if not np.isclose(total, logit, atol=1e-4):
            raise AssertionError(f"Contributions {total:.5f} do not add up to the logit {logit:.5f}")
        return {
            "words": units,
            "contribution": contrib,
            "other": other,
            "intercept": float(self.intercept[j]),
            "logit": logit,
            "probability": float(1 / (1 + np.exp(-logit))),
        }


def identity_weight_table(explainers: dict[str, LinearExplainer], terms: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Coefficient of each identity term that is a word feature in the vocabulary, per run and label.

    One row per (term_set, axis, identity). Terms missing from a run's vocabulary give NaN.
    """
    rows = []
    for (term_set, axis, identity), group in terms.groupby(["term_set", "axis", "identity"], sort=False):
        patterns = [p.lower() for p in group["pattern"]]
        row = {"term_set": term_set, "axis": axis, "identity": identity}
        for run, explainer in explainers.items():
            for label in labels:
                found = [w for p in patterns if (w := explainer.weight(label, p)) is not None]
                row[f"{run}_{label}"] = max(found) if found else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    value_cols = [f"{run}_{label}" for run in explainers for label in labels]
    return table.dropna(subset=value_cols, how="all").reset_index(drop=True)
