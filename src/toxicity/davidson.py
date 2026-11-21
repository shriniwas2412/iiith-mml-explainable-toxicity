"""Davidson et al. (2017) feature set and model, adapted to Wikipedia comments.

Davidson, Warmsley, Macy and Weber, "Automated Hate Speech Detection and the
Problem of Offensive Language", ICWSM 2017. The paper stacks these features:
word 1-3-gram TF-IDF (lower-cased, Porter-stemmed), POS-tag 1-3-gram TF-IDF,
Flesch-Kincaid grade and reading ease, VADER sentiment, counts of URLs,
mentions and hashtags, and character, word and syllable counts. A logistic
regression with L1 penalty first drops features, then a logistic regression with
L2 penalty is fitted on what is left.

Deviations from the paper:
  * Data: Wikipedia talk-page comments, not tweets. There are no mentions or
    hashtags, so the "mention" count is the number of User:/User talk: links
    and the "hashtag" slot is the number of wiki-markup tokens ([[ ]], {{ }},
    ==). URLs and IP addresses are counted as they are.
  * Labels: six binary labels instead of three exclusive classes (hate /
    offensive / neither), so the selector and the classifier are fitted once
    per label (one-vs-rest) instead of once for the 3-way problem.
  * Class weights are "balanced" in both stages, as in the other baselines here;
    the paper used the default weights of its toolkit.
  * No retweet feature (not applicable) and no manual tuning of the grid.
  * Syllables come from a vowel-group heuristic, as in the paper's own code, so
    no pronunciation dictionary is needed.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

_TOKEN = re.compile(r"\w+|[^\w\s]")
_WORD = re.compile(r"[a-z]+")
_STEM = PorterStemmer()
_VOWELS = re.compile(r"[aeiouy]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER_LINK = re.compile(r"(?i)\buser(?:[ _]talk)?:")
_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|={2,}")

# The paper's own vocabulary limits.
WORD_MAX_FEATURES = 10_000
POS_MAX_FEATURES = 5_000
MAX_TAGGED_TOKENS = 200  # POS-tag only the start of very long comments, for speed


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEM.stem(word)


def stem_tokens(text: str) -> list[str]:
    """Lower-case, keep alphabetic words, drop stop words, Porter-stem."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]


def count_syllables(word: str) -> int:
    """Number of vowel groups, at least 1, minus a silent final 'e'."""
    n = len(_VOWELS.findall(word.lower()))
    if word.lower().endswith("e") and n > 1 and not word.lower().endswith("le"):
        n -= 1
    return max(n, 1)


def readability(text: str) -> tuple[float, float, int, int, int]:
    """Flesch-Kincaid grade, Flesch reading ease, then syllable, word and sentence counts."""
    words = _WORD.findall(text.lower())
    n_words = max(len(words), 1)
    n_sent = max(len(re.findall(r"[.!?]+", text)), 1)
    n_syll = sum(count_syllables(w) for w in words)
    per_word, per_sent = n_syll / n_words, n_words / n_sent
    grade = 0.39 * per_sent + 11.8 * per_word - 15.59
    ease = 206.835 - 1.015 * per_sent - 84.6 * per_word
    return grade, ease, n_syll, len(words), n_sent


class PosTags(BaseEstimator, TransformerMixin):
    """Replace each comment by the string of its NLTK part-of-speech tags."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            tokens = _TOKEN.findall(str(text))[:MAX_TAGGED_TOKENS]
            out.append(" ".join(tag for _, tag in pos_tag(tokens)))
        return out


class TextStats(BaseEstimator, TransformerMixin):
    """Readability, VADER sentiment and count features, one dense row per comment."""

    NAMES = [
        "fk_grade", "reading_ease", "vader_neg", "vader_neu", "vader_pos", "vader_compound",
        "n_urls", "n_user_links", "n_ips", "n_markup", "n_chars", "n_words", "n_syllables", "n_sentences",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        vader = SentimentIntensityAnalyzer()
        rows = []
        for text in X:
            text = str(text)
            grade, ease, n_syll, n_words, n_sent = readability(text)
            s = vader.polarity_scores(text[:5000])
            rows.append([
                grade, ease, s["neg"], s["neu"], s["pos"], s["compound"],
                len(_URL.findall(text)), len(_USER_LINK.findall(text)), len(_IP.findall(text)),
                len(_MARKUP.findall(text)), len(text), n_words, n_syll, n_sent,
            ])
        out = np.asarray(rows, dtype=np.float64)
        out[:, 6:] = np.log1p(out[:, 6:])  # counts are heavy-tailed
        out[:, :2] = np.clip(out[:, :2], -50, 150)  # one-word comments give extreme grades
        return out


class SelectThenFit(BaseEstimator, ClassifierMixin):
    """Per label: an L1 logistic regression picks features, an L2 one is fitted on them."""

    def __init__(self, l1_C: float = 0.1, l2_C: float = 1.0, class_weight: str | None = "balanced", seed: int = 42):
        self.l1_C = l1_C
        self.l2_C = l2_C
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, Y):
        Y = np.asarray(Y)
        self.selectors_, self.estimators_ = [], []
        for j in range(Y.shape[1]):
            l1 = LogisticRegression(
                penalty="l1", C=self.l1_C, class_weight=self.class_weight, solver="liblinear", random_state=self.seed
            )
            selector = SelectFromModel(l1).fit(X, Y[:, j])
            if not selector.get_support().any():  # a label so rare that L1 zeroes everything
                selector = SelectFromModel(l1, threshold=-np.inf).fit(X, Y[:, j])
            l2 = LogisticRegression(
                C=self.l2_C, class_weight=self.class_weight, solver="liblinear", random_state=self.seed
            )
            l2.fit(selector.transform(X), Y[:, j])
            self.selectors_.append(selector)
            self.estimators_.append(l2)
        return self

    def n_selected(self) -> list[int]:
        return [int(s.get_support().sum()) for s in self.selectors_]

    def predict_proba(self, X):
        return np.column_stack(
            [m.predict_proba(s.transform(X))[:, 1] for s, m in zip(self.selectors_, self.estimators_)]
        )


def build_davidson(
    l1_C: float = 0.1,
    l2_C: float = 1.0,
    class_weight: str | None = "balanced",
    min_df: int = 5,
    seed: int = 42,
) -> Pipeline:
    """Davidson features (shared by all labels) into per-label L1 selection and L2 LR."""
    shared = dict(sublinear_tf=False, min_df=min_df, max_df=0.75, dtype=np.float32)
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(tokenizer=stem_tokens, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 3), max_features=WORD_MAX_FEATURES, **shared)),
            ("pos", Pipeline([
                ("tags", PosTags()),
                ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=r"\S+", ngram_range=(1, 3),
                                          max_features=POS_MAX_FEATURES, **shared)),
            ])),
            ("stats", Pipeline([("stats", TextStats()), ("scale", StandardScaler())])),
        ]
    )
    return Pipeline([("features", features), ("clf", SelectThenFit(l1_C, l2_C, class_weight, seed))])
