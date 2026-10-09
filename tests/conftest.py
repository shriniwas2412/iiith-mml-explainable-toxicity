"""Small synthetic stand-ins for the real files, with the same columns."""

from __future__ import annotations

import pandas as pd
import pytest

from toxicity import LABELS

# Row i gets a label when i is a multiple of the number: nested, like the real
# data where most labelled comments are also `toxic`.
_EVERY = {"toxic": 7, "obscene": 14, "insult": 21, "severe_toxic": 70, "identity_hate": 84, "threat": 105}
_WORDS = {
    "toxic": "stupid idiot",
    "obscene": "damn crap",
    "insult": "you loser",
    "severe_toxic": "worthless scum",
    "identity_hate": "hate your kind",
    "threat": "i will hurt you",
}


def make_comments(n: int, start: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(start, start + n):
        flags = {label: int(i % every == 0) for label, every in _EVERY.items()}
        words = [_WORDS[label] for label in LABELS if flags[label]]
        text = " ".join(words) if words else "thanks for the helpful edit to this article"
        rows.append({"id": f"id{i:05d}", "comment_text": f"{text} number {i % 13}", **flags})
    return pd.DataFrame(rows)[["id", "comment_text", *LABELS]]


@pytest.fixture
def comments() -> pd.DataFrame:
    return make_comments(840)


@pytest.fixture
def raw_dir(tmp_path, comments):
    """A data/raw folder: train.csv, test.csv and test_labels.csv with unscored rows."""
    raw = tmp_path / "raw"
    raw.mkdir()
    comments.to_csv(raw / "train.csv", index=False)

    test = make_comments(420, start=1000)
    labels = test[["id", *LABELS]].copy()
    labels.loc[labels.index % 3 == 0, LABELS] = -1
    test[["id", "comment_text"]].to_csv(raw / "test.csv", index=False)
    labels.to_csv(raw / "test_labels.csv", index=False)
    return raw


MINI_TERMS = ["african", "african american", "gay", "muslim"]


@pytest.fixture
def dixon_file(tmp_path):
    """A few rows in the format of the Dixon et al. en_sentence_templates.csv."""
    rows = []
    for term in MINI_TERMS:
        rows += [
            ("am_hate_adj", "nontoxic", f"I am {term}"),
            ("am_hate_adj", "toxic", f"I hate {term}"),
            ("you_are_adj", "nontoxic", f"You are a kind {term}"),
            ("you_are_adj", "toxic", f"You are a awful {term}"),
            ("being_adj", "nontoxic", f"Being {term} is great"),
            ("being_adj", "toxic", f"Being {term} is awful"),
            ("name_adj", "nontoxic", f"Asha is a kind {term}"),
            ("name_adj", "nontoxic", f"Joe is a kind {term}"),
            ("name_adj", "toxic", f"Asha is a awful {term}"),
            ("name_adj", "toxic", f"Joe is a awful {term}"),
        ]
    rows += [
        ("you_occupation", "nontoxic", "You are a kind actor"),
        ("you_occupation", "toxic", "You are a awful actor"),
    ]
    path = tmp_path / "en_sentence_templates.csv"
    pd.DataFrame(rows, columns=["template", "toxicity", "phrase"]).to_csv(path, index=False)
    return path


@pytest.fixture
def india_dir(tmp_path):
    """Term lists with the two header spellings used in the released files."""
    folder = tmp_path / "india"
    folder.mkdir()
    (folder / "caste_idterms.tsv").write_text("identity term\nbrahmin\ndalit\nkhsatirya\n")
    (folder / "religion_idterms.tsv").write_text("identity term\nhindu\nmuslim\n")
    (folder / "region_idterms.tsv").write_text("identity_terms\nbihari\nuttar pradeshi\n")
    return folder
