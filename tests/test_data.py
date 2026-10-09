import pandas as pd

from toxicity import LABELS
from toxicity.data import (
    label_summary,
    load_test,
    load_train,
    multilabel_split,
    normalize_text,
    subsample,
)


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello\n\nworld \t again ") == "hello world again"


def test_load_train_keeps_columns_and_rows(raw_dir, comments):
    df = load_train(raw_dir / "train.csv")
    assert list(df.columns) == ["id", "comment_text", *LABELS]
    assert len(df) == len(comments)


def test_comment_that_reads_NA_stays_text(tmp_path, comments):
    comments.loc[0, "comment_text"] = "NA"
    comments.to_csv(tmp_path / "train.csv", index=False)
    assert load_train(tmp_path / "train.csv").loc[0, "comment_text"] == "NA"


def test_load_test_drops_unscored_rows(raw_dir):
    labels = pd.read_csv(raw_dir / "test_labels.csv")
    scored = int((labels[LABELS] != -1).all(axis=1).sum())
    df = load_test(raw_dir / "test.csv", raw_dir / "test_labels.csv")
    assert 0 < scored < len(labels)
    assert len(df) == scored
    assert (df[LABELS] >= 0).all().all()


def test_split_is_disjoint_complete_and_stratified(comments):
    train, val = multilabel_split(comments, 0.2, seed=1)
    assert len(train) + len(val) == len(comments)
    assert set(train["id"]).isdisjoint(val["id"])
    for label in LABELS:
        assert val[label].sum() >= 1, f"{label} missing from validation"
        assert abs(train[label].mean() - val[label].mean()) < 0.01


def test_split_is_reproducible(comments):
    first = multilabel_split(comments, 0.2, seed=3)[1]["id"].tolist()
    again = multilabel_split(comments, 0.2, seed=3)[1]["id"].tolist()
    other = multilabel_split(comments, 0.2, seed=4)[1]["id"].tolist()
    assert first == again
    assert first != other


def test_subsample_keeps_label_rates(comments):
    sample = subsample(comments, 420, seed=0)
    assert len(sample) == 420
    assert sample["id"].is_unique
    for label in LABELS:
        assert abs(sample[label].mean() - comments[label].mean()) < 0.01
    assert len(subsample(comments, 10_000, seed=0)) == len(comments)


def test_label_summary_counts(comments):
    summary = label_summary(comments)
    assert summary.loc["toxic", "count"] == comments["toxic"].sum()
    assert summary.loc["no_label", "count"] == (comments[LABELS].sum(axis=1) == 0).sum()
