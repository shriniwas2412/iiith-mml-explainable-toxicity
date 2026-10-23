from sklearn.metrics import roc_auc_score

from toxicity import LABELS
from toxicity.baseline import build_baseline
from toxicity.data import multilabel_split


def test_baseline_learns_the_toy_labels(comments):
    train, val = multilabel_split(comments, 0.25, seed=0)
    model = build_baseline(min_df=1)
    model.fit(train["comment_text"], train[LABELS].to_numpy())
    prob = model.predict_proba(val["comment_text"])

    assert prob.shape == (len(val), len(LABELS))
    assert ((prob >= 0) & (prob <= 1)).all()
    assert roc_auc_score(val["toxic"], prob[:, LABELS.index("toxic")]) > 0.95


def test_baseline_without_char_features(comments):
    model = build_baseline(min_df=1, use_char=False)
    model.fit(comments["comment_text"], comments[LABELS].to_numpy())
    assert [name for name, _ in model.named_steps["features"].transformer_list] == ["word"]


def test_every_model_kind_gives_six_probabilities(comments):
    import numpy as np
    import pytest

    from toxicity.baseline import MODEL_KINDS, build_model

    train, val = multilabel_split(comments, 0.25, seed=0)
    for kind in MODEL_KINDS:
        model = build_model(kind, min_df=1)
        weights = np.ones(len(train))
        model.fit(train["comment_text"], train[LABELS].to_numpy(), clf__sample_weight=weights)
        prob = model.predict_proba(val["comment_text"])
        assert prob.shape == (len(val), len(LABELS)), kind
        assert roc_auc_score(val["toxic"], prob[:, LABELS.index("toxic")]) > 0.9, kind
    with pytest.raises(ValueError):
        build_model("forest")
