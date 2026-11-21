import numpy as np
from sklearn.metrics import roc_auc_score

from toxicity import LABELS
from toxicity.data import multilabel_split
from toxicity.davidson import TextStats, build_davidson, count_syllables, readability


def test_syllables_and_readability():
    assert count_syllables("cat") == 1
    assert count_syllables("table") == 2
    assert count_syllables("make") == 1
    grade, ease, n_syll, n_words, n_sent = readability("The cat sat. It was happy!")
    assert (n_words, n_sent) == (6, 2)
    assert n_syll >= n_words
    assert ease > 90  # short monosyllabic sentences are easy to read


def test_text_stats_counts_wiki_artifacts():
    text = "see http://example.com and User:Bob from 192.168.0.1 [[link]]"
    stats = TextStats().transform([text, "hello"])
    names = TextStats.NAMES
    assert stats.shape == (2, len(names))
    assert np.expm1(stats[0, names.index("n_urls")]) == 1
    assert np.expm1(stats[0, names.index("n_user_links")]) == 1
    assert np.expm1(stats[0, names.index("n_ips")]) == 1
    assert np.expm1(stats[1, names.index("n_urls")]) == 0


def test_davidson_model_learns_the_toy_labels(comments):
    train, val = multilabel_split(comments, 0.25, seed=0)
    model = build_davidson(min_df=1)
    model.fit(train["comment_text"], train[LABELS].to_numpy())
    prob = model.predict_proba(val["comment_text"])

    assert prob.shape == (len(val), len(LABELS))
    assert ((prob >= 0) & (prob <= 1)).all()
    assert roc_auc_score(val["toxic"], prob[:, LABELS.index("toxic")]) > 0.9
    assert all(n > 0 for n in model.named_steps["clf"].n_selected())
