import numpy as np
import pytest
from sklearn.metrics import f1_score

from toxicity.metrics import apply_thresholds, evaluate, tune_thresholds
from toxicity.predictions import load_predictions, save_predictions

TWO = ["a", "b"]


def _random_case(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, 2)) < [0.3, 0.05]).astype(int)
    y_prob = np.clip(y_true * 0.35 + rng.random((n, 2)) * 0.65, 0, 1)
    return y_true, y_prob


def test_tuned_threshold_is_the_best_f1_cutoff():
    y_true, y_prob = _random_case()
    thresholds = tune_thresholds(y_true, y_prob, TWO)
    for j, label in enumerate(TWO):
        tuned = f1_score(y_true[:, j], y_prob[:, j] >= thresholds[label])
        best = max(f1_score(y_true[:, j], y_prob[:, j] >= t) for t in np.unique(y_prob[:, j]))
        assert tuned == pytest.approx(best)


def test_label_without_positives_keeps_default_threshold():
    y_true, y_prob = _random_case()
    y_true[:, 1] = 0
    assert tune_thresholds(y_true, y_prob, TWO, default=0.5)["b"] == 0.5


def test_apply_thresholds_flat_and_per_label():
    y_prob = np.array([[0.4, 0.4], [0.6, 0.6]])
    assert apply_thresholds(y_prob, 0.5, TWO).tolist() == [[0, 0], [1, 1]]
    assert apply_thresholds(y_prob, {"a": 0.3, "b": 0.7}, TWO).tolist() == [[1, 0], [1, 0]]


def test_evaluate_known_values():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    y_prob = np.array([[0.9, 0.2], [0.4, 0.1], [0.6, 0.8], [0.1, 0.3]])
    table = evaluate(y_true, y_prob, 0.5, TWO)
    # Label a: flagged rows 0 and 2, true rows 0 and 1 -> precision 0.5, recall 0.5.
    assert table.loc["a", ["precision", "recall", "f1"]].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert table.loc["a", "roc_auc"] == pytest.approx(0.75)
    assert table.loc["b", ["precision", "recall", "f1", "roc_auc"]].tolist() == pytest.approx([1, 1, 1, 1])
    assert table.loc["macro", "f1"] == pytest.approx(0.75)
    # Micro: 2 true positives, 1 false positive, 1 false negative.
    assert table.loc["micro", ["precision", "recall"]].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert table.loc["a", "support"] == 2


def test_evaluate_gives_nan_auc_for_single_class_label():
    y_true = np.array([[1, 0], [0, 0], [1, 0]])
    y_prob = np.array([[0.9, 0.2], [0.2, 0.1], [0.7, 0.4]])
    table = evaluate(y_true, y_prob, 0.5, TWO)
    assert np.isnan(table.loc["b", "roc_auc"])
    assert table.loc["macro", "roc_auc"] == pytest.approx(table.loc["a", "roc_auc"])


def test_predictions_round_trip(tmp_path, comments):
    from toxicity import LABELS

    prob = np.random.default_rng(0).random((len(comments), len(LABELS)))
    save_predictions(tmp_path / "run" / "val_predictions.csv", comments, prob)
    ids, y_true, y_prob = load_predictions(tmp_path / "run" / "val_predictions.csv")
    assert ids.tolist() == comments["id"].tolist()
    assert (y_true == comments[LABELS].to_numpy()).all()
    assert y_prob == pytest.approx(prob)
