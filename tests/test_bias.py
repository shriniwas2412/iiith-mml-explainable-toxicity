import numpy as np
import pandas as pd
import pytest

from toxicity.bias import (
    equality_differences,
    mean_scores_by_label,
    overall_rates,
    per_identity_table,
    score_shift,
)

FRAMES = {"nontoxic": ["I am {IDENTITY}", "You are a kind {IDENTITY}"],
          "toxic": ["I hate {IDENTITY}", "You are a awful {IDENTITY}"]}

# Model scores per (term, toxicity). "flagged" is the over-flagged term: its
# harmless sentences score above the other term's toxic ones.
SCORES = {
    ("flagged", "nontoxic"): 0.95, ("flagged", "toxic"): 0.99,
    ("fine", "nontoxic"): 0.10, ("fine", "toxic"): 0.90,
    ("person", "nontoxic"): 0.10, ("person", "toxic"): 0.80,
}


def _table():
    rows = []
    for term, term_set in [("flagged", "india"), ("fine", "dixon"), ("person", "neutral")]:
        for toxicity, frames in FRAMES.items():
            for frame in frames:
                rows.append({
                    "text": frame.replace("{IDENTITY}", term), "toxicity": toxicity, "template": "t",
                    "frame": frame, "identity": term, "axis": "x", "term_set": term_set,
                    "prob": SCORES[(term, toxicity)],
                })
    return pd.DataFrame(rows)


def test_per_identity_rates_and_aucs():
    df = _table()
    df = df[df["term_set"] != "neutral"].reset_index(drop=True)
    prob = df["prob"].to_numpy()
    table = per_identity_table(df, prob, (prob >= 0.5).astype(int)).set_index("identity")

    assert table.loc["flagged", ["fpr", "fnr"]].tolist() == [1.0, 0.0]
    assert table.loc["fine", ["fpr", "fnr"]].tolist() == [0.0, 0.0]
    assert table.loc["flagged", "n_nontoxic"] == 2 and table.loc["flagged", "n_toxic"] == 2
    # Inside its own group the flagged term is still ranked correctly...
    assert table.loc["flagged", "subgroup_auc"] == 1.0
    # ...but its harmless sentences outrank other terms' toxic ones.
    assert table.loc["flagged", "bpsn_auc"] == 0.0
    assert table.loc["flagged", "bnsp_auc"] == 1.0
    assert table.loc["fine", "bpsn_auc"] == 1.0
    assert table.loc["flagged", "mean_prob_nontoxic"] == pytest.approx(0.95)


def test_equality_differences_sum_gaps_to_the_overall_rate():
    df = _table()
    df = df[df["term_set"] != "neutral"].reset_index(drop=True)
    prob = df["prob"].to_numpy()
    pred = (prob >= 0.5).astype(int)
    overall = overall_rates(df, pred)
    assert overall == {"fpr": 0.5, "fnr": 0.0}
    gaps = equality_differences(per_identity_table(df, prob, pred), overall)
    assert gaps == {"fped": pytest.approx(1.0), "fned": pytest.approx(0.0)}


def test_score_shift_is_measured_against_the_neutral_fill():
    df = _table()
    shift = score_shift(df, df["prob"].to_numpy()).set_index("identity")
    assert "person" not in shift.index
    assert shift.loc["flagged", "shift_nontoxic"] == pytest.approx(0.85)
    assert shift.loc["fine", "shift_nontoxic"] == pytest.approx(0.0)
    assert shift.loc["fine", "shift_toxic"] == pytest.approx(0.10)


def test_score_shift_falls_back_to_the_frame_mean():
    df = _table()
    df = df[df["term_set"] != "neutral"].reset_index(drop=True)
    shift = score_shift(df, df["prob"].to_numpy()).set_index("identity")
    # Frame mean on nontoxic frames is (0.95 + 0.10) / 2.
    assert shift.loc["flagged", "shift_nontoxic"] == pytest.approx(0.425)
    assert shift.loc["fine", "shift_nontoxic"] == pytest.approx(-0.425)


def test_mean_scores_by_label_uses_one_toxicity_class():
    df = _table()
    probs = np.column_stack([df["prob"].to_numpy(), df["prob"].to_numpy() / 2])
    out = mean_scores_by_label(df, probs, ["toxic", "identity_hate"]).set_index("identity")
    assert out.loc["flagged", "toxic"] == pytest.approx(0.95)
    assert out.loc["flagged", "identity_hate"] == pytest.approx(0.475)
    assert len(out) == 3
