"""Exact linear explanations on the synthetic comments from conftest."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from toxicity import LABELS
from toxicity.baseline import build_baseline, build_model
from toxicity.explain import LinearExplainer, identity_weight_table, mask_text, mask_word

REPO = Path(__file__).resolve().parents[1]


def test_masking_keeps_first_letter_and_only_profane_words():
    assert mask_word("fuck") == "f***"
    assert mask_word("Fucking") == "F******"
    assert mask_word("gay") == "gay"
    assert mask_text("you fucking idiot") == "you f****** idiot"
    assert mask_text("passion class") == "passion class"


@pytest.mark.parametrize("use_char", [False, True])
def test_contributions_add_up_to_the_logit(comments, use_char):
    model = build_baseline(min_df=1, use_char=use_char, word_max_features=500, char_max_features=500)
    model.fit(comments["comment_text"], comments[LABELS].to_numpy())
    explainer = LinearExplainer(model)

    text = "You loser, what a stupid idiot!! number 3"
    for label in LABELS:
        r = explainer.explain_text(text, label)  # raises if the sum differs from the logit
        j = LABELS.index(label)
        assert r["probability"] == pytest.approx(model.predict_proba([text])[0, j], abs=1e-4)
        assert r["intercept"] + r["contribution"].sum() + r["other"] == pytest.approx(r["logit"], abs=1e-4)

    toxic = explainer.explain_text(text, "toxic")
    assert toxic["words"][toxic["contribution"].argmax()] in {"stupid", "idiot"}


def test_per_label_model_with_sample_weights(comments):
    model = build_model("lr", min_df=1)
    model.fit(comments["comment_text"], comments[LABELS].to_numpy(), clf__sample_weight=np.ones(len(comments)))
    explainer = LinearExplainer(model)
    top = explainer.top_features("toxic", 5)
    assert list(top["rank"]) == [1, 2, 3, 4, 5]
    assert top["coefficient"].is_monotonic_decreasing
    assert explainer.weight("toxic", "no such word") is None
    with pytest.raises(ValueError):
        LinearExplainer(build_model("nbsvm", min_df=1).fit(
            comments["comment_text"], comments[LABELS].to_numpy()))


def test_identity_weight_table_keeps_terms_in_vocabulary(comments):
    comments = comments.copy()
    comments.loc[1, "comment_text"] = "muslim neighbours muslim"
    comments.loc[2, "comment_text"] = "gay neighbours"
    model = build_baseline(min_df=1, use_char=False)
    model.fit(comments["comment_text"], comments[LABELS].to_numpy())
    terms = pd.DataFrame(
        {"term_set": ["dixon"] * 3, "axis": ["a"] * 3, "identity": ["muslim", "gay", "zzzz"], "pattern": ["muslim", "gay", "zzzz"]}
    )
    table = identity_weight_table({"m": LinearExplainer(model)}, terms, ["toxic", "identity_hate"])
    assert list(table["identity"]) == ["muslim", "gay"]
    assert table[["m_toxic", "m_identity_hate"]].notna().all().all()


def test_explain_script_writes_reports(tmp_path, comments, india_dir):
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    model = build_baseline(min_df=1, use_char=False)
    model.fit(comments["comment_text"], comments[LABELS].to_numpy())
    joblib.dump(model, run / "model.joblib")
    (run / "thresholds.json").write_text(json.dumps({label: 0.5 for label in LABELS}))
    out = tmp_path / "explain"

    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "16_explain_linear.py"), "--run", str(run),
         "--india", str(india_dir), "--errors", str(tmp_path / "none"), "--out", str(out)],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    for label in LABELS:
        top = pd.read_csv(out / "toy" / f"top_features_{label}.csv")
        assert len(top) == 20 and top["coefficient"].is_monotonic_decreasing
    assert (out / "toy" / "fig_top_features.png").exists()
    assert (out / "fig_identity_term_weights.png").exists()
    assert "toy_toxic" in pd.read_csv(out / "identity_term_weights.csv").columns
    assert "I am a gay person" in (out / "toy" / "local_examples.md").read_text()
    assert "<span" in (out / "toy" / "local_examples.html").read_text()
