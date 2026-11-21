"""Run the error analysis script on the synthetic files from conftest."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from toxicity import LABELS
from toxicity.predictions import save_predictions

REPO = Path(__file__).resolve().parents[1]


def run_script(*args) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "13_error_analysis.py"), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_error_analysis_two_toy_runs(tmp_path, comments, india_dir):
    # Two clean comments mention identity terms (the India list has "muslim" and "hindu").
    comments.loc[1, "comment_text"] = "muslim neighbours"
    comments.loc[2, "comment_text"] = "hindu temple"
    test_csv = tmp_path / "test.csv"
    comments.to_csv(test_csv, index=False)

    rng = np.random.default_rng(0)
    y_prob = rng.uniform(size=(len(comments), len(LABELS)))  # random scores give both FP and FN
    runs = []
    for name, cutoff in [("toy_a", 0.5), ("toy_b", 0.3)]:
        run = tmp_path / "runs" / name
        save_predictions(run / "test_predictions.csv", comments, y_prob)
        (run / "thresholds.json").write_text(json.dumps({label: cutoff for label in LABELS}))
        runs.append(run)

    out_dir = tmp_path / "error_analysis"
    stdout = run_script("--run", *runs, "--test", test_csv, "--india", india_dir, "--out", out_dir)
    assert "identity_over_none" in stdout

    for run in runs:
        folder = out_dir / run.name
        for label in LABELS:
            errors = pd.read_csv(folder / f"{label}_errors.csv")
            assert len(errors) <= 40
            assert errors["comment_text"].str.len().max() <= 300
            assert set(errors["error_type"]) <= {"false_positive", "false_negative"}

        summary = pd.read_csv(folder / "summary.csv")
        assert list(summary["label"]) == [*LABELS, "all_labels"]
        per_label = summary[summary["label"].isin(LABELS)]
        # Every comment is counted once per label.
        assert (per_label[["tp", "fp", "fn", "tn"]].sum(axis=1) == len(comments)).all()
        assert summary.loc[summary["label"] == "all_labels", "tp"].item() == per_label["tp"].sum()

    compare = pd.read_csv(out_dir / "identity_fp_real_comments.csv")
    assert set(compare["run"]) == {"toy_a", "toy_b"}
    assert set(compare["group"]) == {"mentions_identity_term", "no_identity_term"}
    assert compare["fp_rate"].between(0, 1).all()
    identity_comments = compare.loc[compare["group"] == "mentions_identity_term", "comments"].iloc[0]
    assert identity_comments >= 2
    assert (out_dir / "fig_identity_fp_real.png").exists()
