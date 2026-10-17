"""Run the command-line scripts end to end on the synthetic files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from toxicity import LABELS

REPO = Path(__file__).resolve().parents[1]


def run(script: str, *args) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_baseline_pipeline_and_bias_audit(tmp_path, raw_dir, dixon_file, india_dir):
    processed, run_dir = tmp_path / "processed", tmp_path / "runs" / "baseline"

    run("02_eda.py", "--train", raw_dir / "train.csv", "--out", tmp_path / "eda")
    assert (tmp_path / "eda" / "label_counts.csv").exists()

    run("03_make_splits.py", "--raw", raw_dir, "--out", processed, "--val-size", 0.25, "--subset", 300)
    stats = pd.read_csv(processed / "split_stats.csv", index_col=0)
    assert stats.loc["train", "rows"] + stats.loc["val", "rows"] == 840
    assert stats.loc["train_subset_300", "rows"] == 300
    assert stats.loc["test", "rows"] == 280  # a third of the 420 test rows are unscored

    run("04_train_baseline.py", "--train", processed / "train.csv", "--val", processed / "val.csv",
        "--test", processed / "test.csv", "--out", run_dir)
    out = run("06_tune_thresholds.py", "--run", run_dir)
    assert "Macro F1 change from tuning" in out
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    assert list(thresholds) == LABELS
    tuned = pd.read_csv(run_dir / "test_metrics_tuned.csv", index_col=0)
    assert list(tuned.index) == [*LABELS, "macro", "micro"]

    templates = tmp_path / "templates.csv"
    run("08_build_templates.py", "--dixon", dixon_file, "--india-dir", india_dir, "--out", templates)
    bias = tmp_path / "bias"
    run("09_bias_audit.py", "--model", f"baseline:{run_dir / 'model.joblib'}", "--templates", templates,
        "--thresholds", run_dir / "thresholds.json", "--out", bias)

    summary = json.loads((bias / "summary.json").read_text())
    assert set(summary["term_sets"]) == {"dixon", "india", "neutral", "control"}
    assert {"fped", "fned", "fpr", "fnr"} <= set(summary["term_sets"]["india"])
    per_identity = pd.read_csv(bias / "per_identity_toxic.csv")
    assert len(per_identity) == 4 + 7  # terms in the two toy lists
    by_label = pd.read_csv(bias / "mean_scores_nontoxic_by_label.csv")
    assert set(LABELS) <= set(by_label.columns)
    assert len(pd.read_csv(bias / "scores.csv")) == len(pd.read_csv(templates))
