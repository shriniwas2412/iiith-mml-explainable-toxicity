"""Tune one decision threshold per label on validation, then score the test set.

Works on any run folder that has val_predictions.csv and test_predictions.csv.
Thresholds are chosen on validation only; the test set is never used for tuning.

    python scripts/06_tune_thresholds.py --run runs/baseline_tfidf_lr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toxicity.metrics import evaluate, tune_thresholds
from toxicity.predictions import load_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run folder with the two prediction files")
    args = parser.parse_args()
    run = Path(args.run)

    _, val_true, val_prob = load_predictions(run / "val_predictions.csv")
    _, test_true, test_prob = load_predictions(run / "test_predictions.csv")

    thresholds = tune_thresholds(val_true, val_prob)
    (run / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    flat = evaluate(test_true, test_prob, 0.5)
    tuned = evaluate(test_true, test_prob, thresholds)
    flat.to_csv(run / "test_metrics_flat.csv")
    tuned.to_csv(run / "test_metrics_tuned.csv")

    print("Thresholds chosen on validation:")
    for label, value in thresholds.items():
        print(f"  {label:14s} {value:.4f}")
    print("\nTest metrics, flat 0.5 cutoff:")
    print(flat.round(4))
    print("\nTest metrics, tuned per-label thresholds:")
    print(tuned.round(4))
    gain = tuned.loc["macro", "f1"] - flat.loc["macro", "f1"]
    print(f"\nMacro F1 change from tuning: {gain:+.4f}")
    print(f"Saved thresholds.json and metric tables to {run}/")


if __name__ == "__main__":
    main()
