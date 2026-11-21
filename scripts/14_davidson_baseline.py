"""Replicate Davidson et al. (2017) on the Jigsaw comments and audit it for bias.

Features and the L1-then-L2 logistic regression are in src/toxicity/davidson.py,
which also lists where this differs from the paper (Wikipedia instead of Twitter,
six binary labels instead of three classes). Trains on ~30k comments like the
scripts/11 grid, so the numbers are comparable with runs/exp_*.

    python scripts/14_davidson_baseline.py
    python scripts/14_davidson_baseline.py --train data/processed/train_subset_30000.csv --out runs/davidson_30k

Needs the NLTK tagger: python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
Outputs: runs/davidson_30k/ (model, predictions, thresholds, metrics) and
reports/bias/davidson_30k/. scripts/12_compare_runs.py picks the run up as is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib

from toxicity import LABELS, load_config
from toxicity.config import TEXT_COL
from toxicity.data import read_split
from toxicity.davidson import build_davidson
from toxicity.metrics import evaluate, tune_thresholds
from toxicity.predictions import save_predictions


def main() -> None:
    cfg = load_config()
    processed = Path(cfg["data"]["processed_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=str(processed / "train_subset_30000.csv"))
    parser.add_argument("--val", default=str(processed / "val.csv"))
    parser.add_argument("--test", default=str(processed / "test.csv"))
    parser.add_argument("--out", default="runs/davidson_30k")
    parser.add_argument("--bias-out", default="reports/bias/davidson_30k")
    parser.add_argument("--l1-c", type=float, default=0.1, help="C of the L1 feature-selection model")
    parser.add_argument("--l2-c", type=float, default=1.0, help="C of the final L2 model")
    parser.add_argument("--skip-bias", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = read_split(args.train), read_split(args.val), read_split(args.test)

    model = build_davidson(
        l1_C=args.l1_c, l2_C=args.l2_c, class_weight=cfg["baseline"]["class_weight"], seed=cfg["seed"]
    )
    start = time.time()
    model.fit(train[TEXT_COL], train[LABELS].to_numpy())
    seconds = time.time() - start
    n_features = model.named_steps["features"].transform(train[TEXT_COL].head(5)).shape[1]
    selected = dict(zip(LABELS, model.named_steps["clf"].n_selected()))
    print(f"Trained on {len(train):,} comments in {seconds:.0f}s ({n_features:,} features)")
    print(f"Features kept by the L1 step: {selected}")
    joblib.dump(model, out / "model.joblib")

    val_prob, test_prob = model.predict_proba(val[TEXT_COL]), model.predict_proba(test[TEXT_COL])
    save_predictions(out / "val_predictions.csv", val, val_prob)
    save_predictions(out / "test_predictions.csv", test, test_prob)
    thresholds = tune_thresholds(val[LABELS].to_numpy(), val_prob)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    flat = evaluate(test[LABELS].to_numpy(), test_prob, 0.5)
    tuned = evaluate(test[LABELS].to_numpy(), test_prob, thresholds)
    flat.to_csv(out / "test_metrics_flat.csv")
    tuned.to_csv(out / "test_metrics_tuned.csv")
    config = {
        "train_file": Path(args.train).name,
        "train_rows": len(train),
        "model": "davidson",
        "features": "word 1-3-gram TF-IDF (stemmed), POS 1-3-gram TF-IDF, readability, VADER, counts",
        "l1_C": args.l1_c,
        "l2_C": args.l2_c,
        "class_weight": cfg["baseline"]["class_weight"],
        "n_features": n_features,
        "features_kept": selected,
        "fit_seconds": round(seconds, 1),
        "seed": cfg["seed"],
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    print("\nThresholds chosen on validation:")
    for label, value in thresholds.items():
        print(f"  {label:14s} {value:.4f}")
    print("\nTest metrics, tuned per-label thresholds:")
    print(tuned.round(4))
    print(f"\nMacro F1 flat {flat.loc['macro', 'f1']:.4f}, tuned {tuned.loc['macro', 'f1']:.4f}")

    if not args.skip_bias:
        subprocess.run(
            [sys.executable, "scripts/09_bias_audit.py", "--model", f"baseline:{out / 'model.joblib'}",
             "--thresholds", str(out / "thresholds.json"), "--out", args.bias_out],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        sets = json.loads((Path(args.bias_out) / "summary.json").read_text())["term_sets"]
        print(f"Dixon FPR {sets['dixon']['fpr']:.4f}, gen. mean BPSN AUC {sets['dixon']['gmean_bpsn_auc']:.4f}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
