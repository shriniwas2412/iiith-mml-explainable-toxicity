"""Train the TF-IDF + logistic regression baseline and save its predictions.

    python scripts/04_train_baseline.py
    python scripts/04_train_baseline.py --train data/processed/train_subset_30000.csv --out runs/baseline_30k

Then run scripts/06_tune_thresholds.py on the output folder.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from toxicity import LABELS, load_config
from toxicity.baseline import build_baseline
from toxicity.config import TEXT_COL
from toxicity.data import read_split
from toxicity.metrics import evaluate
from toxicity.predictions import save_predictions


def main() -> None:
    cfg = load_config()
    processed = Path(cfg["data"]["processed_dir"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=str(processed / "train.csv"))
    parser.add_argument("--val", default=str(processed / "val.csv"))
    parser.add_argument("--test", default=str(processed / "test.csv"))
    parser.add_argument("--out", default="runs/baseline_tfidf_lr")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = read_split(args.train), read_split(args.val), read_split(args.test)

    model = build_baseline(seed=cfg["seed"], **cfg["baseline"])
    start = time.time()
    model.fit(train[TEXT_COL], train[LABELS].to_numpy())
    print(f"Trained on {len(train):,} comments in {time.time() - start:.0f}s")

    joblib.dump(model, out / "model.joblib")
    for name, df in (("val", val), ("test", test)):
        prob = model.predict_proba(df[TEXT_COL])
        save_predictions(out / f"{name}_predictions.csv", df, prob)
        metrics = evaluate(df[LABELS].to_numpy(), prob)
        metrics.to_csv(out / f"{name}_metrics_flat.csv")
        print(f"\n{name} metrics at a flat 0.5 cutoff:")
        print(metrics.round(4))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
