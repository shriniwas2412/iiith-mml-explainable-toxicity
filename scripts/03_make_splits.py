"""Create the fixed train / validation / test files used by every experiment.

Validation is carved out of train.csv with multi-label stratification. The test
file keeps only the rows Kaggle scored. `--subset N` also writes a smaller
training file with the same label rates; validation and test stay the same, so
runs on different training sizes remain comparable.

    python scripts/03_make_splits.py
    python scripts/03_make_splits.py --subset 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from toxicity import LABELS, load_config
from toxicity.data import load_test, load_train, multilabel_split, subsample


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=cfg["data"]["raw_dir"])
    parser.add_argument("--out", default=cfg["data"]["processed_dir"])
    parser.add_argument("--val-size", type=float, default=cfg["data"]["val_size"])
    parser.add_argument("--subset", type=int, default=None, help="Also write train_subset_<N>.csv")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = multilabel_split(load_train(raw / "train.csv"), args.val_size, args.seed)
    test = load_test(raw / "test.csv", raw / "test_labels.csv")
    splits = {"train": train, "val": val, "test": test}
    if args.subset:
        splits[f"train_subset_{args.subset}"] = subsample(train, args.subset, args.seed)

    stats = {}
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
        stats[name] = {"rows": len(df), **df[LABELS].sum().astype(int).to_dict()}
    table = pd.DataFrame(stats).T
    table.to_csv(out / "split_stats.csv")
    print(table)
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
