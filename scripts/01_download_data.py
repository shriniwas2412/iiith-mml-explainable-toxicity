"""Put the Jigsaw Toxic Comment files in data/raw/ and check their row counts.

On Kaggle, attach the competition data to the notebook and the files are copied
from /kaggle/input. Anywhere else they are downloaded from a public Hugging
Face mirror of the same three CSVs (about 134 MB in total).

    python scripts/01_download_data.py
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from toxicity import LABELS, load_config

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
HF_MIRROR = f"https://huggingface.co/datasets/thesofakillers/{COMPETITION}/resolve/main/"
FILES = ["train.csv", "test.csv", "test_labels.csv"]
EXPECTED_ROWS = {"train": 159_571, "test": 153_164, "scored test": 63_978}


def find_on_kaggle(name: str) -> Path | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for candidate in sorted(root.rglob(f"{name}*")):
        if COMPETITION in str(candidate) and candidate.name in (name, f"{name}.zip"):
            return candidate
    return None


def fetch(name: str, out_dir: Path, source: str) -> None:
    target = out_dir / name
    if target.exists():
        print(f"{name}: already in {out_dir}")
        return
    kaggle_file = find_on_kaggle(name) if source in ("auto", "kaggle") else None
    if kaggle_file is not None:
        pd.read_csv(kaggle_file).to_csv(target, index=False)
        print(f"{name}: copied from {kaggle_file}")
    elif source == "kaggle":
        raise FileNotFoundError(f"{name} not found under /kaggle/input. Attach the competition data.")
    else:
        print(f"{name}: downloading from the Hugging Face mirror ...")
        urllib.request.urlretrieve(HF_MIRROR + name, target)


def check(out_dir: Path) -> bool:
    train = pd.read_csv(out_dir / "train.csv")
    test = pd.read_csv(out_dir / "test.csv")
    labels = pd.read_csv(out_dir / "test_labels.csv")
    found = {
        "train": len(train),
        "test": len(test),
        "scored test": int((labels[LABELS] != -1).all(axis=1).sum()),
    }
    ok = True
    for name, expected in EXPECTED_ROWS.items():
        status = "OK" if found[name] == expected else "MISMATCH"
        ok &= found[name] == expected
        print(f"{name} rows: {found[name]:,} (expected {expected:,}) {status}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Output folder (default: data.raw_dir from the config)")
    parser.add_argument("--source", choices=["auto", "kaggle", "hf"], default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out or load_config()["data"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name, out_dir, args.source)
    if not check(out_dir):
        raise SystemExit("Row counts differ from the published dataset. Do not use these files.")


if __name__ == "__main__":
    main()
