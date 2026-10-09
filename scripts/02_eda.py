"""Describe the training data: label rates, label overlap and comment lengths.

Writes CSV tables (and PNG plots if matplotlib is installed) to reports/eda/.

    python scripts/02_eda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from toxicity import LABELS, load_config
from toxicity.config import TEXT_COL
from toxicity.data import label_summary, load_train


def cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Cell (a, b) = share of comments with label a that also have label b."""
    y = df[LABELS]
    counts = y.T.dot(y)
    return counts.div(y.sum(), axis=0)


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    lengths = pd.DataFrame(
        {"chars": df[TEXT_COL].str.len(), "words": df[TEXT_COL].str.split().str.len()}
    )
    return lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T


def plots(df: pd.DataFrame, summary: pd.DataFrame, overlap: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    (summary.loc[LABELS, "share"] * 100).plot.bar(ax=ax)
    ax.set_ylabel("% of training comments")
    ax.set_title("Label rates")
    fig.tight_layout()
    fig.savefig(out_dir / "label_rates.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    df[TEXT_COL].str.split().str.len().clip(upper=500).plot.hist(bins=50, ax=ax)
    ax.set_xlabel("words per comment (clipped at 500)")
    ax.set_title("Comment length")
    fig.tight_layout()
    fig.savefig(out_dir / "comment_length.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(overlap.to_numpy(), vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_title("Share of row-label comments that also have column label")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(out_dir / "label_overlap.png", dpi=150)
    plt.close("all")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=str(Path(cfg["data"]["raw_dir"]) / "train.csv"))
    parser.add_argument("--out", default="reports/eda")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_train(args.train)

    summary = label_summary(df)
    overlap = cooccurrence(df)
    lengths = length_stats(df)
    summary.to_csv(out_dir / "label_counts.csv")
    overlap.to_csv(out_dir / "label_overlap.csv")
    lengths.to_csv(out_dir / "length_stats.csv")

    print(f"{len(df):,} comments, {int((df[TEXT_COL] == '').sum())} empty after cleaning, "
          f"{int(df[TEXT_COL].duplicated().sum())} duplicate texts\n")
    print(summary.assign(share=(summary["share"] * 100).round(2)).rename(columns={"share": "percent"}), "\n")
    print(overlap.round(2), "\n")
    print(lengths.round(1))
    plots(df, summary, overlap, out_dir)
    print(f"\nTables written to {out_dir}/")


if __name__ == "__main__":
    main()
