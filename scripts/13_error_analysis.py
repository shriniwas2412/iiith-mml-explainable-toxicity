"""Error analysis for one or more runs: which comments each model gets wrong, and why.

Uses each run's tuned thresholds.json and test_predictions.csv. Comment text comes from
data/processed/test.csv (joined on id). Writes to reports/error_analysis/<run>/ and
reports/error_analysis/.

    python scripts/13_error_analysis.py
    python scripts/13_error_analysis.py --run runs/exp_E3_lr_balanced runs/exp_E3_lr_undersampled

1. Per label: TP, FP, FN, TN, plus the 20 highest-scored false positives and the 20
   lowest-scored false negatives, in <label>_errors.csv (text cut to 300 characters).
2. Error categories as shares of each label's false positives or false negatives, in summary.csv.
   The rules are simple heuristics, not manual annotation:
   - identity:         comment mentions a term from the identity lexicon (lexicon.mentions_any).
                       Ordinary words such as "black" also match, so this is an upper bound.
   - profanity_clean:  comment has a profanity token (PROFANITY below) and no label on any of the six.
   - all_caps:         at least 10 letters and at least 70% of them upper case.
   - non_ascii:        at least 5 letters and more than 30% of them non-ASCII.
   - short:            fewer than 5 words.
   The all_caps and non_ascii rules are the same as in scripts/10_data_analysis.py.
3. Across runs: false-positive rate on real test comments that mention an identity term
   versus those that do not, in identity_fp_real_comments.csv and fig_identity_fp_real.png.
   The rate is FP / (true negatives + FP), pooled over the six label decisions for all_labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toxicity import LABELS, load_config
from toxicity.config import ID_COL, TEXT_COL
from toxicity.data import normalize_text, read_split
from toxicity.lexicon import load_identity_terms, mentions_any
from toxicity.metrics import apply_thresholds
from toxicity.plots import SERIES, plt, save, style
from toxicity.predictions import load_predictions

# A short list of common English profanity stems with a few suffixes. It is a rough
# proxy for "profanity-like", not a full lexicon.
PROFANITY = r"\b(?:fuck|shit|bitch|bastard|asshole|cunt|dick|crap|damn|piss)(?:ing|er|ers|s|y|ty|ed|es)?\b"
TOP_N = 20
SNIPPET_CHARS = 300
GROUPS = {"mentions_identity_term": True, "no_identity_term": False}


# The same two rules as scripts/10_data_analysis.py, kept here so this script stands alone.
def shouting(text: pd.Series) -> np.ndarray:
    """At least 10 letters and at least 70% of them upper case."""
    letters = text.str.count(r"[A-Za-z]")
    upper = text.str.count(r"[A-Z]")
    return ((letters >= 10) & (upper / letters.clip(lower=1) >= 0.7)).to_numpy()


def mostly_non_latin(text: pd.Series) -> np.ndarray:
    """At least 5 letters and more than 30% of them outside ASCII."""
    letters = text.str.count(r"[^\W\d_]")
    non_ascii = text.str.count(r"[^\x00-\x7f]")
    return ((letters >= 5) & (non_ascii / letters.clip(lower=1) > 0.3)).to_numpy()


def comment_features(test: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags per test comment. Computed once and shared by all runs."""
    text = test[TEXT_COL]
    profanity = text.str.lower().str.contains(PROFANITY, regex=True).to_numpy()
    clean = (test[LABELS].sum(axis=1) == 0).to_numpy()
    return pd.DataFrame(
        {
            "identity": mentions_any(text, terms),
            "profanity_clean": profanity & clean,
            "all_caps": shouting(text),
            "non_ascii": mostly_non_latin(text),
            "short": (text.str.split().str.len() < 5).to_numpy(),
        }
    )


def flatten(truth: np.ndarray, pred: np.ndarray, score: np.ndarray, feats: pd.DataFrame) -> dict:
    """One row per (comment, label) decision, so a single code path serves each label and the pooled rows."""
    n_labels = len(LABELS)
    flat = {
        "truth": truth.ravel(),
        "pred": pred.ravel(),
        "score": score.ravel(),
        "label": np.tile(np.arange(n_labels), truth.shape[0]),
    }
    for key in feats.columns:
        flat[key] = np.repeat(feats[key].to_numpy(), n_labels)
    return flat


def share(mask: np.ndarray, feature: np.ndarray) -> float:
    """Share of the comments in `mask` that also have `feature`."""
    return float((mask & feature).sum() / mask.sum()) if mask.sum() else np.nan


def summary_row(name: str, flat: dict, sel: np.ndarray) -> dict:
    truth, pred = flat["truth"][sel], flat["pred"][sel]
    fp_mask, fn_mask = ~truth & pred, truth & ~pred
    tp, fp = int((truth & pred).sum()), int(fp_mask.sum())
    fn, tn = int(fn_mask.sum()), int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    row = {"label": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
    for key in ["identity", "profanity_clean", "all_caps", "non_ascii"]:
        row[f"fp_{key}_share"] = share(fp_mask, flat[key][sel])
    for key in ["short", "all_caps", "non_ascii"]:
        row[f"fn_{key}_share"] = share(fn_mask, flat[key][sel])
    return row


def error_examples(label: str, ids: np.ndarray, texts: np.ndarray, score: np.ndarray, fp: np.ndarray,
                   fn: np.ndarray, threshold: float) -> pd.DataFrame:
    """The top false positives (highest score) and false negatives (lowest score) for one label."""
    frames = []
    for kind, mask, rank in [("false_positive", fp, -score), ("false_negative", fn, score)]:
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(rank[idx], kind="stable")][:TOP_N]
        frames.append(
            pd.DataFrame(
                {
                    "error_type": kind,
                    ID_COL: ids[idx],
                    "label": label,
                    "score": score[idx],
                    "threshold": threshold,
                    TEXT_COL: [normalize_text(t)[:SNIPPET_CHARS] for t in texts[idx]],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def analyse_run(run: Path, test: pd.DataFrame, feats: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the per-label error files and summary.csv for one run. Returns (summary, identity rows)."""
    thresholds = json.loads((run / "thresholds.json").read_text())
    ids, y_true, y_prob = load_predictions(run / "test_predictions.csv")
    pos = pd.Index(test[ID_COL]).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"{run}: {(pos < 0).sum()} prediction ids are not in the test file")
    texts = test[TEXT_COL].to_numpy()[pos]
    feats = feats.iloc[pos].reset_index(drop=True)

    truth = y_true == 1
    pred = apply_thresholds(y_prob, thresholds) == 1
    fp, fn = ~truth & pred, truth & ~pred
    out.mkdir(parents=True, exist_ok=True)

    for j, label in enumerate(LABELS):
        error_examples(label, ids.to_numpy(), texts, y_prob[:, j], fp[:, j], fn[:, j], thresholds[label]).to_csv(
            out / f"{label}_errors.csv", index=False
        )

    flat = flatten(truth, pred, y_prob, feats)
    rows = [summary_row(label, flat, flat["label"] == j) for j, label in enumerate(LABELS)]
    rows.append(summary_row("all_labels", flat, np.ones(len(flat["truth"]), dtype=bool)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    identity_rows = []
    for group, in_group in GROUPS.items():
        in_group_flat = flat["identity"] == in_group
        n_comments = int((feats["identity"] == in_group).sum())
        for j, label in enumerate(LABELS + ["all_labels"]):
            sel = in_group_flat if label == "all_labels" else in_group_flat & (flat["label"] == j)
            negatives = int((~flat["truth"][sel]).sum())
            false_pos = int((~flat["truth"][sel] & flat["pred"][sel]).sum())
            identity_rows.append(
                {
                    "run": run.name,
                    "label": label,
                    "group": group,
                    "comments": n_comments,
                    "negatives": negatives,
                    "false_positives": false_pos,
                    "fp_rate": false_pos / negatives if negatives else np.nan,
                }
            )
    return summary, pd.DataFrame(identity_rows)


def plot_identity_fp(compare: pd.DataFrame, out: Path) -> None:
    pooled = compare[compare["label"] == "all_labels"]
    runs = list(dict.fromkeys(pooled["run"]))
    pos = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    series = [("mentions_identity_term", SERIES[1], "mentions an identity term"),
              ("no_identity_term", SERIES[0], "no identity term")]
    for k, (group, color, name) in enumerate(series):
        vals = [float(pooled[(pooled["run"] == r) & (pooled["group"] == group)]["fp_rate"].iloc[0]) for r in runs]
        bars = ax.bar(pos + (k - 0.5) * 0.38, vals, width=0.38, color=color, label=name)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], fontsize=8, color="#52514e", padding=2)
    ax.set_xticks(pos, runs)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_title("False positive rate on real test comments")
    ax.legend()
    save(fig, out / "fig_identity_fp_real.png")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", default=["runs/exp_E3_lr_balanced"],
                        help="One or more run folders with test_predictions.csv and thresholds.json")
    parser.add_argument("--test", default=Path(cfg["data"]["processed_dir"]) / "test.csv",
                        help="Scored test file with comment_text (joined on id)")
    parser.add_argument("--india", default="data/external/india", help="Folder with the Indian term lists")
    parser.add_argument("--out", default="reports/error_analysis")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    test = read_split(args.test)
    terms = load_identity_terms(args.india)
    feats = comment_features(test, terms)

    compare_parts = []
    for run_path in args.run:
        run = Path(run_path)
        summary, identity_rows = analyse_run(run, test, feats, out / run.name)
        compare_parts.append(identity_rows)
        print(f"\n== {run.name}  (thresholds from {run}/thresholds.json)")
        print(summary.round(3).to_string(index=False))
        print(f"Saved errors and summary.csv to {out / run.name}/")

    compare = pd.concat(compare_parts, ignore_index=True)
    compare.to_csv(out / "identity_fp_real_comments.csv", index=False)
    pooled = compare[compare["label"] == "all_labels"].pivot(index="run", columns="group", values="fp_rate")
    pooled = pooled[["mentions_identity_term", "no_identity_term"]]
    pooled["identity_over_none"] = pooled["mentions_identity_term"] / pooled["no_identity_term"]
    print("\nFalse positive rate on real test comments (pooled over the six labels):")
    print(pooled.round(4).to_string())
    plot_identity_fp(compare, out)
    print(f"\nSaved identity_fp_real_comments.csv and fig_identity_fp_real.png to {out}/")


if __name__ == "__main__":
    main()
