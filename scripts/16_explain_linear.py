"""Explain the linear (TF-IDF + logistic regression) runs: global weights, identity terms, local examples.

Needs runs/<run>/model.joblib and, for the error examples, scripts/13_error_analysis.py output.
Writes to reports/explain/.

    python scripts/16_explain_linear.py
    python scripts/16_explain_linear.py --run runs/exp_E3_lr_balanced runs/exp_E6_lr_identity_weight3

1. Global: top_features_<label>.csv per run (the 20 largest positive coefficients) and
   fig_top_features.png (top 10 per label). Profanity is masked in the figure and in the
   markdown/HTML, but not in the CSV.
2. Identity terms: identity_term_weights.csv and fig_identity_term_weights.png. The coefficient
   of every term from the identity lexicon that is a word feature, for toxic and identity_hate,
   in each run. Terms that are not in a run's vocabulary (min_df=2) are left out.
3. Local: local_examples.md and local_examples.html per run. Each word's contribution to the
   logit is tf-idf value x coefficient, summed over the features that belong to the word (see
   toxicity/explain.py). The contributions plus the intercept equal the model's logit exactly.
   Examples are a few template sentences and three real test comments the run gets wrong in
   toxic_errors.csv: the two highest-scored false positives without profanity or repeated words and
   the lowest-scored false negative, among comments of at most 120 characters.

LIME is not used: the attribution is exact for a linear model, so a sampled approximation adds nothing.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toxicity import LABELS
from toxicity.explain import LinearExplainer, identity_weight_table, mask_text, mask_word
from toxicity.lexicon import load_identity_terms
from toxicity.plots import GRID, MUTED, SERIES, hbar, plt, save, style

TOP_N = 20
FIG_TOP_N = 10
IDENTITY_LABELS = ["toxic", "identity_hate"]
FIG_IDENTITY_TERMS = 15
MAX_ERROR_CHARS = 120
TEMPLATES = [
    "I am a gay person",
    "I am a homosexual person",
    "I am a proud person",
    "I am a muslim person",
    "You are a great actor",
]
# Highlight colours for positive (pushes toward the label) and negative contributions.
POS_RGB, NEG_RGB = "235,104,52", "42,120,214"


def load_run(run: Path) -> tuple[LinearExplainer, dict]:
    import joblib

    # joblib files can run code when loaded: only load files you created.
    return LinearExplainer(joblib.load(run / "model.joblib")), json.loads((run / "thresholds.json").read_text())


def plot_top_features(tops: dict[str, pd.DataFrame], run: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, label in zip(axes.ravel(), LABELS):
        top = tops[label].head(FIG_TOP_N)
        hbar(ax, [mask_text(f) for f in top["feature"]], top["coefficient"], fmt="{:.1f}")
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Largest positive coefficients per label: {run}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig_top_features.png")


def plot_identity_weights(table: pd.DataFrame, runs: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(IDENTITY_LABELS), figsize=(10, 5.2), sharey=False)
    for ax, label in zip(axes, IDENTITY_LABELS):
        cols = [f"{r}_{label}" for r in runs]
        top = table.dropna(subset=cols, how="all").nlargest(FIG_IDENTITY_TERMS, cols[0])
        y = np.arange(len(top))
        for k, (run, col) in enumerate(zip(runs, cols)):
            ax.barh(y + (k - (len(runs) - 1) / 2) * 0.38, top[col].fillna(0), height=0.36, color=SERIES[k], label=run)
        ax.set_yticks(y, [mask_text(t) for t in top["identity"]])
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{label} coefficient")
        ax.tick_params(axis="y", labelsize=8)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out / "fig_identity_term_weights.png")


def pick_error_examples(run: Path, errors_dir: Path) -> list[tuple[str, str]]:
    """(kind, text) for the error comments to explain. Empty if 13_error_analysis was not run."""
    path = errors_dir / run.name / "toxic_errors.csv"
    if not path.exists():
        return []
    errors = pd.read_csv(path, keep_default_na=False)
    errors = errors[errors["comment_text"].str.len() <= MAX_ERROR_CHARS]
    # Skip comments with profanity and comments that repeat a word (keyboard spam).
    clean = errors["comment_text"].map(lambda t: mask_text(t) == t and len(set(t.lower().split())) == len(t.split()))
    fp = errors[(errors["error_type"] == "false_positive") & clean].head(2)
    fn = errors[errors["error_type"] == "false_negative"].head(1)
    return [("false positive (toxic)", t) for t in fp["comment_text"]] + [("false negative (toxic)", t) for t in fn["comment_text"]]


def explain_all(explainer: LinearExplainer, text: str) -> dict:
    return {label: explainer.explain_text(text, label) for label in IDENTITY_LABELS}


def tint(value: float, scale: float) -> str:
    rgb = POS_RGB if value > 0 else NEG_RGB
    return f"rgba({rgb},{min(abs(value) / scale, 1) * 0.75:.2f})"


def markdown_block(title: str, text: str, results: dict, thresholds: dict) -> str:
    lines = [f"### {title}", "", f"> {mask_text(text)}", ""]
    scores = ", ".join(
        f"{label} {r['probability']:.3f} (cutoff {thresholds[label]:.3f})" for label, r in results.items()
    )
    lines += [f"Scores: {scores}", ""]
    base = results[IDENTITY_LABELS[0]]
    order = np.argsort(-np.abs(base["contribution"]), kind="stable")[:8]
    lines += ["| word | " + " | ".join(IDENTITY_LABELS) + " |", "|---|" + "---:|" * len(IDENTITY_LABELS)]
    for i in sorted(order):
        cells = [f"{results[label]['contribution'][i]:+.2f}" for label in IDENTITY_LABELS]
        mark = "**" if abs(base["contribution"][i]) >= 0.5 else ""
        lines.append(f"| {mark}{mask_word(base['words'][i])}{mark} | " + " | ".join(cells) + " |")
    lines.append("| (other) | " + " | ".join(f"{results[l]['other']:+.2f}" for l in IDENTITY_LABELS) + " |")
    lines.append("| intercept | " + " | ".join(f"{results[l]['intercept']:+.2f}" for l in IDENTITY_LABELS) + " |")
    lines.append("| logit | " + " | ".join(f"{results[l]['logit']:+.2f}" for l in IDENTITY_LABELS) + " |")
    return "\n".join(lines) + "\n"


def html_block(title: str, text: str, results: dict, thresholds: dict) -> str:
    parts = [f"<h3>{html.escape(title)}</h3>"]
    for label, r in results.items():
        scale = max(np.abs(r["contribution"]).max(), 1.0)
        spans = " ".join(
            f'<span style="background:{tint(c, scale)}" title="{c:+.2f}">{html.escape(mask_word(w))}</span>'
            for w, c in zip(r["words"], r["contribution"])
        )
        parts.append(
            f"<p><b>{label}</b> p={r['probability']:.3f} (cutoff {thresholds[label]:.3f}, "
            f"intercept {r['intercept']:+.2f}, other {r['other']:+.2f})<br>{spans}</p>"
        )
    return "\n".join(parts)


def write_local_examples(explainer: LinearExplainer, thresholds: dict, run: Path, errors_dir: Path, out: Path) -> None:
    examples = [("template sentence", t) for t in TEMPLATES] + pick_error_examples(run, errors_dir)
    md = [
        f"# Local explanations: {run.name}",
        "",
        "Contribution of each word to the logit (tf-idf value x coefficient, summed over the word's features).",
        "Positive pushes toward the label. Contributions + intercept = logit, exactly. Profanity is masked.",
        "",
    ]
    page = [
        "<!doctype html><meta charset='utf-8'><title>Local explanations</title>",
        f"<style>body{{font:15px/1.7 sans-serif;max-width:860px;margin:2em auto;color:#0b0b0b}}"
        f"span{{padding:2px 4px;border-radius:3px}}h3{{margin-bottom:0}}small{{color:#52514e}}</style>",
        f"<h2>Local explanations: {html.escape(run.name)}</h2>",
        "<p><small>Word colour = contribution to the logit: orange pushes toward the label, blue away. "
        "Hover for the value. Profanity is masked.</small></p>",
    ]
    for kind, text in examples:
        results = explain_all(explainer, text)
        md.append(markdown_block(f"{kind}", text, results, thresholds))
        page.append(html_block(kind, text, results, thresholds))
    (out / "local_examples.md").write_text("\n".join(md), encoding="utf-8")
    (out / "local_examples.html").write_text("\n".join(page), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", default=["runs/exp_E3_lr_balanced", "runs/exp_E6_lr_identity_weight3"],
                        help="Run folders with model.joblib and thresholds.json (linear models only)")
    parser.add_argument("--india", default="data/external/india", help="Folder with the Indian term lists")
    parser.add_argument("--errors", default="reports/error_analysis", help="Output folder of 13_error_analysis.py")
    parser.add_argument("--out", default="reports/explain")
    args = parser.parse_args()
    out = Path(args.out)
    style()

    runs = [Path(r) for r in args.run]
    explainers, thresholds = {}, {}
    for run in runs:
        explainers[run.name], thresholds[run.name] = load_run(run)

    # 1. Global top features.
    for run in runs:
        folder = out / run.name
        folder.mkdir(parents=True, exist_ok=True)
        tops = {label: explainers[run.name].top_features(label, TOP_N) for label in LABELS}
        for label, top in tops.items():
            top.to_csv(folder / f"top_features_{label}.csv", index=False)
        plot_top_features(tops, run.name, folder)
        print(f"\n== {run.name}: top 5 features per label (masked)")
        for label in LABELS:
            top = tops[label].head(5)
            print(f"  {label:14s}" + ", ".join(f"{mask_text(f)} {c:.2f}" for f, c in zip(top["feature"], top["coefficient"])))

    # 2. Identity terms.
    terms = load_identity_terms(args.india)
    names = [r.name for r in runs]
    table = identity_weight_table(explainers, terms, IDENTITY_LABELS)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "identity_term_weights.csv", index=False)
    plot_identity_weights(table, names, out)
    print(f"\n== identity terms in the vocabulary: {len(table)} of {terms[['term_set', 'identity']].drop_duplicates().shape[0]}")
    for label in IDENTITY_LABELS:
        cols = [f"{n}_{label}" for n in names]
        print(f"  {label}: highest coefficients")
        print(table.nlargest(6, cols[0])[["identity", *cols]].assign(identity=lambda d: d["identity"].map(mask_text)).round(2).to_string(index=False))

    # 3. Local explanations.
    for run in runs:
        write_local_examples(explainers[run.name], thresholds[run.name], run, Path(args.errors), out / run.name)
        for label in IDENTITY_LABELS:
            print(f"\n== {run.name}, {label}: template sentences")
            for text in TEMPLATES:
                r = explainers[run.name].explain_text(text, label)
                top = np.argsort(-np.abs(r["contribution"]))[:3]
                shown = ", ".join(f"{mask_word(r['words'][i])} {r['contribution'][i]:+.2f}" for i in top)
                print(f"  p={r['probability']:.3f}  {text!r}: {shown}  (intercept {r['intercept']:+.2f})")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
