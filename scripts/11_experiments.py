"""Run the week-3 experiment grid on ~30k training comments and compare them.

Every run uses word 1-2-gram TF-IDF (no character n-grams, so the grid runs in
minutes on a laptop), the same validation and test files, thresholds tuned on
validation, and the identity-term bias audit (scripts/09). Needs scripts 01, 03
(with --subset 30000 --subset-strategy rare_preserving --undersample-clean) and 08.

    python scripts/11_experiments.py                 # all experiments
    python scripts/11_experiments.py --only E3_lr_no_weights E4_svm

Groups:
  E2  preprocessing: raw text vs cleaned text (data.clean_text)
  E3  class imbalance: no weights / balanced weights / under-sampled clean rows /
      rare-preserving subset
  E4  model comparison: logistic regression, linear SVM, complement NB, NB-SVM
  E6  bias mitigation: Vaidya et al. (2020) weight of 3 on non-toxic comments
      that mention an identity term

Outputs: runs/exp_<name>/ (model, predictions, thresholds, metrics) and
reports/bias/exp_<name>/, then reports/experiments/ (tables and figures).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from toxicity import LABELS, load_config
from toxicity.baseline import build_model
from toxicity.config import TEXT_COL
from toxicity.data import clean_text, read_split
from toxicity.lexicon import load_identity_terms, mentions_any
from toxicity.metrics import evaluate, tune_thresholds
from toxicity.plots import SERIES, hbar, plt, save, style
from toxicity.predictions import save_predictions

STRATIFIED = "train_subset_30000.csv"
RARE = "train_subset_30000_rare.csv"
UNDERSAMPLED = "train_undersampled.csv"

# name: (group, training file, model kind, class weight, clean text, identity weight, what it tests)
EXPERIMENTS = {
    "E3_lr_no_weights": ("E3 imbalance", STRATIFIED, "lr", None, False, 1.0, "no imbalance handling"),
    "E3_lr_balanced": ("E3 imbalance", STRATIFIED, "lr", "balanced", False, 1.0, "class weights (reference run)"),
    "E3_lr_undersampled": ("E3 imbalance", UNDERSAMPLED, "lr", None, False, 1.0, "drop clean rows to 1:1"),
    "E3_lr_rare_preserving": ("E3 imbalance", RARE, "lr", "balanced", False, 1.0, "keep every rare-label row"),
    "E2_lr_cleaned": ("E2 preprocessing", STRATIFIED, "lr", "balanced", True, 1.0, "cleaned text"),
    "E4_svm": ("E4 models", STRATIFIED, "svm", "balanced", False, 1.0, "linear SVM (Davidson)"),
    "E4_cnb": ("E4 models", STRATIFIED, "cnb", None, False, 1.0, "complement naive Bayes"),
    "E4_nbsvm": ("E4 models", STRATIFIED, "nbsvm", "balanced", False, 1.0, "NB-SVM"),
    "E6_lr_identity_weight3": ("E6 bias", STRATIFIED, "lr", "balanced", False, 3.0, "Vaidya re-weighting"),
}


def run_one(name: str, spec: tuple, processed: Path, val: pd.DataFrame, test: pd.DataFrame, terms, cfg) -> dict:
    group, train_file, kind, class_weight, clean, identity_weight, _ = spec
    out = Path("runs") / f"exp_{name}"
    out.mkdir(parents=True, exist_ok=True)
    train = read_split(processed / train_file)
    prep = clean_text if clean else (lambda t: t)
    x_train = train[TEXT_COL].map(prep)
    x_val, x_test = val[TEXT_COL].map(prep), test[TEXT_COL].map(prep)
    y_train = train[LABELS].to_numpy()

    weights = np.ones(len(train))
    if identity_weight != 1.0:
        nontoxic_identity = mentions_any(train[TEXT_COL], terms) & (y_train.sum(axis=1) == 0)
        weights[nontoxic_identity] = identity_weight

    base = cfg["baseline"]
    model = build_model(
        kind,
        word_max_features=base["word_max_features"],
        use_char=False,
        min_df=base["min_df"],
        C=base["C"],
        class_weight=class_weight,
        seed=cfg["seed"],
    )
    start = time.time()
    model.fit(x_train, y_train, clf__sample_weight=weights)
    seconds = time.time() - start
    joblib.dump(model, out / "model.joblib")

    val_prob, test_prob = model.predict_proba(x_val), model.predict_proba(x_test)
    save_predictions(out / "val_predictions.csv", val, val_prob)
    save_predictions(out / "test_predictions.csv", test, test_prob)
    thresholds = tune_thresholds(val[LABELS].to_numpy(), val_prob)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    flat = evaluate(test[LABELS].to_numpy(), test_prob, 0.5)
    tuned = evaluate(test[LABELS].to_numpy(), test_prob, thresholds)
    flat.to_csv(out / "test_metrics_flat.csv")
    tuned.to_csv(out / "test_metrics_tuned.csv")
    config = {
        "group": group,
        "train_file": train_file,
        "train_rows": len(train),
        "model": kind,
        "features": "word 1-2-gram TF-IDF",
        "class_weight": class_weight,
        "clean_text": clean,
        "identity_weight": identity_weight,
        "upweighted_rows": int((weights != 1.0).sum()),
        "fit_seconds": round(seconds, 1),
        "seed": cfg["seed"],
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    bias_out = Path("reports/bias") / f"exp_{name}"
    subprocess.run(
        [sys.executable, "scripts/09_bias_audit.py", "--model", f"baseline:{out / 'model.joblib'}",
         "--thresholds", str(out / "thresholds.json"), "--out", str(bias_out)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"{name}: macro F1 tuned {tuned.loc['macro', 'f1']:.4f}, fit {seconds:.0f}s", flush=True)
    return config


def collect(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        run = Path("runs") / f"exp_{name}"
        config = json.loads((run / "run_config.json").read_text())
        flat = pd.read_csv(run / "test_metrics_flat.csv", index_col=0)
        tuned = pd.read_csv(run / "test_metrics_tuned.csv", index_col=0)
        bias = json.loads((Path("reports/bias") / f"exp_{name}" / "summary.json").read_text())["term_sets"]
        per_term = pd.read_csv(Path("reports/bias") / f"exp_{name}" / "per_identity_toxic.csv")
        row = {
            "experiment": name,
            "what_it_tests": EXPERIMENTS[name][6],
            **{k: config[k] for k in ("group", "train_rows", "model", "class_weight", "clean_text", "identity_weight", "fit_seconds")},
            "macro_f1_flat": flat.loc["macro", "f1"],
            "macro_f1_tuned": tuned.loc["macro", "f1"],
            "micro_f1_tuned": tuned.loc["micro", "f1"],
            "macro_pr_auc": tuned.loc["macro", "pr_auc"],
            "macro_roc_auc": tuned.loc["macro", "roc_auc"],
        }
        for label in LABELS:
            row[f"f1_{label}"] = tuned.loc[label, "f1"]
            row[f"pr_auc_{label}"] = tuned.loc[label, "pr_auc"]
        for term_set in ("dixon", "india"):
            row[f"{term_set}_fpr"] = bias[term_set]["fpr"]
            row[f"{term_set}_fnr"] = bias[term_set]["fnr"]
            row[f"{term_set}_gmean_bpsn_auc"] = bias[term_set].get("gmean_bpsn_auc")
        row["neutral_fpr"] = bias["neutral"]["fpr"]
        for term in ("gay", "homosexual", "muslim", "black"):
            hit = per_term[(per_term["term_set"] == "dixon") & (per_term["identity"] == term)]
            row[f"fpr_{term}"] = float(hit["fpr"].iloc[0]) if len(hit) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def figures(table: pd.DataFrame, out: Path) -> None:
    style()
    order = table["experiment"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = [SERIES[{"E2": 2, "E3": 0, "E4": 1, "E6": 2}[e[:2]] % 3] for e in order]
    hbar(ax, order, table["macro_f1_tuned"].tolist(), color=colors, fmt="{:.3f}")
    ax.set_title("Macro F1 on the test set (thresholds tuned on validation)")
    ax.set_xlim(0, max(table["macro_f1_tuned"]) * 1.2)
    save(fig, out / "fig_e_macro_f1.png")

    rare = table[table["group"] == "E3 imbalance"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pos = np.arange(len(LABELS))
    width = 0.8 / len(rare)
    palette = SERIES + ["#eda100"]
    for i, row in enumerate(rare.itertuples()):
        values = [getattr(row, f"f1_{label}") for label in LABELS]
        ax.bar(pos + (i - (len(rare) - 1) / 2) * width, values, width=width * 0.92, color=palette[i], label=row.experiment)
    ax.set_xticks(pos, LABELS)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_title("E3: per-label F1 (tuned) for each imbalance strategy")
    ax.legend(fontsize=8, ncol=2)
    save(fig, out / "fig_e3_per_label_f1.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    hbar(axes[0], order, table["dixon_fpr"].tolist(), color=SERIES[0])
    axes[0].set_title("FPR on harmless Dixon sentences (lower = better)", fontsize=10)
    hbar(axes[1], order, table["dixon_gmean_bpsn_auc"].tolist(), color=SERIES[1], fmt="{:.3f}")
    axes[1].set_yticks([])
    axes[1].set_title("Gen. mean BPSN AUC, Dixon terms (higher = better)", fontsize=10)
    save(fig, out / "fig_e6_bias_fpr.png")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment names")
    parser.add_argument("--report-only", action="store_true", help="Skip training; rebuild tables and figures")
    args = parser.parse_args()
    names = args.only or list(EXPERIMENTS)

    processed = Path(cfg["data"]["processed_dir"])
    if not args.report_only:
        val, test = read_split(processed / "val.csv"), read_split(processed / "test.csv")
        terms = load_identity_terms()
        for name in names:
            run_one(name, EXPERIMENTS[name], processed, val, test, terms, cfg)

    done = [n for n in EXPERIMENTS if (Path("runs") / f"exp_{n}" / "run_config.json").exists()]
    out = Path("reports/experiments")
    out.mkdir(parents=True, exist_ok=True)
    table = collect(done)
    table.to_csv(out / "results.csv", index=False)
    figures(table, out)
    columns = ["experiment", "macro_f1_flat", "macro_f1_tuned", "macro_pr_auc", "f1_threat", "f1_severe_toxic",
               "dixon_fpr", "india_fpr", "fpr_gay", "dixon_gmean_bpsn_auc", "fit_seconds"]
    print("\n" + table[columns].round(4).to_string(index=False))
    print(f"\nSaved to {out}/ (results.csv, fig_e*.png)")


if __name__ == "__main__":
    main()
