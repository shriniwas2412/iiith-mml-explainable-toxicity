"""Score the template set with one model and report identity-term bias.

    python scripts/09_bias_audit.py --model baseline:runs/baseline_tfidf_lr/model.joblib \
        --thresholds runs/baseline_tfidf_lr/thresholds.json --out reports/bias/baseline
    python scripts/09_bias_audit.py --model hf:runs/distilbert_wbce/model \
        --thresholds runs/distilbert_wbce/thresholds.json --out reports/bias/distilbert_wbce
    python scripts/09_bias_audit.py --model detoxify:original --out reports/bias/detoxify_original

Without --thresholds a flat 0.5 cutoff is used for the false positive / negative rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from toxicity import LABELS
from toxicity.bias import (
    equality_differences,
    generalized_mean,
    mean_scores_by_label,
    overall_rates,
    per_identity_table,
    score_shift,
)
from toxicity.predictions import prob_columns
from toxicity.scoring import load_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="baseline:<joblib> | hf:<folder> | detoxify:<variant>")
    parser.add_argument("--templates", default="data/templates/templates_en.csv")
    parser.add_argument("--thresholds", default=None, help="thresholds.json from 06_tune_thresholds.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    templates = pd.read_csv(args.templates, keep_default_na=False)
    cutoff = json.loads(Path(args.thresholds).read_text())["toxic"] if args.thresholds else 0.5

    probs = load_scorer(args.model, device=args.device).predict_proba(templates["text"].tolist())
    scored = templates.copy()
    scored[prob_columns()] = probs
    scored.to_csv(out / "scores.csv", index=False)

    toxic_prob = probs[:, LABELS.index("toxic")]
    toxic_pred = (toxic_prob >= cutoff).astype(int)

    is_identity = templates["term_set"].isin(["dixon", "india"]).to_numpy()
    identity_rows = templates[is_identity].reset_index(drop=True)
    table = per_identity_table(identity_rows, toxic_prob[is_identity], toxic_pred[is_identity])
    table.to_csv(out / "per_identity_toxic.csv", index=False)

    has_frame = (templates["frame"] != "").to_numpy()
    framed = templates[has_frame].reset_index(drop=True)
    score_shift(framed, toxic_prob[has_frame]).to_csv(out / "score_shift_toxic.csv", index=False)
    mean_scores_by_label(framed, probs[has_frame], LABELS).to_csv(
        out / "mean_scores_nontoxic_by_label.csv", index=False
    )

    summary = {"model": args.model, "toxic_cutoff": cutoff, "sentences": len(templates), "term_sets": {}}
    for term_set in ["dixon", "india", "neutral", "control"]:
        rows = (templates["term_set"] == term_set).to_numpy()
        if not rows.any():
            continue
        rates = overall_rates(templates[rows], toxic_pred[rows])
        nontoxic = rows & (templates["toxicity"] == "nontoxic").to_numpy()
        entry = {
            "sentences": int(rows.sum()),
            **rates,
            "mean_toxic_prob_on_nontoxic": float(toxic_prob[nontoxic].mean()),
        }
        if term_set in ("dixon", "india"):
            subset = table[table["term_set"] == term_set]
            entry.update(equality_differences(subset, rates))
            for metric in ("subgroup_auc", "bpsn_auc", "bnsp_auc"):
                entry[f"gmean_{metric}"] = generalized_mean(subset[metric])
        summary["term_sets"][term_set] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nTen terms with the highest false positive rate on nontoxic sentences:")
    columns = ["term_set", "axis", "identity", "fpr", "mean_prob_nontoxic", "bpsn_auc"]
    print(table.sort_values("fpr", ascending=False)[columns].head(10).round(4).to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
