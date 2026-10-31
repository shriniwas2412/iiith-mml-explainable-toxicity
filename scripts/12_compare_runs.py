"""Collect every run that has tuned test metrics into one comparison table.

Reads runs/*/test_metrics_tuned.csv (and flat), plus reports/bias/<run>/summary.json
when it exists. Writes reports/compare/runs_table.csv and runs_table.md.

    python scripts/12_compare_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from toxicity import LABELS


def main() -> None:
    rows = []
    for tuned_path in sorted(Path("runs").glob("*/test_metrics_tuned.csv")):
        run = tuned_path.parent
        tuned = pd.read_csv(tuned_path, index_col=0)
        flat = pd.read_csv(run / "test_metrics_flat.csv", index_col=0) if (run / "test_metrics_flat.csv").exists() else None
        row = {
            "run": run.name,
            "macro_f1_flat": flat.loc["macro", "f1"] if flat is not None else None,
            "macro_f1_tuned": tuned.loc["macro", "f1"],
            "micro_f1_tuned": tuned.loc["micro", "f1"],
            "macro_pr_auc": tuned.loc["macro", "pr_auc"],
            **{f"f1_{label}": tuned.loc[label, "f1"] for label in LABELS},
        }
        summary = Path("reports/bias") / run.name / "summary.json"
        if summary.exists():
            sets = json.loads(summary.read_text())["term_sets"]
            for term_set in ("dixon", "india"):
                if term_set in sets:
                    row[f"{term_set}_fpr"] = sets[term_set]["fpr"]
                    row[f"{term_set}_gmean_bpsn_auc"] = sets[term_set].get("gmean_bpsn_auc")
        rows.append(row)
    table = pd.DataFrame(rows)
    out = Path("reports/compare")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "runs_table.csv", index=False)
    columns = [c for c in ["run", "macro_f1_flat", "macro_f1_tuned", "macro_pr_auc", "f1_threat",
                           "dixon_fpr", "india_fpr", "dixon_gmean_bpsn_auc"] if c in table]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in table[columns].itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    (out / "runs_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
