"""Identity-term bias metrics on template sentences.

All functions take the template table from `toxicity.templates` (columns
toxicity, template, frame, identity, axis, term_set) plus model scores for the
same rows. "Toxic" ground truth is the template's own toxic / nontoxic tag.

References: Dixon et al. 2018 (false positive / negative equality difference),
Borkan et al. 2019 (subgroup, BPSN and BNSP AUC), Prabhakaran et al. 2019
(score shift under identity perturbation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FRAME_KEY = ["template", "toxicity", "frame"]
GROUP_KEY = ["term_set", "axis", "identity"]


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, score))


def overall_rates(templates: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """False positive and false negative rate over all rows passed in."""
    toxic = (templates["toxicity"] == "toxic").to_numpy()
    return {
        "fpr": float(pred[~toxic].mean()) if (~toxic).any() else float("nan"),
        "fnr": float(1 - pred[toxic].mean()) if toxic.any() else float("nan"),
    }


def per_identity_table(templates: pd.DataFrame, prob: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """One row per identity term with error rates and the three bias AUCs.

    subgroup_auc: toxic vs nontoxic sentences that mention the term.
    bpsn_auc: nontoxic sentences with the term vs toxic sentences without it.
              Low values mean the term's harmless sentences score too high.
    bnsp_auc: toxic sentences with the term vs nontoxic sentences without it.
              Low values mean the term's toxic sentences score too low.
    "Without it" is every row whose identity differs, so pass in identity rows only.
    """
    toxic = (templates["toxicity"] == "toxic").to_numpy()
    identity = templates["identity"].to_numpy()
    term_sets = templates["term_set"].to_numpy()
    axes = templates["axis"].to_numpy()
    rows = []
    for term_set, axis, term in templates[GROUP_KEY].drop_duplicates().sort_values(GROUP_KEY).itertuples(index=False):
        inside = (term_sets == term_set) & (axes == axis) & (identity == term)
        outside = identity != term
        bpsn = (inside & ~toxic) | (outside & toxic)
        bnsp = (inside & toxic) | (outside & ~toxic)
        rows.append(
            {
                "term_set": term_set,
                "axis": axis,
                "identity": term,
                "n_nontoxic": int((inside & ~toxic).sum()),
                "n_toxic": int((inside & toxic).sum()),
                "fpr": float(pred[inside & ~toxic].mean()) if (inside & ~toxic).any() else np.nan,
                "fnr": float(1 - pred[inside & toxic].mean()) if (inside & toxic).any() else np.nan,
                "mean_prob_nontoxic": float(prob[inside & ~toxic].mean()) if (inside & ~toxic).any() else np.nan,
                "mean_prob_toxic": float(prob[inside & toxic].mean()) if (inside & toxic).any() else np.nan,
                "subgroup_auc": _auc(toxic[inside], prob[inside]),
                "bpsn_auc": _auc(toxic[bpsn], prob[bpsn]),
                "bnsp_auc": _auc(toxic[bnsp], prob[bnsp]),
            }
        )
    return pd.DataFrame(rows)


def equality_differences(table: pd.DataFrame, overall: dict[str, float]) -> dict[str, float]:
    """Sum over terms of |term rate - overall rate| for FPR (FPED) and FNR (FNED)."""
    return {
        "fped": float((table["fpr"] - overall["fpr"]).abs().sum()),
        "fned": float((table["fnr"] - overall["fnr"]).abs().sum()),
    }


def generalized_mean(values, p: float = -5.0) -> float:
    """Power mean ((1/N) * sum v^p)^(1/p) over non-NaN values.

    With p = -5 the lowest values dominate, so one badly treated identity pulls
    the score down (Borkan et al. 2019; the Jigsaw unintended-bias competition
    and Vaidya et al. 2020 use it over per-identity AUCs).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    return float(np.mean(v**p) ** (1 / p))


def score_shift(templates: pd.DataFrame, prob: np.ndarray, reference_set: str = "neutral") -> pd.DataFrame:
    """Mean change in score when a frame is filled with each identity term.

    The reference for a frame is its mean score over `reference_set` rows
    (frames filled with a non-identity word). If the table has no such rows,
    the frame's mean over all rows is used instead.
    """
    df = templates[FRAME_KEY + GROUP_KEY].copy()
    df["prob"] = prob
    df = df[df["frame"] != ""]
    reference_rows = df[df["term_set"] == reference_set]
    if reference_rows.empty:
        reference_rows = df
    reference = reference_rows.groupby(FRAME_KEY)["prob"].mean().rename("reference")
    df = df.join(reference, on=FRAME_KEY)
    df["shift"] = df["prob"] - df["reference"]
    out = (
        df[df["term_set"] != reference_set]
        .groupby(GROUP_KEY + ["toxicity"])["shift"]
        .mean()
        .unstack("toxicity")
        .add_prefix("shift_")
        .reset_index()
    )
    out.columns.name = None
    return out


def mean_scores_by_label(
    templates: pd.DataFrame, probs: np.ndarray, labels: list[str], toxicity: str = "nontoxic"
) -> pd.DataFrame:
    """Mean probability of every label per identity term, on one toxicity class.

    On nontoxic sentences this shows which label an identity mention pushes up.
    """
    keep = (templates["toxicity"] == toxicity).to_numpy()
    df = templates.loc[keep, GROUP_KEY].reset_index(drop=True)
    df[labels] = probs[keep]
    return df.groupby(GROUP_KEY, sort=True)[labels].mean().reset_index()
