"""Data analysis beyond the basic EDA: shift, style, noise, duplicates and identity terms.

Writes tables and figures to reports/data_analysis/. Needs scripts 01, 03 and 07.

    python scripts/10_data_analysis.py

1. Label rates in train vs the scored test set, and labels per labelled comment.
2. Length and style (capital letters, "!", repeated letters) by label.
3. Noise and artifacts (URLs, IPs, wiki markup, signatures, ...) in toxic vs clean
   comments, to decide which preprocessing is worth doing.
4. Exact duplicate texts within train and across the train / val / test files.
5. Identity terms in the data (Vaidya et al. 2020, Table 2): how often each
   Dixon and Indian term appears, and how often comments mentioning it are toxic.
6. Identity axes vs labels (Davidson et al. 2017): among comments that mention a
   group, how often the abuse is labelled identity_hate rather than only insult.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from toxicity import LABELS, load_config
from toxicity.config import TEXT_COL
from toxicity.data import load_test, read_split
from toxicity.lexicon import load_identity_terms, mention_matrix
from toxicity.plots import SERIES, hbar, plt, save, style

ARTIFACTS = {
    "url": r"https?://|www\.",
    "ip_address": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "html_tag": r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?>",
    "wiki_markup": r"\[\[|\]\]|\{\{|\}\}|==",
    "signature_timestamp": r"\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4}|\(UTC\)",
    "user_or_talk_link": r"(?i)\buser(?:[ _]talk)?:",
    "newline": r"\n",
    "repeated_letters_4plus": r"([A-Za-z])\1{3,}",
    "non_ascii_char": r"[^\x00-\x7f]",
}


# The repeated-letter pattern needs a capture group for its back-reference.
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


def shouting(text: pd.Series) -> pd.Series:
    """At least 10 letters and at least 70% of them upper case."""
    letters = text.str.count(r"[A-Za-z]")
    upper = text.str.count(r"[A-Z]")
    return (letters >= 10) & (upper / letters.clip(lower=1) >= 0.7)


def mostly_non_latin(text: pd.Series) -> pd.Series:
    """More than 30% of the letter-like characters are outside ASCII: a proxy for non-English."""
    letters = text.str.count(r"[^\W\d_]")
    non_ascii = text.str.count(r"[^\x00-\x7f]")
    return (letters >= 5) & (non_ascii / letters.clip(lower=1) > 0.3)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=cfg["data"]["raw_dir"])
    parser.add_argument("--processed", default=cfg["data"]["processed_dir"])
    parser.add_argument("--india", default="data/external/india")
    parser.add_argument("--min-mentions", type=int, default=30, help="Minimum mentions for a term to be charted")
    parser.add_argument("--out", default="reports/data_analysis")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    raw = pd.read_csv(Path(args.raw) / "train.csv", keep_default_na=False)
    text = raw[TEXT_COL].astype(str)
    y = raw[LABELS]
    any_label = y.sum(axis=1) > 0
    toxic = raw["toxic"] == 1
    facts = {"train_comments": len(raw), "share_with_any_label": float(any_label.mean())}

    # 1. Train vs test label rates, labels per labelled comment
    test = load_test(Path(args.raw) / "test.csv", Path(args.raw) / "test_labels.csv")
    rates = pd.DataFrame({"train": y.mean(), "test_scored": test[LABELS].mean()})
    rates["test_over_train"] = rates["test_scored"] / rates["train"]
    rates.loc["any_label"] = [any_label.mean(), (test[LABELS].sum(axis=1) > 0).mean(), np.nan]
    rates.loc["any_label", "test_over_train"] = rates.loc["any_label", "test_scored"] / rates.loc["any_label", "train"]
    rates.to_csv(out / "label_rates_train_vs_test.csv")
    cardinality = y[any_label].sum(axis=1).value_counts().sort_index()
    cardinality.rename("comments").to_frame().assign(share=lambda d: d["comments"] / d["comments"].sum()).to_csv(
        out / "labels_per_labelled_comment.csv"
    )

    fig, ax = plt.subplots(figsize=(7, 3.6))
    pos = np.arange(len(LABELS))
    ax.bar(pos - 0.2, rates.loc[LABELS, "train"], width=0.38, color=SERIES[0], label="train.csv")
    ax.bar(pos + 0.2, rates.loc[LABELS, "test_scored"], width=0.38, color=SERIES[1], label="scored test")
    ax.set_xticks(pos, LABELS)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Label rates: train vs scored test")
    ax.legend()
    save(fig, out / "fig1_label_rates_train_vs_test.png")

    # 2. Length and style by label
    words = text.str.split().str.len()
    style_cols = {
        "median_words": words,
        "share_shouting": shouting(text),
        "mean_exclamations": text.str.count("!"),
        "share_repeated_letters": text.str.contains(ARTIFACTS["repeated_letters_4plus"], regex=True),
    }
    groups = {label: y[label] == 1 for label in LABELS}
    groups["no_label"] = ~any_label
    rows = []
    for name, mask in groups.items():
        row = {"group": name, "comments": int(mask.sum())}
        for col, series in style_cols.items():
            row[col] = float(series[mask].median() if col == "median_words" else series[mask].mean())
        rows.append(row)
    style_table = pd.DataFrame(rows).set_index("group")
    style_table.to_csv(out / "length_and_style_by_label.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    order = ["no_label", *LABELS]
    hbar(axes[0], order, style_table.loc[order, "median_words"].tolist(), fmt="{:.0f}")
    axes[0].set_title("Median words per comment")
    hbar(axes[1], order, style_table.loc[order, "share_shouting"].tolist(), color=SERIES[1])
    axes[1].set_title("Share written mostly in CAPITALS")
    axes[1].set_yticks([])
    save(fig, out / "fig2_length_and_shouting_by_label.png")

    # 3. Noise and artifacts in toxic vs clean comments
    flags = {name: text.str.contains(pattern, regex=True) for name, pattern in ARTIFACTS.items()}
    flags["shouting"] = shouting(text)
    flags["mostly_non_latin"] = mostly_non_latin(text)
    artifacts = pd.DataFrame(
        {
            "all": {k: v.mean() for k, v in flags.items()},
            "toxic": {k: v[toxic].mean() for k, v in flags.items()},
            "clean": {k: v[~any_label].mean() for k, v in flags.items()},
        }
    )
    artifacts["toxic_over_clean"] = artifacts["toxic"] / artifacts["clean"].replace(0, np.nan)
    artifacts = artifacts.sort_values("all", ascending=False)
    artifacts.to_csv(out / "artifacts_toxic_vs_clean.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = artifacts.index.tolist()
    pos = np.arange(len(names))
    ax.barh(pos - 0.2, artifacts["clean"], height=0.38, color=SERIES[0], label="no label")
    ax.barh(pos + 0.2, artifacts["toxic"], height=0.38, color=SERIES[1], label="toxic")
    ax.set_yticks(pos, names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Share of comments with each artifact")
    ax.legend(loc="lower right")
    save(fig, out / "fig3_artifacts_toxic_vs_clean.png")

    # 4. Duplicates
    processed = Path(args.processed)
    splits = {name: read_split(processed / f"{name}.csv")[TEXT_COL] for name in ("train", "val", "test")}
    sets = {name: set(s) for name, s in splits.items()}
    duplicates = {
        "duplicate_texts_within_raw_train": int(text.duplicated().sum()),
        "val_rows_whose_text_is_in_train": int(splits["val"].isin(sets["train"]).sum()),
        "test_rows_whose_text_is_in_train": int(splits["test"].isin(sets["train"]).sum()),
        "test_rows_whose_text_is_in_val": int(splits["test"].isin(sets["val"]).sum()),
    }
    dup_rows = raw[text.duplicated(keep=False)]
    conflicting = dup_rows.groupby(TEXT_COL)[LABELS].nunique().gt(1).any(axis=1)
    duplicates["duplicate_texts_with_conflicting_labels"] = int(conflicting.sum())
    (out / "duplicates.json").write_text(json.dumps(duplicates, indent=2))

    # 5. Identity terms in the data (Vaidya et al. 2020, Table 2 for this dataset)
    terms = load_identity_terms(args.india)
    hits = mention_matrix(text, terms)
    overall_toxic = toxic.mean()
    rows = []
    for (term_set, identity) in hits.columns:
        m = hits[(term_set, identity)].to_numpy()
        axis = terms.loc[(terms["term_set"] == term_set) & (terms["identity"] == identity), "axis"].iloc[0]
        n = int(m.sum())
        rows.append(
            {
                "term_set": term_set,
                "axis": axis,
                "identity": identity,
                "mentions": n,
                "share_toxic": float(toxic[m].mean()) if n else np.nan,
                "share_identity_hate": float(raw.loc[m, "identity_hate"].mean()) if n else np.nan,
                "share_non_toxic": float(1 - toxic[m].mean()) if n else np.nan,
                "toxic_lift": float(toxic[m].mean() / overall_toxic) if n else np.nan,
            }
        )
    identity = pd.DataFrame(rows).sort_values(["term_set", "mentions"], ascending=[True, False])
    identity.to_csv(out / "identity_terms_in_train.csv", index=False)
    facts["overall_share_toxic"] = float(overall_toxic)
    facts["comments_mentioning_any_dixon_term"] = int(hits["dixon"].any(axis=1).sum())
    if "india" in hits.columns.get_level_values(0):
        facts["comments_mentioning_any_india_term"] = int(hits["india"].any(axis=1).sum())
        facts["india_terms_never_mentioned"] = int((identity.query("term_set == 'india'")["mentions"] == 0).sum())
        facts["india_terms_total"] = int((identity["term_set"] == "india").sum())

    # Terms on both lists (e.g. muslim, sikh) are charted once, as "both".
    in_both = set(identity.loc[identity["term_set"] == "dixon", "identity"]) & set(
        identity.loc[identity["term_set"] == "india", "identity"]
    )
    charted = identity.drop_duplicates("identity").copy()
    charted.loc[charted["identity"].isin(in_both), "term_set"] = "both"
    charted = charted[charted["mentions"] >= args.min_mentions].sort_values("share_toxic", ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    color_of = {"dixon": SERIES[0], "india": SERIES[2], "both": SERIES[1]}
    labels = [f"{r.identity} ({r.mentions:,})" for r in charted.itertuples()]
    hbar(ax, labels, charted["share_toxic"].tolist(), color=[color_of[s] for s in charted["term_set"]],
         reference=overall_toxic, reference_label="all comments")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"Share toxic among comments mentioning each term (≥{args.min_mentions} mentions)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_of.values()]
    ax.legend(handles, ["Dixon (Western) list", "Indian list", "on both lists"], loc="lower right")
    save(fig, out / "fig4_identity_terms_toxic_rate.png")

    india = identity[identity["term_set"] == "india"].sort_values("mentions", ascending=False)
    if len(india):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        hbar(ax, india["identity"].tolist(), india["mentions"].tolist(), color=SERIES[2], fmt="{:,.0f}")
        ax.set_title("How often each Indian term appears in train.csv")
        save(fig, out / "fig5_india_term_mentions.png")

    # 6. Identity axes vs labels (Davidson et al. 2017 on this dataset)
    rows = []
    for (term_set, axis), group in terms.groupby(["term_set", "axis"]):
        ids = group["identity"].unique()
        m = hits[term_set][ids].any(axis=1).to_numpy()
        tox = m & toxic.to_numpy()
        rows.append(
            {
                "term_set": term_set,
                "axis": axis,
                "comments": int(m.sum()),
                "share_toxic": float(toxic[m].mean()) if m.any() else np.nan,
                "share_identity_hate": float(raw.loc[m, "identity_hate"].mean()) if m.any() else np.nan,
                "share_insult": float(raw.loc[m, "insult"].mean()) if m.any() else np.nan,
                "toxic_ones_labelled_identity_hate": float(raw.loc[tox, "identity_hate"].mean()) if tox.any() else np.nan,
            }
        )
    axes_table = pd.DataFrame(rows).sort_values("toxic_ones_labelled_identity_hate", ascending=False)
    axes_table.to_csv(out / "identity_axes_vs_labels.csv", index=False)

    shown = axes_table[axes_table["comments"] >= args.min_mentions]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    hbar(
        ax,
        [f"{r.term_set}: {r.axis}" for r in shown.itertuples()],
        shown["toxic_ones_labelled_identity_hate"].tolist(),
        color=SERIES[1],
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Of the toxic comments mentioning a group, share labelled identity_hate")
    save(fig, out / "fig6_identity_hate_by_axis.png")

    (out / "facts.json").write_text(json.dumps(facts, indent=2))
    pd.set_option("display.width", 160)
    print(rates.round(4).to_string(), "\n")
    print(style_table.round(3).to_string(), "\n")
    print(artifacts.round(4).to_string(), "\n")
    print(json.dumps(duplicates, indent=2), "\n")
    print(identity.sort_values("mentions", ascending=False).head(15).round(3).to_string(index=False), "\n")
    print(axes_table.round(3).to_string(index=False), "\n")
    print(json.dumps(facts, indent=2))
    print(f"\nSaved to {out}/  (tables .csv/.json, figures fig*.png)", flush=True)



if __name__ == "__main__":
    main()
