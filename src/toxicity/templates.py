"""Matched identity-term templates for the bias audit.

The English sentence templates released with Dixon et al. (2018) are short
phrases such as "You are a friendly muslim": a fixed frame filled with one of
50 identity terms, each marked toxic or nontoxic by construction. This module
recovers the frames and fills the same frames with other term lists (the
Indian caste, religion and region terms released with Bhatt et al., 2022), so
that two sentences differ only in the identity term.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PLACEHOLDER = "{IDENTITY}"
_SELF_ID_PREFIX = "I am "

INDIA_TERM_FILES = {
    "caste": "caste_idterms.tsv",
    "religion": "religion_idterms.tsv",
    "region": "region_idterms.tsv",
}

COLUMNS = ["text", "toxicity", "template", "frame", "identity", "axis", "term_set"]


def load_dixon_templates(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"template", "toxicity", "phrase"} - set(df.columns)
    if missing:
        raise ValueError(f"Template file is missing columns: {sorted(missing)}")
    return df


def dixon_identity_terms(templates: pd.DataFrame) -> list[str]:
    """The identity terms used in the file, read off the "I am <term>" rows."""
    rows = templates[(templates["template"] == "am_hate_adj") & (templates["toxicity"] == "nontoxic")]
    if rows.empty or not rows["phrase"].str.startswith(_SELF_ID_PREFIX).all():
        raise ValueError('Expected "I am <term>" rows under template am_hate_adj / nontoxic')
    return sorted(rows["phrase"].str[len(_SELF_ID_PREFIX) :].unique())


def extract_frames(templates: pd.DataFrame, terms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split phrases into (frames with a placeholder, control phrases with no identity term).

    Raises if the phrases are not a full frames x terms grid, since that would
    mean the identity terms were not located correctly.
    """
    longest_first = sorted(terms, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, longest_first)) + r")(?![\w-])")

    frames, controls = [], []
    for template, toxicity, phrase in templates[["template", "toxicity", "phrase"]].itertuples(index=False):
        matches = list(pattern.finditer(phrase))
        if not matches:
            controls.append((template, toxicity, phrase))
            continue
        last = matches[-1]
        frames.append((template, toxicity, phrase[: last.start()] + PLACEHOLDER + phrase[last.end() :]))

    frame_df = pd.DataFrame(frames, columns=["template", "toxicity", "frame"])
    unique = frame_df.drop_duplicates().reset_index(drop=True)
    if len(frame_df) != len(unique) * len(terms):
        raise ValueError(
            f"{len(frame_df)} identity phrases is not {len(unique)} frames x {len(terms)} terms"
        )
    control_df = pd.DataFrame(controls, columns=["template", "toxicity", "text"])
    return unique, control_df


def load_term_file(path: str | Path) -> list[str]:
    """Terms from a one-column TSV/CSV with a header row."""
    column = pd.read_csv(path, sep="\t", dtype=str).iloc[:, 0]
    return [term.strip() for term in column.dropna() if term.strip()]


def load_india_terms(directory: str | Path, overrides_path: str | Path | None = None) -> pd.DataFrame:
    """Indian identity terms by axis, with the form to print in sentences.

    `overrides_path` is a CSV (term, surface, note) for terms whose released
    spelling should be changed in the sentence text.
    """
    rows = [
        {"identity": term, "surface": term, "axis": axis}
        for axis, filename in INDIA_TERM_FILES.items()
        for term in load_term_file(Path(directory) / filename)
    ]
    terms = pd.DataFrame(rows)
    if overrides_path is not None:
        overrides = pd.read_csv(overrides_path, dtype=str).set_index("term")["surface"]
        terms["surface"] = terms["identity"].map(overrides).fillna(terms["surface"])
    return terms


def fill_frames(frames: pd.DataFrame, terms: pd.DataFrame, term_set: str) -> pd.DataFrame:
    """Every frame x every term. `terms` needs identity, surface and axis columns."""
    grid = frames.merge(terms, how="cross")
    grid["text"] = [
        frame.replace(PLACEHOLDER, surface) for frame, surface in zip(grid["frame"], grid["surface"])
    ]
    grid["term_set"] = term_set
    return grid[COLUMNS]


def build_template_set(
    dixon_path: str | Path,
    india_dir: str | Path,
    overrides_path: str | Path | None = None,
    dixon_axes_path: str | Path | None = None,
    neutral_terms: tuple[str, ...] = ("person",),
    max_name_frames: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the audit set: the same frames filled with four groups of terms.

    term_set values:
      dixon    the 50 identity terms from the original file
      india    caste, religion and region terms
      neutral  frames filled with a non-identity word, used as a reference
      control  the original occupation phrases, which contain no identity term

    `max_name_frames` keeps a random sample of the (very numerous)
    "<name> is a <adjective> <identity>" frames per toxicity class.
    """
    templates = load_dixon_templates(dixon_path)
    dixon_terms = dixon_identity_terms(templates)
    frames, controls = extract_frames(templates, dixon_terms)

    if max_name_frames is not None:
        is_name = frames["template"] == "name_adj"
        rng = np.random.default_rng(seed)
        kept = [
            group.iloc[np.sort(rng.choice(len(group), min(max_name_frames, len(group)), replace=False))]
            for _, group in frames[is_name].groupby("toxicity", sort=True)
        ]
        frames = pd.concat([frames[~is_name], *kept], ignore_index=True)

    dixon = pd.DataFrame({"identity": dixon_terms, "surface": dixon_terms, "axis": "unassigned"})
    if dixon_axes_path is not None:
        axes = pd.read_csv(dixon_axes_path, dtype=str).set_index("term")["axis"]
        dixon["axis"] = dixon["identity"].map(axes).fillna("unassigned")
    neutral = pd.DataFrame({"identity": list(neutral_terms), "surface": list(neutral_terms), "axis": "neutral"})

    control = controls.assign(frame="", identity="", axis="none", term_set="control")[COLUMNS]
    parts = [
        fill_frames(frames, dixon, "dixon"),
        fill_frames(frames, load_india_terms(india_dir, overrides_path), "india"),
        fill_frames(frames, neutral, "neutral"),
        control,
    ]
    return pd.concat(parts, ignore_index=True)
