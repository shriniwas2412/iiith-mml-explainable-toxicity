"""Find identity-term mentions in free text.

Uses the same term lists as the bias templates: Dixon et al. (2018) terms with
their axes from `resources/dixon_term_axes.csv`, and the Indian caste, religion
and region terms from Bhatt et al. (2022). Matching is case-insensitive and on
word boundaries, so "gay" matches "Gay people" but not "gayety".

Some terms are also ordinary words ("black", "old", "straight"), so a match
means the word appears, not that the comment is about that group.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .templates import load_india_terms

ROOT = Path(__file__).resolve().parents[2]
DIXON_AXES = ROOT / "resources" / "dixon_term_axes.csv"
INDIA_OVERRIDES = ROOT / "resources" / "india_term_overrides.csv"


def load_identity_terms(
    india_dir: str | Path = "data/external/india",
    dixon_axes: str | Path = DIXON_AXES,
    india_overrides: str | Path = INDIA_OVERRIDES,
) -> pd.DataFrame:
    """One row per (term_set, axis, identity, pattern) to search for.

    Indian terms with a corrected spelling are searched in both spellings.
    """
    dixon = pd.read_csv(dixon_axes).rename(columns={"term": "identity"})
    dixon["term_set"] = "dixon"
    dixon["pattern"] = dixon["identity"]
    rows = [dixon[["term_set", "axis", "identity", "pattern"]]]
    if Path(india_dir).exists():
        india = load_india_terms(india_dir, india_overrides)
        india["term_set"] = "india"
        spellings = pd.concat(
            [india.assign(pattern=india["identity"]), india.assign(pattern=india["surface"])]
        ).drop_duplicates(["identity", "pattern"])
        rows.append(spellings[["term_set", "axis", "identity", "pattern"]])
    return pd.concat(rows, ignore_index=True)


def _compile(pattern: str) -> re.Pattern:
    # \b fails next to "/" in terms like "sc/st", so use explicit non-letter guards.
    return re.compile(rf"(?<![a-z0-9]){re.escape(pattern.lower())}(?![a-z0-9])")


def mention_matrix(texts: Iterable[str], terms: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: one row per text, one column per (term_set, identity)."""
    lowered = pd.Series(list(texts), dtype=str).str.lower()
    keys = terms[["term_set", "identity"]].drop_duplicates().itertuples(index=False, name=None)
    out = {}
    for term_set, identity in keys:
        patterns = terms.loc[(terms["term_set"] == term_set) & (terms["identity"] == identity), "pattern"]
        hit = np.zeros(len(lowered), dtype=bool)
        for pattern in patterns:
            hit |= lowered.str.contains(_compile(pattern), regex=True).to_numpy()
        out[(term_set, identity)] = hit
    columns = pd.MultiIndex.from_tuples(list(out), names=["term_set", "identity"])
    return pd.DataFrame(np.column_stack(list(out.values())) if out else np.zeros((len(lowered), 0)), columns=columns)


def mentions_any(texts: Iterable[str], terms: pd.DataFrame) -> np.ndarray:
    """True for texts that mention at least one term."""
    return mention_matrix(texts, terms).any(axis=1).to_numpy()
