import pandas as pd
import pytest

from toxicity.templates import (
    PLACEHOLDER,
    build_template_set,
    dixon_identity_terms,
    extract_frames,
    load_dixon_templates,
    load_india_terms,
)

from conftest import MINI_TERMS


def test_identity_terms_are_read_from_the_file(dixon_file):
    assert dixon_identity_terms(load_dixon_templates(dixon_file)) == sorted(MINI_TERMS)


def test_frames_and_controls_are_separated(dixon_file):
    templates = load_dixon_templates(dixon_file)
    frames, controls = extract_frames(templates, MINI_TERMS)
    assert len(frames) == 10
    assert len(controls) == 2
    assert set(frames["frame"]) >= {f"I am {PLACEHOLDER}", f"Being {PLACEHOLDER} is great"}
    # "african american" must be replaced whole, not as "african" + " american".
    assert not frames["frame"].str.contains("american").any()


def test_incomplete_grid_is_rejected(dixon_file):
    templates = load_dixon_templates(dixon_file).iloc[1:]
    with pytest.raises(ValueError, match="frames x"):
        extract_frames(templates, MINI_TERMS)


def test_india_terms_and_spelling_override(india_dir, tmp_path):
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("term,surface,note\nkhsatirya,kshatriya,spelling\n")
    terms = load_india_terms(india_dir, overrides)
    assert terms.groupby("axis")["identity"].count().to_dict() == {"caste": 3, "region": 2, "religion": 2}
    row = terms.set_index("identity").loc["khsatirya"]
    assert row["surface"] == "kshatriya"
    assert terms.set_index("identity").loc["uttar pradeshi", "surface"] == "uttar pradeshi"


def test_template_set_is_a_full_grid(dixon_file, india_dir, tmp_path):
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("term,surface,note\nkhsatirya,kshatriya,spelling\n")
    axes = tmp_path / "axes.csv"
    axes.write_text("term,axis\ngay,gender_sexuality\nmuslim,religion\n")
    built = build_template_set(dixon_file, india_dir, overrides_path=overrides, dixon_axes_path=axes)

    sizes = built.groupby("term_set").size().to_dict()
    assert sizes == {"control": 2, "dixon": 10 * 4, "india": 10 * 7, "neutral": 10}
    assert not built["text"].str.contains(PLACEHOLDER, regex=False).any()
    assert built["text"].is_unique is False  # "muslim" is in both term lists
    assert set(built["toxicity"]) == {"toxic", "nontoxic"}

    # The original sentences are reproduced exactly.
    original = set(load_dixon_templates(dixon_file)["phrase"])
    assert set(built.loc[built["term_set"].isin(["dixon", "control"]), "text"]) == original

    # Override changes the printed word but not the term's identity key.
    fixed = built[(built["identity"] == "khsatirya") & (built["frame"] == f"I am {PLACEHOLDER}")]
    assert fixed["text"].tolist() == ["I am kshatriya"]

    dixon_axes = built[built["term_set"] == "dixon"].drop_duplicates("identity").set_index("identity")["axis"]
    assert dixon_axes["muslim"] == "religion"
    assert dixon_axes["african"] == "unassigned"
    assert (built.loc[built["term_set"] == "neutral", "identity"] == "person").all()


def test_name_frames_can_be_sampled(dixon_file, india_dir):
    built = build_template_set(dixon_file, india_dir, max_name_frames=1, seed=0)
    name_frames = built[(built["template"] == "name_adj") & (built["term_set"] == "neutral")]
    assert name_frames.groupby("toxicity").size().to_dict() == {"nontoxic": 1, "toxic": 1}
    other = built[(built["template"] != "name_adj") & (built["term_set"] == "neutral")]
    assert len(other) == 6


def test_round_trip_through_csv_keeps_empty_strings(dixon_file, india_dir, tmp_path):
    built = build_template_set(dixon_file, india_dir)
    built.to_csv(tmp_path / "t.csv", index=False)
    again = pd.read_csv(tmp_path / "t.csv", keep_default_na=False)
    assert (again.loc[again["term_set"] == "control", "identity"] == "").all()
