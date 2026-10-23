import pandas as pd

from toxicity.lexicon import mention_matrix, mentions_any

TERMS = pd.DataFrame(
    [
        ("dixon", "gender_sexuality", "gay", "gay"),
        ("dixon", "race_ethnicity", "african american", "african american"),
        ("india", "caste", "sc/st", "sc/st"),
        ("india", "caste", "khsatirya", "khsatirya"),
        ("india", "caste", "khsatirya", "kshatriya"),
    ],
    columns=["term_set", "axis", "identity", "pattern"],
)


def test_word_boundaries_case_and_multiword_terms():
    texts = ["Gay people are welcome", "gayety is not a term", "An African American author", "no terms"]
    hits = mention_matrix(texts, TERMS)
    assert hits[("dixon", "gay")].tolist() == [True, False, False, False]
    assert hits[("dixon", "african american")].tolist() == [False, False, True, False]
    assert mentions_any(texts, TERMS).tolist() == [True, False, True, False]


def test_slash_terms_and_alternate_spellings():
    hits = mention_matrix(["reserved for SC/ST students", "a kshatriya family", "scst"], TERMS)
    assert hits[("india", "sc/st")].tolist() == [True, False, False]
    assert hits[("india", "khsatirya")].tolist() == [False, True, False]
