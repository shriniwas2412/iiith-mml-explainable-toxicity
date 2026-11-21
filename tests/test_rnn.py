"""The attention RNN: fast checks of its pieces, and a slow end-to-end training run."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")  # toxicity.train, which the RNN trainer reuses, imports it

import torch  # noqa: E402

from toxicity import LABELS, load_config  # noqa: E402
from toxicity.data import multilabel_split  # noqa: E402
from toxicity.lexicon import axis_mentions  # noqa: E402
from toxicity.predictions import load_predictions  # noqa: E402
from toxicity.rnn import (  # noqa: E402
    AttentionPooling,
    AttentionRNN,
    build_embedding_matrix,
    build_vocab,
    encode,
    pad_batch,
    random_embedding_matrix,
)

TERMS = pd.DataFrame(
    [
        ("dixon", "gender_sexuality", "gay", "gay"),
        ("dixon", "religion", "muslim", "muslim"),
        ("india", "religion", "hindu", "hindu"),
    ],
    columns=["term_set", "axis", "identity", "pattern"],
)


def test_attention_weights_sum_to_one_and_ignore_padding():
    attention = AttentionPooling(8)
    states = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    pooled, weights = attention(states, mask)
    assert pooled.shape == (2, 8)
    assert weights.sum(dim=1).detach() == pytest.approx(torch.ones(2), abs=1e-5)
    assert weights[0, 3:].detach().tolist() == [0.0, 0.0]
    # Changing a padded state must not change the pooled vector.
    changed = states.clone()
    changed[0, 3:] += 100
    assert attention(changed, mask)[0][0].detach() == pytest.approx(pooled[0].detach(), abs=1e-5)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_forward_shapes_with_and_without_identity_head(cell):
    vocab = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
    emb = random_embedding_matrix(vocab, dim=12)
    ids, lengths = pad_batch([[2, 3, 2], [3]])
    single = AttentionRNN(emb, hidden_size=6, cell=cell)
    logits, identity = single(ids, lengths)
    assert logits.shape == (2, len(LABELS)) and identity is None
    multi = AttentionRNN(emb, hidden_size=6, cell=cell, num_identity=3)
    logits, identity = multi(ids, lengths)
    assert logits.shape == (2, len(LABELS)) and identity.shape == (2, 3)


def test_prediction_does_not_depend_on_padding_length():
    emb = random_embedding_matrix({"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}, dim=12)
    model = AttentionRNN(emb, hidden_size=6, dropout=0.0).eval()
    alone, a_len = pad_batch([[2, 3]])
    padded, p_len = pad_batch([[2, 3], [3, 2, 2, 3, 3, 2]])
    with torch.no_grad():
        assert model(padded, p_len)[0][0] == pytest.approx(model(alone, a_len)[0][0], abs=1e-5)


def test_frozen_embeddings_do_not_train():
    emb = random_embedding_matrix({"<pad>": 0, "<unk>": 1, "a": 2}, dim=4)
    assert not AttentionRNN(emb, hidden_size=4, freeze_embeddings=True).embedding.weight.requires_grad
    assert AttentionRNN(emb, hidden_size=4, freeze_embeddings=False).embedding.weight.requires_grad


def test_vocab_encoding_and_embedding_file(tmp_path):
    vocab = build_vocab(["The cat sat", "the cat ran", "a dog"], max_vocab=10, min_count=2)
    assert set(vocab) == {"<pad>", "<unk>", "the", "cat"}
    assert encode("The dog cat", vocab, 10) == [vocab["the"], 1, vocab["cat"]]
    assert encode("", vocab, 10) == [1]
    vectors = tmp_path / "tiny.vec"
    vectors.write_text("2 3\nthe 1 2 3\ncat 4 5 6\nzebra 7 8 9\n")  # header line is skipped
    matrix, coverage = build_embedding_matrix(vocab, vectors, dim=3)
    assert matrix[vocab["cat"]].tolist() == [4.0, 5.0, 6.0]
    assert matrix[0].tolist() == [0.0, 0.0, 0.0]
    assert coverage == 1.0


def test_axis_mentions_groups_terms_by_axis():
    hits = axis_mentions(["Gay and Muslim people", "a hindu temple", "nothing here"], TERMS)
    assert list(hits.columns) == ["dixon:gender_sexuality", "dixon:religion", "india:religion"]
    assert hits["dixon:religion"].tolist() == [True, False, False]
    assert hits.any(axis=1).tolist() == [True, True, False]


@pytest.mark.slow
@pytest.mark.parametrize("multitask", [False, True])
def test_training_writes_a_loadable_model_and_predictions(tmp_path, comments, multitask):
    from toxicity.rnn import train_rnn
    from toxicity.scoring import load_scorer

    cfg = load_config()
    cfg["rnn"].update(embedding="random", embedding_dim=16, hidden_size=8, epochs=2, batch_size=64, max_length=32)
    cfg["train"].update(loss="weighted_bce", eval_batch_size=64)
    train, val = multilabel_split(comments, 0.25, seed=0)
    vocab = build_vocab(train["comment_text"].tolist(), min_count=1)
    emb = random_embedding_matrix(vocab, 16)
    weights = np.where(train[LABELS].sum(axis=1) == 0, 3.0, 1.0)
    targets = np.stack([(train["id"].str[-1].astype(int) % 2 == k).to_numpy() for k in range(2)], axis=1) if multitask else None
    out = tmp_path / "run"

    result = train_rnn(
        train, val, cfg, out, vocab, emb, device="cpu", sample_weight=weights,
        identity_targets=targets, identity_names=["a", "b"] if multitask else None,
    )

    history = json.loads((out / "history.json").read_text())
    assert history["best_epoch"] == result["best_epoch"] in (1, 2)
    assert all(np.isfinite(e["train_loss"]) for e in history["history"])
    config = json.loads((out / "run_config.json").read_text())
    assert config["multitask"] is multitask
    assert config["upweighted_rows"] == int((weights != 1.0).sum())

    ids, _, y_prob = load_predictions(out / "val_predictions.csv")
    assert ids.tolist() == val["id"].tolist()
    assert y_prob.shape == (len(val), len(LABELS))

    scorer = load_scorer(f"rnn:{out / 'model'}", device="cpu")
    texts = val["comment_text"].tolist()
    reloaded = scorer.predict_proba(texts)
    # The saved model reproduces the validation predictions of its best epoch.
    assert reloaded == pytest.approx(y_prob, abs=1e-5)
    shuffled = np.random.default_rng(0).permutation(len(texts))
    assert scorer.predict_proba([texts[i] for i in shuffled]) == pytest.approx(reloaded[shuffled], abs=1e-5)
