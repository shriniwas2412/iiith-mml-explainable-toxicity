"""Integrated Gradients on a tiny random DistilBERT classifier built on the spot (no download)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from toxicity import LABELS  # noqa: E402
from toxicity.explain_transformer import IntegratedGradients, merge_wordpieces  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SENTENCES = ["you are a great person", "i am a gay person", "stupid idiot you loser"]
VOCAB_WORDS = sorted({w for s in SENTENCES for w in s.split()} | {"unbelievable", "hug", "muslim", "dalit"})


@pytest.fixture
def tiny_classifier(tmp_path):
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    # "un" + "##able" lets the tests check word merging.
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *VOCAB_WORDS, "un", "##believable", "##able", "."]
    vocab = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    folder = tmp_path / "model"
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(folder)
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=2, n_heads=2, max_position_embeddings=64,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    )
    DistilBertForSequenceClassification(config).save_pretrained(folder)
    return folder


@pytest.fixture
def ig(tiny_classifier):
    from toxicity.scoring import TransformerScorer

    scorer = TransformerScorer(str(tiny_classifier), max_length=32, device="cpu")
    return IntegratedGradients(scorer.model, scorer.tokenizer, max_length=32, n_steps=200)


def test_completeness(ig):
    for text in SENTENCES:
        results = ig.explain(text, ["toxic", "threat"])
        for r in results.values():
            diff = r["logit"] - r["baseline_logit"]
            assert r["token_attribution"].sum() == pytest.approx(diff, abs=2e-2)
            assert r["delta"] == pytest.approx(r["token_attribution"].sum() - diff)
            # Special tokens are identical in the input and the baseline: no attribution.
            assert r["token_attribution"][0] == pytest.approx(0, abs=1e-6)
            assert r["token_attribution"][-1] == pytest.approx(0, abs=1e-6)


def test_more_steps_shrink_the_error(ig):
    coarse = IntegratedGradients(ig.model, ig.tokenizer, 32, n_steps=4).explain(SENTENCES[0])["toxic"]
    fine = ig.explain(SENTENCES[0])["toxic"]
    assert fine["abs_delta"] <= coarse["abs_delta"] + 1e-6


def test_step_batching_and_mask_baseline(ig):
    whole = ig.explain(SENTENCES[1])["toxic"]["attribution"]
    chunked = IntegratedGradients(ig.model, ig.tokenizer, 32, n_steps=200, step_batch=30).explain(SENTENCES[1])["toxic"]
    assert chunked["attribution"] == pytest.approx(whole, abs=1e-4)
    masked = IntegratedGradients(ig.model, ig.tokenizer, 32, n_steps=200, baseline="mask").explain(SENTENCES[1])["toxic"]
    assert masked["token_attribution"].sum() == pytest.approx(masked["logit"] - masked["baseline_logit"], abs=2e-2)


def test_wordpieces_are_merged_into_words(ig):
    r = ig.explain("Hug unable muslim.")["toxic"]
    assert r["words"] == ["Hug", "unable", "muslim", "."]
    pieces = [t for t in r["tokens"] if t not in ("[CLS]", "[SEP]")]
    assert pieces == ["hug", "un", "##able", "muslim", "."]
    # The word's attribution is the sum of its pieces.
    assert r["attribution"][1] == pytest.approx(r["token_attribution"][2] + r["token_attribution"][3])
    assert r["attribution"].sum() == pytest.approx(r["token_attribution"].sum())


def test_merge_wordpieces_unit():
    text = "unbelievable stuff"
    words, attr, spans = merge_wordpieces(
        np.array([0.0, 1.0, 2.0, 4.0, 0.0]), [None, 0, 0, 1, None], [(0, 0), (0, 2), (2, 12), (13, 18), (0, 0)], text
    )
    assert words == ["unbelievable", "stuff"]
    assert attr.tolist() == [3.0, 4.0]
    assert spans == [(0, 12), (13, 18)]


def test_invalid_settings(ig):
    with pytest.raises(ValueError):
        IntegratedGradients(ig.model, ig.tokenizer, baseline="zeros")
    with pytest.raises(ValueError):
        IntegratedGradients(ig.model, ig.tokenizer, n_steps=0)


@pytest.mark.slow
def test_cli_smoke_run(tmp_path, tiny_classifier, monkeypatch):
    spec = importlib.util.spec_from_file_location("explain_transformer_cli", ROOT / "scripts" / "17_explain_transformer.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    (run / "thresholds.json").write_text(json.dumps({l: 0.5 for l in LABELS}))
    test = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "comment_text": ["you are a great person", "stupid idiot you loser", "i am a gay person", "hug muslim"],
        "toxic": [0, 1, 0, 1],
    })
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "toxic": test["toxic"], "prob_toxic": [0.9, 0.1, 0.8, 0.2]}).to_csv(
        run / "test_predictions.csv", index=False
    )
    templates = pd.DataFrame({
        "text": ["hug muslim", "hug dalit", "hug gay", "hug muslim"],
        "toxicity": ["nontoxic"] * 4,
        "template": ["verb"] * 4,
        "frame": ["hug {IDENTITY}"] * 4,
        "identity": ["muslim", "dalit", "gay", "muslim"],
        "axis": ["religion", "caste", "sexual_orientation", "religion"],
        "term_set": ["dixon", "india", "dixon", "dixon"],
    })
    templates.to_csv(tmp_path / "templates.csv", index=False)

    out = tmp_path / "reports" / "tiny"
    monkeypatch.setattr(sys, "argv", [
        "17", "--model", str(tiny_classifier), "--thresholds", str(run / "thresholds.json"), "--out", str(out),
        "--test", str(tmp_path / "test.csv"), "--templates", str(tmp_path / "templates.csv"),
        "--steps", "20", "--max-length", "32", "--device", "cpu", "--max-examples", "3", "--per-set", "2",
    ])
    cli.main()

    for name in ["local_examples.md", "local_examples.html", "identity_term_attribution.csv",
                 "fig_identity_term_attribution.png", "summary.json"]:
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run_name"] == "tiny"
    assert summary["n_examples"] == 6
    assert np.isfinite(summary["mean_completeness_error"])
    table = pd.read_csv(out / "identity_term_attribution.csv")
    assert {"identity", "mean_attribution", "n"} <= set(table.columns)
    assert "completeness error" in (out / "local_examples.md").read_text()
