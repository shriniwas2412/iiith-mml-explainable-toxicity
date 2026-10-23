"""Train a one-layer model built on the spot: checks the loop, not accuracy."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from toxicity import LABELS, load_config  # noqa: E402
from toxicity.data import multilabel_split  # noqa: E402
from toxicity.predictions import load_predictions  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_model_dir(tmp_path, comments):
    """A random DistilBERT backbone and a word-level tokenizer, saved locally (no download)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    words = sorted({word for text in comments["comment_text"] for word in text.lower().split()})
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words])}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    folder = tmp_path / "tiny-distilbert"
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(folder)
    config = DistilBertConfig(
        vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=64
    )
    DistilBertModel(config).save_pretrained(folder)
    return folder


@pytest.mark.parametrize("loss", ["bce", "weighted_bce", "focal"])
def test_training_writes_a_loadable_model_and_predictions(tmp_path, comments, tiny_model_dir, loss):
    from toxicity.scoring import TransformerScorer
    from toxicity.train import train_model

    cfg = load_config()
    cfg["model"].update(name=str(tiny_model_dir), max_length=32)
    cfg["train"].update(loss=loss, epochs=2, batch_size=32, eval_batch_size=64, lr=1e-3)
    train, val = multilabel_split(comments, 0.25, seed=0)
    out = tmp_path / "run"

    result = train_model(train, val, cfg, out, device="cpu")

    history = json.loads((out / "history.json").read_text())
    assert history["best_epoch"] == result["best_epoch"] in (1, 2)
    assert len(history["history"]) == 2
    assert all(np.isfinite(epoch["train_loss"]) for epoch in history["history"])
    assert (out / "run_config.json").exists()

    ids, y_true, y_prob = load_predictions(out / "val_predictions.csv")
    assert ids.tolist() == val["id"].tolist()
    assert y_prob.shape == (len(val), len(LABELS))
    assert ((y_prob >= 0) & (y_prob <= 1)).all()

    scorer = TransformerScorer(str(out / "model"), batch_size=16, max_length=32, device="cpu")
    texts = val["comment_text"].tolist()
    reloaded = scorer.predict_proba(texts)
    # The saved model reproduces the validation predictions of its best epoch.
    assert reloaded == pytest.approx(y_prob, abs=1e-5)
    # Batching by length must not change which row a prediction belongs to.
    shuffled = np.random.default_rng(0).permutation(len(texts))
    assert scorer.predict_proba([texts[i] for i in shuffled]) == pytest.approx(reloaded[shuffled], abs=1e-5)
