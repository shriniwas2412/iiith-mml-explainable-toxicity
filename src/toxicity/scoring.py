"""One scoring interface for every model: texts in, six label probabilities out.

A scorer has `predict_proba(texts) -> array of shape (n, 6)` with columns in
`LABELS` order. Model spec strings:

    baseline:<path to model.joblib>
    hf:<directory saved by the training script>
    detoxify:<original | unbiased | multilingual>
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import LABELS


def pick_device(name: str | None = None):
    """Use the named device, else CUDA, else Apple MPS, else CPU."""
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_proba_transformer(
    model, tokenizer, texts: Sequence[str], batch_size: int = 128, max_length: int = 256
) -> np.ndarray:
    """Sigmoid probabilities for each text, returned in the input order.

    Texts are batched by length so short comments are not padded to the longest one.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()
    encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
    order = np.argsort([len(ids) for ids in encoded["input_ids"]], kind="stable")
    probs = np.zeros((len(order), model.config.num_labels), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = tokenizer.pad(
                {key: [values[i] for i in rows] for key, values in encoded.items()},
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            probs[rows] = torch.sigmoid(logits.float()).cpu().numpy()
    return probs


class BaselineScorer:
    """TF-IDF + logistic regression pipeline saved with joblib."""

    def __init__(self, path: str):
        import joblib

        # joblib files can run code when loaded: only load files you created.
        self.pipeline = joblib.load(path)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(list(texts)))


class TransformerScorer:
    """A model directory written by `toxicity.train.train_model`."""

    def __init__(self, model_dir: str, batch_size: int = 128, max_length: int = 256, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(pick_device(device))
        saved = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if saved != LABELS:
            raise ValueError(f"Model labels {saved} do not match {LABELS}")
        self.batch_size = batch_size
        self.max_length = max_length

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return predict_proba_transformer(self.model, self.tokenizer, texts, self.batch_size, self.max_length)


# Detoxify reports some labels under different names than the Kaggle columns.
_DETOXIFY_KEYS = {
    "toxic": ("toxicity", "toxic"),
    "severe_toxic": ("severe_toxicity", "severe_toxic"),
    "obscene": ("obscene",),
    "threat": ("threat",),
    "insult": ("insult",),
    "identity_hate": ("identity_attack", "identity_hate"),
}


class DetoxifyScorer:
    """Pretrained Detoxify checkpoints, used as external comparison models."""

    def __init__(self, variant: str, batch_size: int = 64, device: str | None = None):
        from detoxify import Detoxify

        self.model = Detoxify(variant, device=str(pick_device(device)))
        self.batch_size = batch_size

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            result = self.model.predict(texts[start : start + self.batch_size])
            columns = []
            for label in LABELS:
                key = next((k for k in _DETOXIFY_KEYS[label] if k in result), None)
                if key is None:
                    raise KeyError(f"Detoxify output has no score for '{label}': {sorted(result)}")
                columns.append(np.asarray(result[key], dtype=np.float32))
            chunks.append(np.column_stack(columns))
        return np.vstack(chunks)


def load_scorer(spec: str, device: str | None = None):
    """Build a scorer from a `kind:value` spec string (see module docstring)."""
    kind, _, value = spec.partition(":")
    if not value:
        raise ValueError(f"Model spec must look like 'kind:value', got '{spec}'")
    if kind == "baseline":
        return BaselineScorer(value)
    if kind == "hf":
        return TransformerScorer(value, device=device)
    if kind == "detoxify":
        return DetoxifyScorer(value, device=device)
    raise ValueError(f"Unknown model kind '{kind}'. Use baseline, hf or detoxify.")
