"""Bi-GRU / Bi-LSTM with attention on pre-trained word vectors, after Vaidya et al. (2020).

The model reads GloVe or fastText embeddings, encodes the comment with a
bidirectional recurrent layer and pools it with additive attention. The
multi-task variant adds a second head that predicts which identity axes a
comment mentions. Jigsaw 2018 has no identity annotations, so those targets are
weak labels from the identity lexicon (see `toxicity.lexicon.axis_mentions`).
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .config import LABELS, TEXT_COL
from .losses import build_loss
from .predictions import save_predictions
from .scoring import pick_device
from .train import _optimizer, mean_pr_auc, set_seed

PAD, UNK = 0, 1

EMBEDDING_SOURCES = {
    # name: (download url, file inside the zip)
    "glove": ("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip", "glove.6B.300d.txt"),
    "fasttext": (
        "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        "wiki-news-300d-1M.vec",
    ),
}
DEFAULT_EMBEDDING_DIR = Path.home() / ".cache" / "toxicity" / "embeddings"

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase words (with simple contractions) and single punctuation marks."""
    return _TOKEN.findall(str(text).lower())


def build_vocab(texts: Sequence[str], max_vocab: int = 50000, min_count: int = 2) -> dict[str, int]:
    """Word -> id for the most frequent training words. 0 is padding, 1 is unknown."""
    counts = Counter(token for text in texts for token in tokenize(text))
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_count]
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(token, UNK) for token in tokenize(text)[:max_length]]
    return ids or [UNK]  # keep empty comments as one unknown token


def embedding_file(name: str, cache_dir: str | Path = DEFAULT_EMBEDDING_DIR) -> Path:
    """Path of the cached embedding zip, downloaded on first use (about 1 GB for GloVe)."""
    if name not in EMBEDDING_SOURCES:
        raise ValueError(f"Unknown embedding '{name}'. Use {', '.join(EMBEDDING_SOURCES)}.")
    url, _ = EMBEDDING_SOURCES[name]
    path = Path(cache_dir) / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {path} ...", flush=True)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def _open_vectors(path: Path, member: str | None):
    """Yield text lines of a vectors file, plain or inside a zip."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = member or next(n for n in archive.namelist() if n.endswith((".txt", ".vec")))
            with archive.open(member) as raw:
                yield from io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            yield from f


def build_embedding_matrix(
    vocab: dict[str, int], path: str | Path, dim: int = 300, member: str | None = None, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Embedding matrix (len(vocab), dim) and the share of vocabulary words found in the file.

    Words missing from the file get small random vectors with the same spread as
    the found ones; padding is all zeros. The file is streamed, so only the
    vocabulary's vectors are held in memory.
    """
    found = np.zeros((len(vocab), dim), dtype=np.float32)
    has_vector = np.zeros(len(vocab), dtype=bool)
    for line in _open_vectors(Path(path), member):
        word, _, rest = line.rstrip().partition(" ")
        idx = vocab.get(word)
        if idx is None or idx < 2 or has_vector[idx]:
            continue
        values = rest.split(" ")
        if len(values) != dim:  # header line or a malformed row
            continue
        found[idx] = np.asarray(values, dtype=np.float32)
        has_vector[idx] = True
    rng = np.random.default_rng(seed)
    std = float(found[has_vector].std()) if has_vector.any() else 0.1
    matrix = found
    missing = ~has_vector
    missing[PAD] = False
    matrix[missing] = rng.normal(0.0, std, size=(int(missing.sum()), dim)).astype(np.float32)
    coverage = float(has_vector[2:].mean()) if len(vocab) > 2 else 0.0
    return matrix, coverage


def random_embedding_matrix(vocab: dict[str, int], dim: int = 300, seed: int = 0) -> np.ndarray:
    """Untrained vectors, for tests that must not download anything."""
    matrix = np.random.default_rng(seed).normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD] = 0.0
    return matrix


class AttentionPooling(nn.Module):
    """Additive attention over time steps: softmax(v . tanh(W h_t)), padding masked out."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """states (B, T, H), mask (B, T) True for real tokens -> (pooled (B, H), weights (B, T))."""
        scores = self.score(torch.tanh(self.proj(states))).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        return (weights.unsqueeze(-1) * states).sum(dim=1), weights


class AttentionRNN(nn.Module):
    """Embedding -> bidirectional GRU/LSTM -> attention -> toxicity logits (+ identity logits)."""

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        hidden_size: int = 128,
        cell: str = "gru",
        dropout: float = 0.3,
        num_labels: int = len(LABELS),
        num_identity: int = 0,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"Unknown cell '{cell}'. Use gru or lstm.")
        weights = torch.as_tensor(embeddings, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_embeddings, padding_idx=PAD)
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(weights.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.attention = AttentionPooling(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.toxicity_head = nn.Linear(2 * hidden_size, num_labels)
        # The identity head only exists in the multi-task variant.
        self.identity_head = nn.Linear(2 * hidden_size, num_identity) if num_identity else None

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor):
        """Return (toxicity logits (B, labels), identity logits (B, axes) or None)."""
        mask = ids != PAD
        packed = pack_padded_sequence(
            self.dropout(self.embedding(ids)), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        states, _ = self.rnn(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=ids.shape[1])
        pooled, _ = self.attention(states, mask)
        pooled = self.dropout(pooled)
        identity = self.identity_head(pooled) if self.identity_head is not None else None
        return self.toxicity_head(pooled), identity


class _EncodedComments(Dataset):
    def __init__(self, ids: list[list[int]], labels: np.ndarray, weights: np.ndarray, identity: np.ndarray):
        self.ids, self.labels, self.weights, self.identity = ids, labels, weights, identity

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        return self.ids[i], self.labels[i], self.weights[i], self.identity[i]


def pad_batch(sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad id lists to the longest one; returns (ids (B, T), lengths (B,))."""
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    ids = torch.full((len(sequences), int(lengths.max())), PAD, dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, lengths


def predict_proba_rnn(
    model: AttentionRNN, vocab: dict[str, int], texts: Sequence[str], batch_size: int = 256, max_length: int = 200
) -> np.ndarray:
    """Sigmoid toxicity probabilities for each text, returned in the input order."""
    device = next(model.parameters()).device
    model.eval()
    encoded = [encode(text, vocab, max_length) for text in texts]
    order = np.argsort([len(e) for e in encoded], kind="stable")
    probs = np.zeros((len(encoded), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            ids, lengths = pad_batch([encoded[i] for i in rows])
            logits, _ = model(ids.to(device), lengths)
            probs[rows] = torch.sigmoid(logits.float()).cpu().numpy()
    return probs


def model_settings(cfg: dict, num_identity: int, vocab_size: int, embedding_dim: int) -> dict:
    rnn_cfg = cfg["rnn"]
    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_size": rnn_cfg["hidden_size"],
        "cell": rnn_cfg["cell"],
        "dropout": rnn_cfg["dropout"],
        "num_identity": num_identity,
        "max_length": rnn_cfg["max_length"],
    }


def save_rnn(model: AttentionRNN, vocab: dict[str, int], settings: dict, model_dir: str | Path) -> None:
    """Weights (embeddings included), vocabulary and architecture settings, in one folder."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (model_dir / "vocab.json").write_text(json.dumps(vocab))
    (model_dir / "rnn_config.json").write_text(json.dumps(settings, indent=2))


def load_rnn(model_dir: str | Path, device=None) -> tuple[AttentionRNN, dict[str, int], dict]:
    model_dir = Path(model_dir)
    settings = json.loads((model_dir / "rnn_config.json").read_text())
    vocab = json.loads((model_dir / "vocab.json").read_text())
    model = AttentionRNN(
        np.zeros((settings["vocab_size"], settings["embedding_dim"]), dtype=np.float32),
        hidden_size=settings["hidden_size"],
        cell=settings["cell"],
        dropout=settings["dropout"],
        num_identity=settings["num_identity"],
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return model.to(pick_device(device)), vocab, settings


def train_rnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    out_dir: str | Path,
    vocab: dict[str, int],
    embeddings: np.ndarray,
    device: str | None = None,
    log_every: int = 50,
    sample_weight: np.ndarray | None = None,
    identity_targets: np.ndarray | None = None,
    identity_names: list[str] | None = None,
) -> dict:
    """Train, evaluate on `val_df` after each epoch and keep the best epoch.

    Writes to `out_dir`: model/ (best weights, vocabulary, settings),
    val_predictions.csv (from the best epoch), history.json and run_config.json.
    The best epoch is the one with the highest mean validation PR-AUC, as in
    `toxicity.train.train_model`.

    `sample_weight` scales each comment's toxicity loss (Vaidya et al., 2020).
    Passing `identity_targets` (one 0/1 column per identity axis, one row per
    training comment) turns on the multi-task head; its loss is a plain BCE
    scaled by `rnn.aux_weight` and added to the toxicity loss.
    """
    rnn_cfg, train_cfg = cfg["rnn"], cfg["train"]
    set_seed(cfg["seed"])
    device = pick_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multitask = identity_targets is not None
    num_identity = identity_targets.shape[1] if multitask else 0
    model = AttentionRNN(
        embeddings,
        hidden_size=rnn_cfg["hidden_size"],
        cell=rnn_cfg["cell"],
        dropout=rnn_cfg["dropout"],
        num_identity=num_identity,
        freeze_embeddings=rnn_cfg["freeze_embeddings"],
    ).to(device)
    settings = model_settings(cfg, num_identity, len(vocab), embeddings.shape[1])

    y_train = train_df[LABELS].to_numpy(dtype=np.float32)
    y_val = val_df[LABELS].to_numpy()
    weights = np.ones(len(train_df), dtype=np.float32) if sample_weight is None else sample_weight.astype(np.float32)
    aux_targets = (
        identity_targets.astype(np.float32) if multitask else np.zeros((len(train_df), 0), dtype=np.float32)
    )
    loss_fn = build_loss(train_cfg["loss"], y_train, train_cfg, reduction="none").to(device)
    aux_fn = nn.BCEWithLogitsLoss(reduction="none")

    encoded = [encode(text, vocab, rnn_cfg["max_length"]) for text in train_df[TEXT_COL]]

    def collate(items):
        seqs, labels, row_weights, identity = zip(*items)
        ids, lengths = pad_batch(seqs)
        return (
            ids,
            lengths,
            torch.from_numpy(np.stack(labels)),
            torch.tensor(row_weights, dtype=torch.float32),
            torch.from_numpy(np.stack(identity)),
        )

    loader = DataLoader(
        _EncodedComments(encoded, y_train, weights, aux_targets),
        batch_size=rnn_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    optimizer = _optimizer(model, rnn_cfg["lr"], train_cfg["weight_decay"])

    run_config = {
        "config": cfg,
        "model": "attention_rnn",
        "multitask": multitask,
        "identity_axes": identity_names or [],
        "model_settings": settings,
        "device": str(device),
        "train_rows": len(train_df),
        "upweighted_rows": int((weights != 1.0).sum()),
        "val_rows": len(val_df),
        "torch": torch.__version__,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    history, best = [], {"epoch": 0, "val_pr_auc": -1.0}
    for epoch in range(1, rnn_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for step, (ids, lengths, labels, row_weights, identity) in enumerate(loader, 1):
            ids, labels, row_weights = ids.to(device), labels.to(device), row_weights.to(device)
            logits, identity_logits = model(ids, lengths)
            per_row = loss_fn(logits.float(), labels).mean(dim=1)
            loss = (per_row * row_weights).sum() / row_weights.sum()
            if multitask:
                aux = aux_fn(identity_logits.float(), identity.to(device)).mean()
                loss = loss + rnn_cfg["aux_weight"] * aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f}", flush=True)

        val_prob = predict_proba_rnn(
            model, vocab, val_df[TEXT_COL].tolist(), train_cfg["eval_batch_size"], rnn_cfg["max_length"]
        )
        score = mean_pr_auc(y_val, val_prob)
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val_pr_auc": score})
        print(f"epoch {epoch} done: train_loss {running / len(loader):.4f} val_pr_auc {score:.4f}", flush=True)

        if score > best["val_pr_auc"] or epoch == 1:
            best = {"epoch": epoch, "val_pr_auc": score}
            save_rnn(model, vocab, settings, out_dir / "model")
            save_predictions(out_dir / "val_predictions.csv", val_df, val_prob)

    result = {"best_epoch": best["epoch"], "best_val_pr_auc": best["val_pr_auc"], "history": history}
    (out_dir / "history.json").write_text(json.dumps(result, indent=2))
    return result
