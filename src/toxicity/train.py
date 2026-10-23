"""Fine-tune a transformer for multi-label toxicity with a plain PyTorch loop."""

from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import LABELS, TEXT_COL
from .losses import build_loss
from .predictions import save_predictions
from .scoring import pick_device, predict_proba_transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TokenizedComments(Dataset):
    def __init__(self, encodings, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {key: values[i] for key, values in self.encodings.items()}, self.labels[i]


def mean_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean PR-AUC over the labels that have at least one positive and one negative."""
    scores = [
        average_precision_score(y_true[:, j], y_prob[:, j])
        for j in range(y_true.shape[1])
        if 0 < y_true[:, j].sum() < len(y_true)
    ]
    return float(np.mean(scores)) if scores else float("nan")


def _optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    # Standard practice: no weight decay on biases and LayerNorm weights.
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay, skip = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (skip if name.endswith(no_decay) else decay).append(param)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": skip, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    out_dir: str | Path,
    device: str | None = None,
    log_every: int = 100,
) -> dict:
    """Train, evaluate on `val_df` after each epoch and keep the best epoch.

    Writes to `out_dir`: model/ (best weights + tokenizer), val_predictions.csv
    (from the best epoch), history.json and run_config.json. The best epoch is
    the one with the highest mean validation PR-AUC.
    """
    model_cfg, train_cfg = cfg["model"], cfg["train"]
    set_seed(cfg["seed"])
    device = pick_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)

    y_train = train_df[LABELS].to_numpy(dtype=np.float32)
    y_val = val_df[LABELS].to_numpy()
    loss_fn = build_loss(train_cfg["loss"], y_train, train_cfg).to(device)

    encodings = tokenizer(
        train_df[TEXT_COL].tolist(), truncation=True, max_length=model_cfg["max_length"]
    )

    def collate(items):
        features, labels = zip(*items)
        batch = tokenizer.pad(list(features), return_tensors="pt")
        batch["labels"] = torch.from_numpy(np.stack(labels))
        return batch

    loader = DataLoader(
        _TokenizedComments(encodings, y_train),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    total_steps = len(loader) * train_cfg["epochs"]
    optimizer = _optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(train_cfg["warmup_ratio"] * total_steps), total_steps
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_config = {
        "config": cfg,
        "device": str(device),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "torch": torch.__version__,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    history, best = [], {"epoch": 0, "val_pr_auc": -1.0}
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with autocast:
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f}", flush=True)

        val_prob = predict_proba_transformer(
            model, tokenizer, val_df[TEXT_COL].tolist(), train_cfg["eval_batch_size"], model_cfg["max_length"]
        )
        score = mean_pr_auc(y_val, val_prob)
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val_pr_auc": score})
        print(f"epoch {epoch} done: train_loss {running / len(loader):.4f} val_pr_auc {score:.4f}", flush=True)

        if score > best["val_pr_auc"] or epoch == 1:
            best = {"epoch": epoch, "val_pr_auc": score}
            model.save_pretrained(out_dir / "model")
            tokenizer.save_pretrained(out_dir / "model")
            save_predictions(out_dir / "val_predictions.csv", val_df, val_prob)

    result = {"best_epoch": best["epoch"], "best_val_pr_auc": best["val_pr_auc"], "history": history}
    (out_dir / "history.json").write_text(json.dumps(result, indent=2))
    return result
