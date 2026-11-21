"""Integrated Gradients for a Hugging Face sequence classifier, written directly in PyTorch.

For one comment and one label the attribution of input embedding x is

    IG(x) = (x - x0) * integral over alpha in [0, 1] of  d logit(x0 + alpha (x - x0)) / d x

where x0 is a baseline. The baseline keeps [CLS] and [SEP] and replaces every other token
by the [PAD] (or [MASK]) embedding, so the sequence length and the position embeddings are
unchanged. The integral is a midpoint Riemann sum over `n_steps` interpolated inputs, all
run through the model in one batch (or a few chunks). The model sees `inputs_embeds`, so
gradients are taken with respect to the word embeddings; position embeddings are added by
the model and are not attributed.

Attributions are per WordPiece token, summed over the embedding dimension, then summed per
word (words come from the tokenizer's own word ids, so punctuation is a word of its own).

Completeness: the token attributions add up to logit(input) - logit(baseline). The gap is
the integration error, reported for every explained text as `delta` and shrinks with more
steps. The target is the raw logit of one label, not the sigmoid probability.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import LABELS

BASELINES = ("pad", "mask")


def merge_wordpieces(
    token_attr: np.ndarray,
    word_ids: Sequence[int | None],
    offsets: Sequence[tuple[int, int]],
    text: str,
) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    """Sum token attributions into words.

    `word_ids` and `offsets` are the tokenizer's (None for special tokens, character span of
    each token). Returns the word strings cut from `text`, their summed attributions and their
    character spans. Special tokens belong to no word; their attribution is zero for the
    baselines used here because they are identical in the input and the baseline.
    """
    order: list[int] = []
    spans: dict[int, list[int]] = {}
    sums: dict[int, float] = {}
    for attr, wid, (start, end) in zip(token_attr, word_ids, offsets):
        if wid is None:
            continue
        if wid not in spans:
            order.append(wid)
            spans[wid] = [start, end]
            sums[wid] = 0.0
        spans[wid][0] = min(spans[wid][0], start)
        spans[wid][1] = max(spans[wid][1], end)
        sums[wid] += float(attr)
    words = [text[spans[w][0] : spans[w][1]] for w in order]
    return words, np.array([sums[w] for w in order]), [(spans[w][0], spans[w][1]) for w in order]


def span_attribution(spans: Sequence[tuple[int, int]], attr: np.ndarray, start: int, end: int) -> float:
    """Sum of the word attributions whose character span overlaps [start, end)."""
    return float(sum(a for (s, e), a in zip(spans, attr) if s < end and e > start))


class IntegratedGradients:
    """Integrated Gradients on the input embeddings of a sequence classification model."""

    def __init__(self, model, tokenizer, max_length: int = 256, n_steps: int = 50, baseline: str = "pad",
                 step_batch: int | None = None):
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}, got '{baseline}'")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is needed (word ids and character offsets).")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_steps = n_steps
        self.baseline = baseline
        # Interpolation steps per forward pass; all steps at once by default.
        self.step_batch = step_batch or n_steps
        self.device = next(model.parameters()).device
        fill = tokenizer.pad_token_id if baseline == "pad" else tokenizer.mask_token_id
        if fill is None:
            raise ValueError(f"Tokenizer has no {baseline} token")
        self.fill_id = fill

    def _encode(self, text: str):
        return self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_offsets_mapping=True,
            return_special_tokens_mask=True, return_tensors="pt",
        )

    def explain(self, text: str, labels: Sequence[str] = ("toxic",)) -> dict[str, dict]:
        """Attributions of one text to the logit of each label in `labels`.

        Returns {label: result}. Each result has: words, attribution (per word), spans,
        tokens, token_attribution, logit, baseline_logit, probability, delta (sum of
        attributions minus the logit difference) and abs_delta.
        """
        import torch

        enc = self._encode(text)
        input_ids = enc["input_ids"].to(self.device)
        special = enc["special_tokens_mask"].bool().to(self.device)
        attention = enc["attention_mask"].to(self.device)
        word_ids = enc.word_ids(0)
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        targets = [LABELS.index(label) for label in labels]

        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            x = embed(input_ids)  # (1, T, D)
            base_ids = torch.where(special, input_ids, torch.full_like(input_ids, self.fill_id))
            x0 = embed(base_ids)
            logit_x = self.model(inputs_embeds=x, attention_mask=attention).logits[0]
            logit_x0 = self.model(inputs_embeds=x0, attention_mask=attention).logits[0]

        # Midpoint rule: alpha_k = (k + 0.5) / n.
        alphas = (torch.arange(self.n_steps, device=self.device, dtype=x.dtype) + 0.5) / self.n_steps
        grad_sum = {t: torch.zeros_like(x) for t in targets}
        for start in range(0, self.n_steps, self.step_batch):
            a = alphas[start : start + self.step_batch].view(-1, 1, 1)
            point = (x0 + a * (x - x0)).detach().requires_grad_(True)
            logits = self.model(
                inputs_embeds=point, attention_mask=attention.expand(point.shape[0], -1)
            ).logits
            for i, t in enumerate(targets):
                (grad,) = torch.autograd.grad(
                    logits[:, t].sum(), point, retain_graph=i < len(targets) - 1
                )
                grad_sum[t] += grad.sum(dim=0, keepdim=True)

        results = {}
        for label, t in zip(labels, targets):
            avg = grad_sum[t] / self.n_steps
            token_attr = ((x - x0) * avg).sum(dim=-1)[0].detach().cpu().numpy().astype(np.float64)
            words, word_attr, spans = merge_wordpieces(token_attr, word_ids, offsets, text)
            diff = float(logit_x[t] - logit_x0[t])
            delta = float(token_attr.sum() - diff)
            results[label] = {
                "words": words,
                "attribution": word_attr,
                "spans": spans,
                "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                "token_attribution": token_attr,
                "logit": float(logit_x[t]),
                "baseline_logit": float(logit_x0[t]),
                "probability": float(torch.sigmoid(logit_x[t])),
                "delta": delta,
                "abs_delta": abs(delta),
            }
        return results
