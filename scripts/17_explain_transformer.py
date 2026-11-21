"""Explain a fine-tuned transformer with Integrated Gradients (see toxicity/explain_transformer.py).

Run it right after training, on the model folder written by 05_train_transformer.py.
Writes to reports/explain/<run_name>/.

    python scripts/17_explain_transformer.py --model runs/distilbert_wbce_30k/model \
        --thresholds runs/distilbert_wbce_30k/thresholds.json --out reports/explain/distilbert_wbce_30k

1. Local: local_examples.md and local_examples.html. Template sentences (the ones used by
   16_explain_linear.py plus a few Indian-term sentences) and real test comments of at most
   40 words that the model gets wrong for `toxic` at its tuned cutoff: the highest-scored false
   positives (comments with profanity come last) and the lowest-scored false negatives, taken from
   runs/<run>/test_predictions.csv joined to data/processed/test.csv. The word attributions are
   Integrated Gradients on the `toxic` and `identity_hate` logits. Profanity is masked.
2. Aggregate: identity_term_attribution.csv and fig_identity_term_attribution.png. Over the
   harmless sentences in data/templates/templates_en.csv, the mean attribution of the identity
   term (all its word pieces) to the `toxic` logit, per identity term. A stratified sample per
   term set keeps it fast.
3. summary.json: number of explained examples, mean completeness error and the settings.

Completeness: the attributions of a text add up to logit(text) - logit(baseline); the gap is
reported per example (and averaged in summary.json). More --steps shrinks it.
Runs on CPU (slow) or CUDA (picked automatically, or --device).
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from toxicity.explain import mask_text, mask_word
from toxicity.explain_transformer import IntegratedGradients, span_attribution
from toxicity.plots import MUTED, SERIES, hbar, plt, save, style
from toxicity.scoring import TransformerScorer

LABELS_SHOWN = ["toxic", "identity_hate"]
MAX_WORDS = 40
N_ERRORS = 3
TEMPLATES = [
    "I am a gay person",
    "I am a homosexual person",
    "I am a proud person",
    "I am a muslim person",
    "You are a great actor",
    "I am a dalit person",
    "I am an obc person",
    "I am a brahmin person",
    "I am a bihari person",
]
TERM_SETS = ["dixon", "india", "neutral"]
FIG_TERMS = 15
POS_RGB, NEG_RGB = "235,104,52", "42,120,214"


def pick_error_examples(preds_path: Path, test_path: Path, cutoff: float, n: int = N_ERRORS) -> list[tuple[str, str]]:
    """(kind, text): top-scored toxic false positives and lowest-scored false negatives, short comments only."""
    if not preds_path.exists() or not test_path.exists():
        return []
    preds = pd.read_csv(preds_path, usecols=["id", "toxic", "prob_toxic"])
    text = pd.read_csv(test_path, usecols=["id", "comment_text"], keep_default_na=False)
    df = preds.merge(text, on="id")
    df = df[df["comment_text"].str.split().str.len().between(1, MAX_WORDS)]
    pred = df["prob_toxic"] >= cutoff
    fp = df[pred & (df["toxic"] == 0)].copy()
    fn = df[~pred & (df["toxic"] == 1)].copy()
    # Comments with profanity are the less informative false positives: list them last.
    fp["profane"] = fp["comment_text"].map(lambda t: mask_text(t) != t)
    fp = fp.sort_values(["profane", "prob_toxic"], ascending=[True, False], kind="stable").head(n)
    fn = fn.sort_values("prob_toxic", kind="stable").head(n)
    return [("false positive (toxic)", t) for t in fp["comment_text"]] + [
        ("false negative (toxic)", t) for t in fn["comment_text"]
    ]


def sample_templates(templates: Path, per_set: int, term_sets: list[str], seed: int) -> pd.DataFrame:
    """Harmless sentences with an identity slot, about `per_set` per term set, spread evenly over its terms."""
    df = pd.read_csv(templates, keep_default_na=False)
    df = df[(df["toxicity"] == "nontoxic") & df["term_set"].isin(term_sets) & df["frame"].str.contains("{IDENTITY}", regex=False)]
    parts = []
    for _, group in df.groupby("term_set"):
        k = max(1, int(np.ceil(per_set / group["identity"].nunique())))
        parts += [g.sample(min(k, len(g)), random_state=seed) for _, g in group.groupby("identity")]
    return pd.concat(parts, ignore_index=True)


def identity_span(text: str, frame: str) -> tuple[int, int]:
    """Character span of the identity term, from where {IDENTITY} sits in the frame."""
    start = frame.index("{IDENTITY}")
    return start, start + len(text) - (len(frame) - len("{IDENTITY}"))


def tint(value: float, scale: float) -> str:
    rgb = POS_RGB if value > 0 else NEG_RGB
    return f"rgba({rgb},{min(abs(value) / scale, 1) * 0.75:.2f})"


def markdown_block(title: str, text: str, results: dict, thresholds: dict) -> str:
    lines = [f"### {title}", "", f"> {mask_text(text)}", ""]
    lines.append("Scores: " + ", ".join(f"{l} {r['probability']:.3f} (cutoff {thresholds[l]:.3f})" for l, r in results.items()))
    lines.append("")
    base = results[LABELS_SHOWN[0]]
    order = np.argsort(-np.abs(base["attribution"]), kind="stable")[:8]
    lines += ["| word | " + " | ".join(LABELS_SHOWN) + " |", "|---|" + "---:|" * len(LABELS_SHOWN)]
    for i in sorted(order):
        cells = " | ".join(f"{results[l]['attribution'][i]:+.2f}" for l in LABELS_SHOWN)
        mark = "**" if abs(base["attribution"][i]) >= 0.5 else ""
        lines.append(f"| {mark}{mask_word(base['words'][i])}{mark} | {cells} |")
    for name, key in [("sum of attributions", None), ("logit", "logit"), ("baseline logit", "baseline_logit"),
                      ("completeness error", "delta")]:
        vals = [results[l]["attribution"].sum() if key is None else results[l][key] for l in LABELS_SHOWN]
        lines.append(f"| {name} | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")
    return "\n".join(lines) + "\n"


def html_block(title: str, text: str, results: dict, thresholds: dict) -> str:
    parts = [f"<h3>{html.escape(title)}</h3>"]
    for label, r in results.items():
        scale = max(np.abs(r["attribution"]).max(), 1.0)
        spans = " ".join(
            f'<span style="background:{tint(a, scale)}" title="{a:+.2f}">{html.escape(mask_word(w))}</span>'
            for w, a in zip(r["words"], r["attribution"])
        )
        parts.append(
            f"<p><b>{label}</b> p={r['probability']:.3f} (cutoff {thresholds[label]:.3f}, "
            f"completeness error {r['delta']:+.3f})<br>{spans}</p>"
        )
    return "\n".join(parts)


def write_local_examples(examples, ig, thresholds, run_name, out: Path) -> list[float]:
    md = [
        f"# Local explanations (Integrated Gradients): {run_name}",
        "",
        f"Attribution of each word to the logit, {ig.n_steps} steps, baseline = {ig.baseline.upper()} embeddings with CLS/SEP kept,",
        "word pieces summed per word. Positive pushes toward the label. Attributions add up to logit minus baseline logit",
        "up to the integration error shown in the last row. Profanity is masked.",
        "",
    ]
    page = [
        "<!doctype html><meta charset='utf-8'><title>Local explanations</title>",
        "<style>body{font:15px/1.7 sans-serif;max-width:860px;margin:2em auto;color:#0b0b0b}"
        "span{padding:2px 4px;border-radius:3px}h3{margin-bottom:0}small{color:#52514e}</style>",
        f"<h2>Local explanations (Integrated Gradients): {html.escape(run_name)}</h2>",
        "<p><small>Word colour = attribution to the logit: orange pushes toward the label, blue away. "
        "Hover for the value. Profanity is masked.</small></p>",
    ]
    errors = []
    for kind, text in examples:
        results = ig.explain(text, LABELS_SHOWN)
        errors += [r["abs_delta"] for r in results.values()]
        md.append(markdown_block(kind, text, results, thresholds))
        page.append(html_block(kind, text, results, thresholds))
        print(f"  {kind:24s} p_toxic={results['toxic']['probability']:.3f}  {mask_text(text)[:60]!r}")
    (out / "local_examples.md").write_text("\n".join(md), encoding="utf-8")
    (out / "local_examples.html").write_text("\n".join(page), encoding="utf-8")
    return errors


def aggregate_identity(sample: pd.DataFrame, ig: IntegratedGradients) -> tuple[pd.DataFrame, list[float]]:
    rows, errors = [], []
    for i, r in enumerate(sample.itertuples(index=False), 1):
        res = ig.explain(r.text, ["toxic"])["toxic"]
        start, end = identity_span(r.text, r.frame)
        rows.append({
            "term_set": r.term_set, "axis": r.axis, "identity": r.identity,
            "term_attribution": span_attribution(res["spans"], res["attribution"], start, end),
            "logit": res["logit"], "probability": res["probability"],
        })
        errors.append(res["abs_delta"])
        if i % 100 == 0:
            print(f"  {i}/{len(sample)}")
    df = pd.DataFrame(rows)
    table = (
        df.groupby(["term_set", "axis", "identity"], as_index=False)
        .agg(n=("logit", "size"), mean_attribution=("term_attribution", "mean"),
             std_attribution=("term_attribution", "std"), mean_logit=("logit", "mean"),
             mean_probability=("probability", "mean"))
        .sort_values("mean_attribution", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return table, errors


def plot_identity_attribution(table: pd.DataFrame, run_name: str, out: Path) -> None:
    top = table.head(FIG_TERMS)
    colours = {s: SERIES[k % len(SERIES)] for k, s in enumerate(sorted(table["term_set"].unique()))}
    fig, ax = plt.subplots(figsize=(7, 5))
    hbar(ax, [mask_text(t) for t in top["identity"]], top["mean_attribution"].to_numpy(), fmt="{:+.2f}")
    for patch, s in zip(ax.patches, top["term_set"]):
        patch.set_color(colours[s])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Identity term attribution to the toxic logit", fontsize=11)
    ax.set_xlabel(f"mean Integrated Gradients attribution on harmless templates ({run_name})")
    ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, list(colours), fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out / "fig_identity_term_attribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Folder written by 05_train_transformer.py (runs/<run>/model)")
    parser.add_argument("--thresholds", required=True, help="runs/<run>/thresholds.json")
    parser.add_argument("--out", required=True, help="Output folder, e.g. reports/explain/<run_name>")
    parser.add_argument("--predictions", help="Default: test_predictions.csv next to thresholds.json")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--templates", default="data/templates/templates_en.csv")
    parser.add_argument("--steps", type=int, default=50, help="Integration steps")
    parser.add_argument("--step-batch", type=int, default=None, help="Steps per forward pass (default: all; lower it if memory is short)")
    parser.add_argument("--baseline", choices=["pad", "mask"], default="pad")
    parser.add_argument("--per-set", type=int, default=300, help="Harmless template sentences per term set")
    parser.add_argument("--term-sets", nargs="+", default=TERM_SETS)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap the local examples and the aggregate sample (quick runs)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(Path(args.thresholds).read_text())
    run_name = out.name
    style()
    t0 = time.time()

    scorer = TransformerScorer(args.model, max_length=args.max_length, device=args.device)
    ig = IntegratedGradients(scorer.model, scorer.tokenizer, args.max_length, args.steps, args.baseline, args.step_batch)
    print(f"Model on {ig.device}, {args.steps} steps, baseline {args.baseline}")

    # 1. Local explanations.
    preds = Path(args.predictions) if args.predictions else Path(args.thresholds).parent / "test_predictions.csv"
    errors_found = pick_error_examples(preds, Path(args.test), thresholds["toxic"])
    if not errors_found:
        print(f"No error examples ({preds} or {args.test} missing): template sentences only")
    examples = [("template sentence", t) for t in TEMPLATES] + errors_found
    if args.max_examples:
        examples = examples[: args.max_examples]
    print("\n== local examples")
    errors = write_local_examples(examples, ig, thresholds, run_name, out)
    n_local = len(examples)

    # 2. Aggregate identity-term attribution.
    sample = sample_templates(Path(args.templates), args.per_set, args.term_sets, args.seed)
    if args.max_examples:
        sample = sample.sample(min(args.max_examples, len(sample)), random_state=args.seed).reset_index(drop=True)
    print(f"\n== identity attribution over {len(sample)} harmless template sentences")
    table, agg_errors = aggregate_identity(sample, ig)
    errors += agg_errors
    table.to_csv(out / "identity_term_attribution.csv", index=False)
    plot_identity_attribution(table, run_name, out)
    print(table.head(8).assign(identity=lambda d: d["identity"].map(mask_text))[["term_set", "identity", "n", "mean_attribution"]]
          .round(3).to_string(index=False))

    # 3. Summary.
    summary = {
        "run_name": run_name,
        "n_examples": n_local + len(sample),
        "n_local_examples": n_local,
        "n_template_examples": len(sample),
        "mean_completeness_error": float(np.mean(errors)),
        "max_completeness_error": float(np.max(errors)),
        "steps": args.steps,
        "baseline": args.baseline,
        "device": str(ig.device),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMean completeness error {summary['mean_completeness_error']:.4f} logits. Saved to {out}/")


if __name__ == "__main__":
    main()
