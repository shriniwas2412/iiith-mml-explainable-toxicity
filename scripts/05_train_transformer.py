"""Fine-tune DistilBERT on the six toxicity labels and save its predictions.

Needs a GPU for the full data: run it in the Kaggle / Colab notebook.

    python scripts/05_train_transformer.py --loss weighted_bce --out runs/distilbert_wbce
    python scripts/05_train_transformer.py --loss focal --out runs/distilbert_focal
    python scripts/05_train_transformer.py --train data/processed/train_subset_30000.csv --out runs/distilbert_30k

Then run scripts/06_tune_thresholds.py on the output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toxicity import LABELS, load_config
from toxicity.config import TEXT_COL
from toxicity.data import read_split
from toxicity.metrics import evaluate
from toxicity.predictions import save_predictions
from toxicity.scoring import TransformerScorer
from toxicity.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None, help="Hugging Face model name or local folder")
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {"loss": args.loss, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    cfg["train"].update({key: value for key, value in overrides.items() if value is not None})
    if args.model:
        cfg["model"]["name"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed

    processed = Path(cfg["data"]["processed_dir"])
    train = read_split(args.train or processed / "train.csv")
    val = read_split(args.val or processed / "val.csv")
    test = read_split(args.test or processed / "test.csv")
    out = Path(args.out or f"runs/transformer_{cfg['train']['loss']}_seed{cfg['seed']}")

    result = train_model(train, val, cfg, out, device=args.device)
    print(f"Best epoch {result['best_epoch']} (validation mean PR-AUC {result['best_val_pr_auc']:.4f})")

    scorer = TransformerScorer(
        str(out / "model"),
        batch_size=cfg["train"]["eval_batch_size"],
        max_length=cfg["model"]["max_length"],
        device=args.device,
    )
    test_prob = scorer.predict_proba(test[TEXT_COL].tolist())
    save_predictions(out / "test_predictions.csv", test, test_prob)
    print("\ntest metrics at a flat 0.5 cutoff:")
    print(evaluate(test[LABELS].to_numpy(), test_prob).round(4))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
