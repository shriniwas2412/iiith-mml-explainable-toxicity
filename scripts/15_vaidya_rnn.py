"""Replicate the model of Vaidya, Mai and Ning (2020): Bi-GRU/LSTM with attention on GloVe or
fastText vectors, with an optional multi-task identity head and identity re-weighting.

Needs a GPU for comfort on the full data: run it in the Kaggle notebook. The first run
downloads the embeddings (about 1 GB for GloVe) to --embedding-dir.

    python scripts/15_vaidya_rnn.py --train data/processed/train_subset_30000.csv --out runs/vaidya_gru_30k
    python scripts/15_vaidya_rnn.py --identity-weight 3 --out runs/vaidya_gru_idw3_30k
    python scripts/15_vaidya_rnn.py --identity-weight 3 --multitask on --out runs/vaidya_gru_mt_idw3_30k

--identity-weight c multiplies the loss of non-toxic training comments that mention a Dixon
or Indian identity term by c, as in scripts/05_train_transformer.py.

--multitask on adds a second head that predicts which identity axes a comment mentions.
Deviation from the paper: Jigsaw 2018 has no identity annotations, so these targets are weak
labels from the identity lexicon (term mentions, one target per "term_set:axis"), not human
labels of who a comment is about.

Then run scripts/06_tune_thresholds.py on the output folder. The model folder is scored in
scripts/09_bias_audit.py with --model rnn:<out>/model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from toxicity import LABELS, load_config
from toxicity.config import TEXT_COL
from toxicity.data import read_split
from toxicity.lexicon import axis_mentions, load_identity_terms
from toxicity.metrics import evaluate
from toxicity.predictions import save_predictions
from toxicity.rnn import (
    DEFAULT_EMBEDDING_DIR,
    build_embedding_matrix,
    build_vocab,
    embedding_file,
    random_embedding_matrix,
    train_rnn,
)
from toxicity.scoring import RNNScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--embedding", choices=["glove", "fasttext", "random"], default=None)
    parser.add_argument("--embedding-dir", default=str(DEFAULT_EMBEDDING_DIR), help="Cache folder for the download")
    parser.add_argument("--embedding-file", default=None, help="Local vectors file (txt, vec or zip) instead of the download")
    parser.add_argument("--cell", choices=["gru", "lstm"], default=None)
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal"], default=None)
    parser.add_argument("--multitask", choices=["on", "off"], default=None, help="Identity-axis auxiliary head")
    parser.add_argument("--aux-weight", type=float, default=None, help="Weight of the identity loss")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--identity-weight", type=float, default=1.0, help="Weight for non-toxic identity mentions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.loss:
        cfg["train"]["loss"] = args.loss
    overrides = {
        "embedding": args.embedding,
        "cell": args.cell,
        "multitask": None if args.multitask is None else args.multitask == "on",
        "aux_weight": args.aux_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    cfg["rnn"].update({key: value for key, value in overrides.items() if value is not None})
    if args.seed is not None:
        cfg["seed"] = args.seed
    rnn_cfg = cfg["rnn"]

    processed = Path(cfg["data"]["processed_dir"])
    train = read_split(args.train or processed / "train.csv")
    val = read_split(args.val or processed / "val.csv")
    test = read_split(args.test or processed / "test.csv")
    multitask = bool(rnn_cfg["multitask"])
    name = f"vaidya_{rnn_cfg['cell']}{'_mt' if multitask else ''}"
    out = Path(args.out or f"runs/{name}_seed{cfg['seed']}")

    weights, identity_targets, axes = None, None, None
    if args.identity_weight != 1.0 or multitask:
        terms = load_identity_terms()
        by_axis = axis_mentions(train[TEXT_COL], terms)
        if args.identity_weight != 1.0:
            nontoxic_identity = by_axis.any(axis=1).to_numpy() & (train[LABELS].sum(axis=1) == 0).to_numpy()
            weights = np.where(nontoxic_identity, args.identity_weight, 1.0)
            print(f"Identity weight {args.identity_weight} on {int(nontoxic_identity.sum()):,} training rows")
        if multitask:
            axes = list(by_axis.columns)
            identity_targets = by_axis.to_numpy(dtype=np.float32)
            print(f"Identity head on {len(axes)} axes; rows mentioning any: {int(by_axis.any(axis=1).sum()):,}")

    vocab = build_vocab(train[TEXT_COL].tolist(), rnn_cfg["max_vocab"])
    if rnn_cfg["embedding"] == "random":
        embeddings = random_embedding_matrix(vocab, rnn_cfg["embedding_dim"], cfg["seed"])
    else:
        path = args.embedding_file or embedding_file(rnn_cfg["embedding"], args.embedding_dir)
        embeddings, coverage = build_embedding_matrix(vocab, path, rnn_cfg["embedding_dim"], seed=cfg["seed"])
        print(f"Vocabulary {len(vocab):,} words, {coverage:.1%} found in the embedding file")

    result = train_rnn(
        train, val, cfg, out, vocab, embeddings, device=args.device, sample_weight=weights,
        identity_targets=identity_targets, identity_names=axes,
    )
    print(f"Best epoch {result['best_epoch']} (validation mean PR-AUC {result['best_val_pr_auc']:.4f})")

    scorer = RNNScorer(str(out / "model"), batch_size=cfg["train"]["eval_batch_size"], device=args.device)
    test_prob = scorer.predict_proba(test[TEXT_COL].tolist())
    save_predictions(out / "test_predictions.csv", test, test_prob)
    print("\ntest metrics at a flat 0.5 cutoff:")
    print(evaluate(test[LABELS].to_numpy(), test_prob).round(4))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
