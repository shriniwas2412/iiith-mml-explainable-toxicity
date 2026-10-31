# iiith-mml-explainable-toxicity

MML Capstone 4 — Multi-Label Toxic Comment Classification with Explainability
(IIIT Hyderabad, Modern Machine Learning programme).

The project trains a multi-label classifier on the Jigsaw Toxic Comment data
(toxic, severe_toxic, obscene, threat, insult, identity_hate), tunes one
decision threshold per label, explains individual predictions, and tests
whether the model over-flags harmless sentences that mention an identity group.
The bias test covers the identity terms of Dixon et al. (2018) and Indian
caste, religion and region terms from Bhatt et al. (2022).

> The dataset and the template sentences contain offensive language.

New to the project? Start with the [team guide](docs/TEAM_GUIDE.md).

## Status

| Part | State |
|---|---|
| Data download with row-count checks, EDA, stratified splits | Implemented |
| TF-IDF + logistic regression baseline | Implemented |
| DistilBERT fine-tuning with BCE, weighted BCE or focal loss | Implemented |
| Per-label threshold tuning on validation | Implemented |
| Matched template set (same frames, different identity terms) | Implemented |
| Bias metrics: per-term FPR/FNR, FPED/FNED, subgroup / BPSN / BNSP AUC, score shift | Implemented |
| LIME / SHAP / Integrated Gradients explanations | Planned |
| Gradio demo and deployment brief | Planned |

The code is covered by unit tests on small synthetic files. No results on the
real dataset are reported here yet.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pytest -q
```

`requirements-train.txt` adds PyTorch and Transformers for fine-tuning.
`requirements-detoxify.txt` adds the Detoxify comparison models.

## Running the pipeline

Run from the repository root. Each script has `--help`.

```bash
python scripts/01_download_data.py            # data/raw/, checks the row counts
python scripts/02_eda.py                      # reports/eda/
python scripts/03_make_splits.py --subset 30000

python scripts/04_train_baseline.py --out runs/baseline_tfidf_lr
python scripts/06_tune_thresholds.py --run runs/baseline_tfidf_lr

python scripts/07_fetch_bias_resources.py     # data/external/
python scripts/08_build_templates.py          # data/templates/templates_en.csv
python scripts/09_bias_audit.py --model baseline:runs/baseline_tfidf_lr/model.joblib \
    --thresholds runs/baseline_tfidf_lr/thresholds.json --out reports/bias/baseline
```

Fine-tuning needs a GPU. `notebooks/train_gpu.ipynb` runs the same scripts on
Kaggle or Colab:

```bash
python scripts/05_train_transformer.py --loss weighted_bce --out runs/distilbert_wbce
python scripts/06_tune_thresholds.py --run runs/distilbert_wbce
python scripts/09_bias_audit.py --model hf:runs/distilbert_wbce/model \
    --thresholds runs/distilbert_wbce/thresholds.json --out reports/bias/distilbert_wbce
```

Settings live in `configs/default.yaml`. They are starting values, not tuned.

## Method notes

- **Splits.** Validation is taken from `train.csv` with iterative multi-label
  stratification, because `threat` covers about 0.3% of comments. The test set
  keeps only the rows Kaggle scored (label `-1` rows are dropped).
- **Imbalance.** Handled in the loss (class weights or focal loss), not by
  resampling.
- **Thresholds.** Chosen per label for best F1 on validation only, then applied
  to the test set once.
- **Bias audit.** Each template frame, such as `You are a kind ___`, is filled
  with every identity term and with a neutral word, so sentences differ only in
  that term. The script reports how often harmless sentences are flagged per
  term, and which of the six labels rises.

## Layout

```
configs/      settings
src/toxicity/ library code (data, metrics, baseline, losses, train, scoring, templates, bias)
scripts/      numbered command-line steps
notebooks/    GPU notebook for Kaggle / Colab
resources/    small lookup tables kept in the repo
tests/        unit tests
```

`data/`, `runs/` and `reports/` are created by the scripts and are not tracked.

## Data and licences

- Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018). Comment text is
  from Wikipedia talk pages.
- Sentence templates from `conversationai/unintended-ml-bias-analysis` (CC BY 4.0).
- Indian identity term lists from `google-research-datasets/nlp-fairness-for-india` (Apache-2.0).

None of these files are stored in this repository; the scripts download them.

## References

- Dixon et al. (2018). Measuring and Mitigating Unintended Bias in Text Classification. AIES.
- Borkan et al. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. WWW Companion.
- Bhatt et al. (2022). Re-contextualizing Fairness in NLP: The Case of India. AACL-IJCNLP.
- Prabhakaran et al. (2019). Perturbation Sensitivity Analysis to Detect Unintended Model Biases. EMNLP-IJCNLP.
- Sechidis et al. (2011). On the Stratification of Multi-Label Data. ECML PKDD.
- Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- van Aken et al. (2018). Challenges for Toxic Comment Classification: An In-Depth Error Analysis. ALW2.
