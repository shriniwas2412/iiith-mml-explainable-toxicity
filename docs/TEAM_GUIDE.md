# Team guide

Start here if you are new to the project. This guide explains what we are building, the ideas behind each part, how the code is organised, how to run it, and what each workstream does next.

> **Content note.** The Jigsaw data and the template sentences contain slurs, threats and other offensive language. Do not paste raw examples into chats or slides without a warning.

**How to read this guide**

| You are… | Read |
|---|---|
| Everyone (first day) | 1, 2, 4, 5, 9 |
| Data and baseline owner | 3.1–3.4, 6, 8.1 |
| DistilBERT and thresholds owner | 3.4–3.7, 7, 8.2 |
| Explainability owner | 3.7, 3.8, 8.3 |
| Bias audit owner | 3.9, 3.10, 8.4 |
| Demo, report and slides owner | 3.6, 3.8, 8.5, 10 |

Questions go to the project lead (Shriniwas) or the team chat. Workstream owners and checkpoint dates are in [ROADMAP.md](ROADMAP.md).

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The pipeline at a glance](#2-the-pipeline-at-a-glance)
3. [Background you need](#3-background-you-need)
4. [Repository tour](#4-repository-tour)
5. [Set up your machine](#5-set-up-your-machine)
6. [Run the pipeline](#6-run-the-pipeline)
7. [Train DistilBERT on Kaggle or Colab](#7-train-distilbert-on-kaggle-or-colab)
8. [Workstreams: what each owner does next](#8-workstreams-what-each-owner-does-next)
9. [How we work together](#9-how-we-work-together)
10. [Rules for results](#10-rules-for-results)
11. [Troubleshooting](#11-troubleshooting)
12. [Glossary](#12-glossary)
13. [Reading list](#13-reading-list)

---

## 1. The project in one page

**Course:** IIIT Hyderabad, Modern Machine Learning programme, Capstone 4: *Multi-Label Toxic Comment Classification with Explainability*.

**Task.** Given an online comment, predict six yes/no labels at once: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. A comment can have none, one or several of them.

**What we deliver**

1. A multi-label classifier: a simple baseline and a fine-tuned DistilBERT.
2. One decision threshold per label, tuned on validation data.
3. Explanations for individual predictions (LIME, SHAP, Integrated Gradients).
4. A bias audit: does the model flag *harmless* sentences just because they mention an identity group? We test the standard Western identity terms *and* Indian caste, religion and region terms.
5. A Gradio demo, a deployment brief, a final report and slides.

**Our research angle.** Most published bias tests of toxicity classifiers use Western identity terms. We ask:

- **RQ1.** Do Jigsaw-trained classifiers flag harmless sentences more often when they mention Indian identities than when they mention the Western identities of Dixon et al. (2018)?
- **RQ2.** Which label goes up: the general `toxic` label or `identity_hate`?
- **RQ3.** On those false positives, do LIME, SHAP and Integrated Gradients agree that the identity word is what drove the prediction?
- **RQ4 (optional).** Does a cheap fix close the gap, and what does it cost in F1?

If the results hold up, we aim to turn this into a workshop paper after the capstone. Authorship will be agreed with the mentor. Either way, the capstone deliverables come first.

**Checkpoints** (dates in [ROADMAP.md](ROADMAP.md)):

| # | Done when |
|---|---|
| Proposal | 7–8 pages: problem, literature review, method, staged outcomes, applicability |
| 1 | EDA tables, fixed splits, baseline and DistilBERT on a 30k subset, metrics table |
| 2 | DistilBERT on all training data, weighted BCE vs focal loss, tuned thresholds |
| 3 | Demo running, explanations for 15–20 predictions, bias audit results |
| 4 | Final model, deployment brief, report, slides, live demo |

---

## 2. The pipeline at a glance

```
 Jigsaw CSVs ──01──▶ data/raw/ ──02──▶ reports/eda/            (tables + plots)
                         │
                         └──03──▶ data/processed/ train / val / test (+ 30k subset)
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          │
     04 TF-IDF + LR baseline    05 DistilBERT (GPU)                │
             │                          │                          │
             └──────────┬───────────────┘                          │
                        ▼                                          │
            runs/<run>/val_predictions.csv, test_predictions.csv   │
                        │                                          │
                        ▼                                          │
            06 tune thresholds on val ─▶ thresholds.json,          │
                                         test_metrics_*.csv        │
                        │                                          │
 Dixon templates +      ▼                                          │
 Indian term lists ─07─▶ data/external/ ─08─▶ data/templates/templates_en.csv
                        │
                        ▼
            09 bias audit (any model) ─▶ reports/bias/<model>/
                        │
                        ▼
            10 explanations (to write) ─▶ demo, report, paper
```

Every model writes its predictions in the same CSV format. Scripts 06 and 09 therefore work on any model: the baseline, DistilBERT or the pretrained Detoxify models.

---

## 3. Background you need

Each subsection explains one idea and points to where it lives in the code.

### 3.1 The data

- **Source:** Kaggle's *Jigsaw Toxic Comment Classification Challenge* (2018). The comments come from Wikipedia talk pages and were labelled by crowd workers.
- **Files:** `train.csv` (comments with six 0/1 labels), `test.csv` (comments only) and `test_labels.csv`. In `test_labels.csv`, Kaggle marked the rows it did not score with `-1`. We keep only the scored test rows.
- **Expected row counts** (checked by `scripts/01_download_data.py`): 159,571 train, 153,164 test, 63,978 scored test.
- **Imbalance:** about 90% of training comments have no label at all. `threat` is the rarest label, with 478 positives in all of `train.csv` (`reports/eda/label_counts.csv`).
- **Overlap:** labels co-occur heavily. Every `severe_toxic` comment is also `toxic` (`reports/eda/label_overlap.csv`).

### 3.2 Multi-label classification

This is not one 6-way choice. It is **six separate yes/no questions** about the same comment. In practice:

- The model outputs six independent probabilities, each passed through a sigmoid. They are not one softmax.
- The loss is binary cross-entropy summed or averaged over the six labels.
- Each label can have its own decision threshold (see 3.6).

Label order is fixed in `src/toxicity/config.py` (`LABELS`). Always use that constant, never retype the list.

### 3.3 Splits

`scripts/03_make_splits.py` produces:

| File | What it is |
|---|---|
| `data/processed/train.csv` | 90% of Kaggle's train set |
| `data/processed/val.csv` | 10% of Kaggle's train set. Used for model selection and threshold tuning |
| `data/processed/test.csv` | Kaggle's scored test rows. Used **once**, for final numbers |
| `data/processed/train_subset_30000.csv` | 30k rows from `train.csv` with the same label rates (checkpoint 1) |

Why not a plain random split? With labels as rare as `threat`, a random 10% could get very few positives. **Iterative stratification** (Sechidis et al., 2011; `iterative-stratification` package) keeps every label's rate roughly equal on both sides. Code: `multilabel_split` in `src/toxicity/data.py`.

Text cleaning is deliberately minimal: whitespace is collapsed and nothing else changes (`normalize_text`). Lower-casing is left to each model's tokenizer.

### 3.4 Baseline: TF-IDF + logistic regression

`src/toxicity/baseline.py`, run by `scripts/04_train_baseline.py`.

- **TF-IDF** turns a comment into a sparse vector of word and word-pair weights. Words that are frequent in this comment but rare overall get high weight.
- We use **word 1–2-grams** plus **character 3–5-grams**. Character n-grams catch misspellings and obfuscated swear words.
- One **logistic regression** per label (one-vs-rest). `class_weight="balanced"` up-weights the rare positives.
- Settings live in `configs/default.yaml` under `baseline:`. If your laptop runs out of memory, set `use_char: false`.

The baseline is fast, strong on this dataset and easy to explain. It is a real comparison point, not just a warm-up.

### 3.5 DistilBERT and the loss functions

`src/toxicity/train.py`, `src/toxicity/losses.py`, run by `scripts/05_train_transformer.py`.

- **DistilBERT** (`distilbert-base-uncased`) is a smaller, faster BERT. We fine-tune it with a 6-output classification head. Inputs are cut at 256 tokens (`model.max_length`).
- **Weighted BCE:** each label's positive examples get weight `(negatives / positives) ** power`. `power = 0.5` (the default) softens the weight, so `threat` doesn't dominate.
- **Focal loss** (Lin et al., 2017) down-weights easy examples. The model then spends its effort on the hard, often rare, ones. `gamma = 0` reduces it to plain BCE.
- Training evaluates on validation after every epoch and keeps the epoch with the best mean PR-AUC.
- Outputs go to `runs/<name>/`: `model/` (weights and tokenizer), `val_predictions.csv`, `test_predictions.csv`, `history.json`, `run_config.json`.

This needs a GPU: see section 7.

### 3.6 Thresholds

A model gives probabilities, but moderation needs yes/no decisions. A flat cutoff of 0.5 is rarely best for rare labels. `scripts/06_tune_thresholds.py`:

1. For each label, tries every possible threshold on **validation** predictions and keeps the one with the best F1 (`tune_thresholds` in `src/toxicity/metrics.py`).
2. Writes `thresholds.json`.
3. Scores the test set twice, with a flat 0.5 and with the tuned thresholds, and saves `test_metrics_flat.csv` and `test_metrics_tuned.csv`.

The test set is never used to choose anything.

### 3.7 Metrics

`evaluate` in `src/toxicity/metrics.py` reports, per label and averaged:

| Metric | What it tells you |
|---|---|
| Precision | Of the comments we flagged, how many really had the label |
| Recall | Of the comments that had the label, how many we flagged |
| F1 | Harmonic mean of precision and recall; depends on the threshold |
| ROC-AUC | How well the scores rank positives above negatives; threshold-free. Looks optimistic on rare labels |
| PR-AUC | Area under the precision-recall curve; threshold-free and much more honest for rare labels |
| macro | Plain average over the six labels: every label counts equally |
| micro | Pools all decisions: frequent labels dominate |

For this project, read **macro F1** and **PR-AUC** first. ROC-AUC values above 0.95 are normal here and hide large differences on rare labels.

### 3.8 Explainability

An explanation assigns each word a score for how much it pushed a label's probability up or down. We plan three methods because they work differently and often disagree:

| Method | Idea | Works with |
|---|---|---|
| **LIME** | Delete random words, watch the score change, fit a small linear model to those changes | Any model (black box). Random, so run several seeds |
| **SHAP** | Shapley values from game theory: average contribution of a word over many subsets of words | Any model; slower |
| **Integrated Gradients (IG)** | Integrate the gradient of the output along a path from a blank input to the real input | Neural models only (DistilBERT), via `captum` |

What we measure (RQ3): on the bias templates' false positives, do the methods agree that the identity word is the top cause? We compare them by top-k overlap, rank correlation and "is the identity token ranked first?".

A caution to repeat in the report: a high attribution is not causal proof. We pair it with the behavioural test in 3.10, which swaps the identity word for a neutral word and measures the score change.

All models share one interface, `predict_proba(texts) -> (n, 6) array`, in `src/toxicity/scoring.py`. LIME and SHAP can call that directly.

### 3.9 The bias templates

`src/toxicity/templates.py`, run by scripts 07 and 08.

Dixon et al. (2018) released sentences built from **frames** such as `You are a kind {IDENTITY}` or `Being {IDENTITY} is terrible`. Each frame is filled with every identity term, so the sentences differ *only* in that word. Each sentence is tagged `toxic` or `nontoxic` by its frame, not by any model.

We build four arms from the same frames:

| `term_set` | Terms | Why |
|---|---|---|
| `dixon` | Dixon's original Western terms (gay, muslim, african american, …) | Reproduces the published test |
| `india` | Caste, religion and region terms from Bhatt et al. (2022) | Our extension |
| `neutral` | `person` | Reference: what a frame scores with no identity in it |
| `control` | Dixon's sentences that have no slot | Sanity check |

`scripts/08_build_templates.py` writes 146,208 sentences to `data/templates/templates_en.csv` (dixon 75,700 · india 68,130 · neutral 1,514 · control 864).

Small fixes to the released lists are kept in `resources/` (for example, `khsatirya` is printed as `kshatriya`), so they are visible and reviewable.

Jigsaw also released `hi-Latn_sentence_templates.csv`, romanised Hindi templates with Indian terms. `scripts/07_fetch_bias_resources.py --hindi` downloads it. It needs its own loader (not written yet).

### 3.10 Bias metrics

`src/toxicity/bias.py`, run by `scripts/09_bias_audit.py`. The audit uses the `toxic` label's probability, and the threshold from `thresholds.json` if you pass one.

| Metric | Meaning | Bad sign |
|---|---|---|
| **FPR per term** | Share of *harmless* sentences with that term that get flagged | High for some terms |
| **FNR per term** | Share of *toxic* sentences with that term that are missed | High for some terms |
| **FPED / FNED** | Sum over terms of how far each term's FPR / FNR is from the overall rate (Dixon 2018) | Large |
| **Subgroup AUC** | Ranking quality using only sentences with that term | Low |
| **BPSN AUC** | Harmless sentences *with* the term vs toxic sentences *without* it (Borkan 2019) | Low: the term's harmless sentences score too high |
| **BNSP AUC** | Toxic sentences *with* the term vs harmless sentences *without* it | Low: the term's toxic sentences score too low |
| **Score shift** | Mean score change when `person` in a frame is replaced by the term (Prabhakaran 2019) | Large positive shift on harmless frames |
| **Mean score per label** | On harmless sentences, which of the six labels each term pushes up | `identity_hate` rising for identity mentions |

Outputs in `reports/bias/<model>/`: `summary.json`, `per_identity_toxic.csv`, `score_shift_toxic.csv`, `mean_scores_nontoxic_by_label.csv`, `scores.csv`.

---

## 4. Repository tour

```
configs/default.yaml        all settings (seed, split size, baseline, DistilBERT, loss, training)
src/toxicity/               library code (import it, don't copy it)
  config.py                 LABELS order, column names, load_config()
  data.py                   load / clean / stratified split / subsample
  metrics.py                evaluate(), tune_thresholds(), apply_thresholds()
  predictions.py            save_predictions() / load_predictions(): the shared CSV format
  baseline.py               TF-IDF + logistic regression pipeline
  losses.py                 weighted BCE, focal loss
  train.py                  DistilBERT fine-tuning loop
  scoring.py                one predict_proba interface for baseline, DistilBERT and Detoxify
  templates.py              builds the matched template set
  bias.py                   bias metrics
scripts/                    numbered steps; thin wrappers around src/toxicity
  01_download_data.py       get the data, check row counts
  02_eda.py                 label counts, overlap, lengths -> reports/eda/
  03_make_splits.py         train / val / test (+ --subset N)
  04_train_baseline.py      baseline -> runs/<name>/
  05_train_transformer.py   DistilBERT -> runs/<name>/   (GPU)
  06_tune_thresholds.py     thresholds.json + test metric tables
  07_fetch_bias_resources.py  Dixon templates and Indian term lists -> data/external/
  08_build_templates.py     data/templates/templates_en.csv
  09_bias_audit.py          bias report for any model -> reports/bias/<name>/
notebooks/train_gpu.ipynb   runs scripts 01, 03, 05, 06, 07–09 on Kaggle / Colab
resources/                  small hand-made lookup tables (term axes, spelling fixes)
tests/                      unit tests on tiny synthetic data
docs/ROADMAP.md             owners, checkpoint dates, status
demo/, report/              placeholders for checkpoint 3–4 deliverables
.github/workflows/          runs pytest on every push and pull request
```

Folders the scripts create and git ignores: `data/`, `runs/`, `reports/`. Never commit them (see 9.4).

---

## 5. Set up your machine

You need Python 3.10 or newer and git. About 350 MB of packages; a laptop is fine for everything except DistilBERT training.

**macOS / Linux**

```bash
git clone https://github.com/shriniwas2412/iiith-mml-explainable-toxicity.git
cd iiith-mml-explainable-toxicity
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pytest -q
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/shriniwas2412/iiith-mml-explainable-toxicity.git
cd iiith-mml-explainable-toxicity
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
pytest -q
```

Without PyTorch, `pytest` should end with **29 passed, 2 skipped**. The two skipped files test the losses and the training loop.

Optional extras:

| File | Adds | When you need it |
|---|---|---|
| `requirements-train.txt` | PyTorch, Transformers (about 750 MB) | Running the two skipped tests locally. Already installed on Kaggle / Colab |
| `requirements-detoxify.txt` | Detoxify | Scoring the pretrained Detoxify models in the bias audit |

Activate the environment (`source .venv/bin/activate`) every time you open a new terminal. All commands below run from the repository root.

---

## 6. Run the pipeline

Run these in order. The right-hand column is what a correct run prints. If you see something different, stop and post the output in the team chat.

| Step | Command | Expect |
|---|---|---|
| 1 | `python scripts/01_download_data.py` | Downloads about 134 MB. Three lines ending `OK`: 159,571 train · 153,164 test · 63,978 scored test |
| 2 | `python scripts/02_eda.py` | toxic 15,294 · severe_toxic 1,595 · obscene 8,449 · threat 478 · insult 7,877 · identity_hate 1,405 |
| 3 | `python scripts/03_make_splits.py --subset 30000` | Table of rows and label counts for train, val, test and the 30k subset; train + val = 159,571 |
| 4 | `python scripts/04_train_baseline.py --train data/processed/train_subset_30000.csv --out runs/baseline_30k` | "Trained on 30,000 comments", then validation and test tables. Takes several minutes |
| 5 | `python scripts/06_tune_thresholds.py --run runs/baseline_30k` | Six thresholds, then flat vs tuned test tables and the macro F1 change |
| 6 | `python scripts/04_train_baseline.py` then `python scripts/06_tune_thresholds.py --run runs/baseline_tfidf_lr` | Same on the full training split. Peak memory was about 2.6 GB on an 8 GB Mac and it swapped heavily with other apps open; close them first (see 11) |
| 7 | `python scripts/07_fetch_bias_resources.py` | Four small files under `data/external/` (add `--hindi` for the fifth) |
| 8 | `python scripts/08_build_templates.py` | 146,208 sentences: dixon 75,700 · india 68,130 · neutral 1,514 · control 864 |
| 9 | `python scripts/09_bias_audit.py --model baseline:runs/baseline_tfidf_lr/model.joblib --thresholds runs/baseline_tfidf_lr/thresholds.json --out reports/bias/baseline` | `summary.json` plus per-term tables, and the ten terms with the highest FPR |

The `Makefile` has shortcuts: `make test`, `make data`, `make eda`, `make splits`, `make baseline`, `make thresholds`, `make templates`, `make audit`.

Every script takes `--help`.

---

## 7. Train DistilBERT on Kaggle or Colab

Don't train DistilBERT on a laptop. `notebooks/train_gpu.ipynb` clones this repo and runs the same scripts on a free GPU.

**One-time Kaggle setup**

1. Create a Kaggle account and **verify your phone number**. Without it, the GPU and Internet switches stay locked.
2. Optional: join the *Jigsaw Toxic Comment Classification Challenge* and accept its rules, then attach it to the notebook with *Add Input*. If you skip this, script 01 downloads the same files from a public mirror.
3. Check your weekly GPU quota on your Kaggle profile before long runs.

**Each run**

1. Kaggle → *Create* → *New Notebook* → *File* → *Import Notebook*, and upload `notebooks/train_gpu.ipynb`.
2. *Settings* → *Accelerator* → a GPU (T4 or P100); *Internet* → *On*.
3. Run the setup cells, then **section 1** (data and splits).
4. Checkpoint 1: run **section 2** only (DistilBERT, weighted BCE, 30k subset).
5. Save your outputs **before the session ends**. Everything in `/kaggle/working` is lost when it stops. Add a cell such as:

   ```
   !zip -qr distilbert_30k.zip runs/distilbert_wbce_30k -x "runs/*/model/*"
   ```

   Then download the zip from the *Output* panel on the right. Keep `model/` only if you need the weights. They are large.
6. Do **not** use *Save Version → Save & Run All* unless you want every section to run, including the multi-hour full-data training.

**Notes**

- The notebook clones `main`. To test code from your branch, add `!git checkout <your-branch>` after the clone cell.
- Note the GPU type, the run time and the commit hash (`!git rev-parse --short HEAD`) for every run, and put them in the run log (see 10).
- Colab works the same way: *Runtime* → *Change runtime type* → *T4 GPU*, then download from the file browser.

---

## 8. Workstreams: what each owner does next

Owners are listed in [ROADMAP.md](ROADMAP.md). Each owner writes the code, the tests and the matching report section for their workstream, and reviews one other workstream's pull requests.

### 8.1 Data, EDA and baseline

**Owns:** scripts 01–04, `data.py`, `baseline.py`.

Next:
- [ ] Run steps 1–6 yourself and check every output against section 6.
- [ ] Add a check for **exact-duplicate texts across splits**: count comments whose text appears in more than one of train / val / test, write the counts to `reports/`, and propose how to handle them in the report.
- [ ] Error analysis on the baseline: for each label, look at 20 false positives and 20 false negatives from `runs/baseline_*/val_predictions.csv`. Write down the patterns (quoted slurs, sarcasm, very long comments, non-English).
- [ ] Write the Data and Baseline sections of the proposal and report, with tables taken from `reports/eda/` and `runs/`.

### 8.2 DistilBERT training and thresholds

**Owns:** scripts 05–06, `train.py`, `losses.py`, `metrics.py`, the GPU notebook.

Next:
- [ ] Checkpoint 1: run notebook sections 1–2 on Kaggle (weighted BCE, 30k subset), then script 06 on the run.
- [ ] Checkpoint 2: full data, **weighted BCE vs focal loss**, **3 seeds each** (`--seed 1`, `--seed 2`, `--seed 3`, each with its own `--out` folder). Same splits for all.
- [ ] Keep a run log (see 10) with config, seed, GPU, time and commit for every run.
- [ ] Compare against the baseline: macro F1 (flat and tuned), per-label PR-AUC.
- [ ] Hyperparameters in `configs/default.yaml` are starting values. If you tune them, tune on validation only and record what you tried.

### 8.3 Explainability

**Owns:** `src/toxicity/explain.py` and `scripts/10_explain.py` (both still to write).

Suggested design:
- One function per method that takes a scorer (`toxicity.scoring.load_scorer(...)`), a text and a label, and returns a list of `(token, score)`.
  - LIME: `lime.lime_text.LimeTextExplainer`, using `scorer.predict_proba`. Run 3–5 seeds and keep the mean and spread.
  - SHAP: `shap.Explainer` with a text masker over `predict_proba`.
  - IG: `captum.attr.LayerIntegratedGradients` on DistilBERT's embedding layer, with a padding-token baseline. DistilBERT only.
- Agreement measures between methods: top-k overlap (k = 1, 3, 5), Spearman / Kendall rank correlation, and "identity token ranked first?".
- Inputs: (a) 15–20 chosen test comments for the demo and report, with a mix of correct, false-positive and false-negative cases; (b) all **false positives on harmless template sentences** from `reports/bias/<model>/scores.csv` (RQ3).
- Put the new packages in a separate `requirements-explain.txt` so the core install stays light.
- Tests: on a tiny model, check output shapes, that scores are finite, and that the agreement measures behave as expected on hand-made rankings.

Read first: Ribeiro 2016 (LIME), Lundberg 2017 (SHAP), Sundararajan 2017 (IG), Krishna 2022 (disagreement), Liu & Avci 2019 (IG on identity terms in toxicity models).

### 8.4 Bias audit

**Owns:** scripts 07–09, `templates.py`, `bias.py`, `resources/`.

Next:
- [ ] Run steps 7–9 on the baseline and check that `summary.json` is written.
- [ ] Run the audit on DistilBERT (`--model hf:runs/<run>/model`) once 8.2 has a model, and on Detoxify `original`, `unbiased` and `multilingual` (`--model detoxify:<variant>`, needs `requirements-detoxify.txt`). Check with the mentor that Detoxify is allowed.
- [ ] Write the **hi-Latn loader**: the Hindi file uses different template names, so it needs its own reader in `templates.py` plus a test.
- [ ] Add **bootstrap confidence intervals over frames** for per-term FPR and score shift.
- [ ] Figures: per-term FPR grouped by axis; a heatmap of identity term × label (mean score on harmless sentences).
- [ ] Bring these open design questions to the mentor:
  - `obc` and `sc/st` read oddly in sentences. Keep, reword or drop?
  - Region demonyms (e.g. "bihari") can carry a derogatory sense. How do we report them?
  - The neutral reference is the single word `person`. Should we add two or three more?
  - Dixon's frames have grammar slips ("a excellent"). Fixing them breaks the one-to-one match.
  - All terms are lowercase, which matters for the cased Detoxify `unbiased` model.

### 8.5 Demo, report and slides

**Owns:** `demo/`, `report/`.

Next:
- [ ] Proposal (7–8 pages): collect each owner's section and keep one consistent style. Sections: problem, literature review, method, staged outcomes, applicability.
- [ ] Demo plan (checkpoint 3): a Gradio app on Hugging Face Spaces that takes a comment and shows the six probabilities, which labels pass their thresholds, the highlighted words from 8.3, and a note when an identity term is present. Start with the baseline model so the demo doesn't wait on GPU work.
- [ ] Keep a shared reference library (Zotero group or one `.bib` file) so citations are consistent.
- [ ] Final deliverables: report (Word), slides (PowerPoint), deployment brief.

---

## 9. How we work together

### 9.1 Git workflow

1. Pull the latest `main`: `git checkout main && git pull`.
2. Make a branch: `git checkout -b <your-name>/<short-task>`, e.g. `shriniwas/hilatn-loader`.
3. Make the change. If you add code in `src/toxicity/`, add or update a test in `tests/`.
4. Run `pytest -q`. It must pass.
5. Push your branch and open a pull request into `main`. Ask one teammate to review.
6. GitHub Actions runs the tests on the PR. Fix it if it goes red.
7. The reviewer approves; then merge.

### 9.2 Commits

- Commit under **your own name and email**. Marks are individual, and the commit history shows who did what. Never commit someone else's work under your name, or yours under theirs.
- Read and run everything before you commit it. You should be able to explain any line you committed in a viva.
- Small commits with clear messages: `Add hi-Latn template loader`, not `update`.

### 9.3 Code conventions

- Library code goes in `src/toxicity/`; scripts in `scripts/` stay thin and call the library.
- Settings go in `configs/default.yaml`, not hard-coded in scripts.
- Use `LABELS` from `config.py` for label order everywhere.
- New models must write predictions with `toxicity.predictions.save_predictions`, so 06 and 09 work on them unchanged.
- New models should expose `predict_proba(texts) -> (n, 6)` through `scoring.py`, so the audit and explainers can use them.

### 9.4 Never commit

- `data/`, `runs/`, `reports/`, model weights (`*.joblib`, `*.pt`, `*.safetensors`). They are in `.gitignore`. Regenerate them with the scripts, or share them through the team drive.
- API tokens, `kaggle.json`, `.env`.
- Notebooks with large outputs. Clear the outputs first.

### 9.5 Weekly rhythm

- Weekly team sync before the Sunday mentoring session. Each owner brings: what ran, what the numbers are (with the file they come from), what is blocked.
- The lead updates the Status table in [ROADMAP.md](ROADMAP.md) after the sync.

---

## 10. Rules for results

These protect the report, and the paper if we write one.

1. **Every number we write comes from a file** produced by a script in this repo, under `runs/` or `reports/`. No numbers from memory, from a chat or from a notebook cell that wasn't saved.
2. **Thresholds and model choices use validation only.** The test set is scored once, at the end.
3. **No claim from a single run.** DistilBERT results use 3 seeds; report the mean and the spread.
4. **Report effect sizes**, such as the size of an FPR gap, not only whether a difference exists.
5. **Keep a run log** in the team drive, one row per run: run folder name, model, loss, seed, train file, commit hash, GPU, time, notes.
6. **Run folder names** say what was run: `baseline_30k`, `distilbert_wbce_30k`, `distilbert_focal_seed2`.
7. **Share run outputs** by zipping the run folder (without `model/` unless it's needed) and uploading it to the team drive.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: toxicity` | Activate the venv and run `pip install -e .` from the repo root |
| `01_download_data.py` prints `MISMATCH` | Delete `data/raw/` and run it again. If it persists, the mirror changed: tell the lead, don't use the files |
| Baseline is very slow or the laptop freezes | Memory. Close other heavy apps, or set `use_char: false` in `configs/default.yaml`, or run the baseline on Kaggle. Train on the 30k subset first |
| `pytest` says 2 skipped | Expected without PyTorch |
| Kaggle has no GPU or Internet option | Verify your phone number on Kaggle |
| Kaggle run disappeared | `/kaggle/working` is wiped when the session ends. Zip and download before stopping |
| CUDA out of memory | Pass `--batch-size 16` to script 05, or lower `train.batch_size` in the config |
| Detoxify fails to install or load | Use a separate venv for `requirements-detoxify.txt` and report the versions in the team chat |

---

## 12. Glossary

| Term | Meaning |
|---|---|
| Multi-label | Each example can have several labels at once |
| Stratified split | A split that keeps label rates the same on both sides |
| TF-IDF | Word-weighting scheme: frequent in this text, rare overall → high weight |
| Logit / sigmoid | Raw model score / function that turns it into a 0–1 probability |
| BCE | Binary cross-entropy, the standard loss for yes/no outputs |
| pos_weight | Extra weight on positive examples in BCE, for rare labels |
| Focal loss | BCE that down-weights easy examples |
| Threshold | Probability above which we say "yes" for a label |
| PR-AUC | Area under precision-recall curve; the honest metric for rare labels |
| Macro / micro average | Average over labels equally / pool all decisions |
| Identity term | A word naming a social group (e.g. "muslim", "dalit", "gay") |
| Frame / template | A sentence with a slot, filled with each identity term |
| False positive (bias sense) | A harmless sentence flagged as toxic |
| FPED / FNED | Total gap between per-term and overall false positive / negative rates |
| BPSN / BNSP AUC | Background-Positive-Subgroup-Negative / Background-Negative-Subgroup-Positive AUC |
| Attribution | A per-word score from an explainer |
| Detoxify | Pretrained toxicity models released by Unitary, used as external comparisons |

---

## 13. Reading list

Start with the papers for your workstream. Add everything you cite to the shared reference library.

**Everyone**
- van Aken et al. (2018). *Challenges for Toxic Comment Classification: An In-Depth Error Analysis.* ALW2.
- Dixon et al. (2018). *Measuring and Mitigating Unintended Bias in Text Classification.* AIES.

**Data and baseline**
- Sechidis et al. (2011). *On the Stratification of Multi-Label Data.* ECML PKDD.

**DistilBERT and thresholds**
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.

**Explainability**
- Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
- Krishna et al. (2022). *The Disagreement Problem in Explainable Machine Learning.* arXiv:2202.01602.
- Liu & Avci (2019). *Incorporating Priors with Feature Attribution on Text Classification.* ACL.
- Balkir et al. (2022). *Challenges in Applying Explainability Methods to Improve the Fairness of NLP Models.* TrustNLP.

**Bias audit**
- Borkan et al. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.* WWW Companion.
- Prabhakaran et al. (2019). *Perturbation Sensitivity Analysis to Detect Unintended Model Biases.* EMNLP-IJCNLP.
- Garg et al. (2019). *Counterfactual Fairness in Text Classification through Robustness.* AIES.
- Ghosh et al. (2021). *Detecting Cross-Geographic Biases in Toxicity Modeling on Social Media.* W-NUT.
- Bhatt et al. (2022). *Re-contextualizing Fairness in NLP: The Case of India.* AACL-IJCNLP.
- Röttger et al. (2022). *Multilingual HateCheck: Functional Tests for Multilingual Hate Speech Detection Models.* WOAH.
- Sahoo et al. (2024). *IndiBias: A Benchmark Dataset to Measure Social Biases in Language Models for Indian Context.* NAACL.
