# Run guide: where to run what, and where the results go

Three places to run things. Pick by how heavy the job is.

| Job | Where | Time | Output folder |
|---|---|---|---|
| Data analysis (script 10) | Your laptop or Colab (CPU) | ~5 min | `reports/data_analysis/` |
| Experiment grid E2–E6 (script 11) | Your laptop or Colab (CPU) | ~10 min | `runs/exp_*/`, `reports/experiments/`, `reports/bias/exp_*/` |
| Full-data TF-IDF baseline (script 04) | Colab, or a laptop with 8 GB+ free | ~50 min on an 8 GB Mac | `runs/baseline_tfidf_lr/` |
| DistilBERT (script 05) | **Kaggle GPU** (or Colab T4) | ~20 min for 30k rows; hours for full data | `runs/distilbert_*/` |

`data/`, `runs/` and `reports/` are not uploaded to GitHub. Download the zip at the end of every cloud run and keep it in the team drive (see "Saving results").

---

## A. Laptop (macOS / Linux)

One-time setup (see `TEAM_GUIDE.md` section 5), then from the repo folder:

```bash
source .venv/bin/activate
python scripts/01_download_data.py
python scripts/03_make_splits.py --subset 30000 --subset-strategy rare_preserving --undersample-clean
python scripts/03_make_splits.py --subset 30000
python scripts/07_fetch_bias_resources.py
python scripts/08_build_templates.py
python scripts/10_data_analysis.py
python scripts/11_experiments.py
```

The second `03` call restores `split_stats.csv` with the stratified subset. Both calls write identical train/val/test files, because the seed is fixed.

To run only some experiments: `python scripts/11_experiments.py --only E3_lr_no_weights E4_svm`.
To rebuild the table and figures without retraining: `python scripts/11_experiments.py --report-only`.

---

## B. Google Colab (CPU is enough for scripts 10 and 11)

1. Go to <https://colab.research.google.com> → *New notebook*.
2. Paste each block into its own cell and run them in order:

```python
!git clone -q https://github.com/shriniwas2412/iiith-mml-explainable-toxicity.git repo
%cd repo
!pip install -q -e . -r requirements.txt
```

```python
!python scripts/01_download_data.py
!python scripts/03_make_splits.py --subset 30000 --subset-strategy rare_preserving --undersample-clean
!python scripts/03_make_splits.py --subset 30000
!python scripts/07_fetch_bias_resources.py
!python scripts/08_build_templates.py
```

```python
!python scripts/10_data_analysis.py
!python scripts/11_experiments.py
```

```python
!zip -qr results.zip reports runs -x "runs/*/model.joblib" -x "runs/*/model/*"
from google.colab import files
files.download("results.zip")
```

Colab deletes everything when the session ends. Download `results.zip` before closing the tab.

---

## C. Kaggle GPU (DistilBERT)

**One-time:** a Kaggle account with a **verified phone number** (otherwise the GPU and Internet switches stay locked).

1. kaggle.com → *Create* → *New Notebook* → *File* → *Import Notebook* → upload `notebooks/train_gpu.ipynb` from the repo.
2. Right panel → *Session options*: *Accelerator* = **GPU T4 x2** (or P100), *Internet* = **On**.
3. Run the first two code cells (clone + install, GPU check). The GPU line must not say "none".
4. Run **section 1** (data and splits) and **section 2** (DistilBERT on the 30k subset, about 20 minutes).
5. Add a new cell and run the bias audit for that model:

```python
!python scripts/07_fetch_bias_resources.py
!python scripts/08_build_templates.py
!python scripts/09_bias_audit.py --model hf:runs/distilbert_wbce_30k/model --thresholds runs/distilbert_wbce_30k/thresholds.json --out reports/bias/distilbert_wbce_30k
```

6. Add a cell to zip the results without the weights, then download it from the *Output* panel (right side, `/kaggle/working`):

```python
!zip -qr distilbert_30k.zip runs/distilbert_wbce_30k reports/bias/distilbert_wbce_30k -x "runs/*/model/*"
```

7. **Do not** use *Save Version → Save & Run All* unless you want sections 3–4 too (the full-data runs, several GPU hours).

---

## Saving results

1. Unzip the download **into the repo folder on your laptop**, so the files land in `runs/...` and `reports/...` exactly as on the cloud machine.
2. Re-run `python scripts/11_experiments.py --report-only` if you added `exp_*` runs, to refresh the table and figures.
3. Copy the same zip to the team drive, named `<date>_<who>_<what>.zip`, e.g. `2026-10-24_shriniwas_distilbert30k.zip`.
4. Add a row to the team run log: run folder, settings, commit (`git rev-parse --short HEAD`), machine, time.

Only numbers from these files go into slides, the report or the paper.

---

## What each output file is

| File | What it holds |
|---|---|
| `reports/data_analysis/facts.json` | Headline counts (comments, share toxic, identity mentions) |
| `reports/data_analysis/label_rates_train_vs_test.csv` | Label shift between train and the scored test set |
| `reports/data_analysis/length_and_style_by_label.csv` | Words, CAPITALS, "!" and repeated letters per label |
| `reports/data_analysis/artifacts_toxic_vs_clean.csv` | How often URLs, IPs, wiki markup etc. appear: the evidence for the cleaning rules |
| `reports/data_analysis/duplicates.json` | Duplicate texts within and across splits |
| `reports/data_analysis/identity_terms_in_train.csv` | Per identity term: mentions and share toxic (Vaidya et al., Table 2 for our data) |
| `reports/data_analysis/identity_axes_vs_labels.csv` | Per group axis: how often abuse is labelled `identity_hate` (Davidson et al.) |
| `reports/experiments/results.csv` | One row per experiment: F1, PR-AUC, per-label scores, bias rates |
| `reports/experiments/fig_e*.png`, `reports/data_analysis/fig*.png` | Figures for slides |
| `runs/exp_<name>/` | Model, predictions, thresholds, metric tables, settings of each experiment |
| `reports/bias/exp_<name>/summary.json` | Bias audit of each experiment |
