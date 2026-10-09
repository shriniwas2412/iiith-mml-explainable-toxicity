# Roadmap

MML Capstone 4 — Multi-Label Toxic Comment Classification with Explainability.
Team of five. Project lead: Shriniwas Maheshwari.

## What we are building

1. A multi-label classifier for the six Jigsaw labels (toxic, severe_toxic,
   obscene, threat, insult, identity_hate).
2. One decision threshold per label, tuned on validation data.
3. Explanations for individual predictions (LIME / SHAP).
4. A bias test: does the model flag harmless sentences just because they
   mention an identity group?
5. A deployed demo and a deployment brief.

Extension, if time allows: run the bias test on Indian identity terms (caste,
religion, region) as well as the standard Western ones.

## Workstreams

One owner per workstream. Owners write the code, the tests and the matching
section of the report. Fill in the names.

| # | Workstream | Owner | Main files |
|---|---|---|---|
| 1 | Data, EDA and baseline | _TBD_ | `scripts/01`–`04`, `src/toxicity/data.py`, `baseline.py` |
| 2 | DistilBERT training and thresholds | _TBD_ | `scripts/05`–`06`, `train.py`, `losses.py`, `metrics.py` |
| 3 | Explainability | _TBD_ | `src/toxicity/explain.py` (to write), `scripts/10_explain.py` (to write) |
| 4 | Bias audit | _TBD_ | `scripts/07`–`09`, `templates.py`, `bias.py` |
| 5 | Demo, report and slides | _TBD_ | `demo/`, `report/` |

The lead also reviews pull requests, keeps this file current and runs the
weekly sync before the Sunday mentoring session.

## Checkpoints

Dates come from the programme. Fill them in when announced.

| Checkpoint | Date | Done when |
|---|---|---|
| Proposal (7–8 pages) | _TBD_ | Problem, literature review, method, staged outcomes, applicability |
| 1. Preliminary model on a subset | _TBD_ | EDA tables, fixed splits, baseline and DistilBERT on a 30k subset, metrics table |
| 2. Full-data training | _TBD_ | DistilBERT on all training data, weighted BCE vs focal loss, tuned thresholds |
| 3. Deployment testing | _TBD_ | Demo running, explanations for 15–20 predictions, bias audit results |
| 4. Final deployment | _TBD_ | Final model, deployment brief, report, slides, live demo |

## Status

| Item | State |
|---|---|
| Project skeleton, config | Done |
| Data loading, EDA, stratified splits | Code done, not yet run on the full data |
| Metrics and threshold tuning | Code done |
| Baseline | In progress |
| DistilBERT training | In progress |
| Bias audit | In progress |
| Explainability | Not started |
| Demo | Not started |
| Report and slides | Not started |

## Ground rules

- Every number in the report comes from a file produced by a script in this repo.
- Thresholds are tuned on validation only. The test set is used once, at the end.
- The data contains offensive language. Do not paste raw examples into chats
  or slides without a content note.
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to work in the repo.
