# Contributing

## Setup

```bash
git clone https://github.com/shriniwas2412/iiith-mml-explainable-toxicity.git
cd iiith-mml-explainable-toxicity
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pytest -q
```

Fine-tuning needs a GPU. Use Kaggle or Colab for that; everything else runs on a laptop.

## Workflow

1. Pick a task from your workstream in [docs/ROADMAP.md](docs/ROADMAP.md).
2. Create a branch: `git checkout -b <your-name>/<short-task>`.
3. Make the change. Add or update a test in `tests/` when you add code to `src/toxicity/`.
4. Run `pytest -q` before you push.
5. Open a pull request into `main` and ask one teammate to review it.

Commit under your own name and email. Individual contributions are assessed, so
each person pushes their own work.

## Conventions

- Library code goes in `src/toxicity/`. Scripts in `scripts/` stay thin and call the library.
- Settings go in `configs/default.yaml`, not in the scripts.
- Label order is fixed in `src/toxicity/config.py` (`LABELS`). Use it everywhere.
- Models write predictions with `toxicity.predictions.save_predictions`, so every
  model can be scored by the same scripts.

## What not to commit

- Data, model weights and run outputs (`data/`, `runs/`, `reports/`). They are
  in `.gitignore`; regenerate them with the scripts.
- API tokens or `kaggle.json`.
- Notebooks with large outputs. Clear outputs first.
