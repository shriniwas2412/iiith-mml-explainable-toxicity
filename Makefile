PYTHON ?= python
RUN ?= runs/baseline_tfidf_lr

.PHONY: setup test data eda splits baseline thresholds templates audit

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) scripts/01_download_data.py

eda:
	$(PYTHON) scripts/02_eda.py

splits:
	$(PYTHON) scripts/03_make_splits.py --subset 30000

baseline:
	$(PYTHON) scripts/04_train_baseline.py --out $(RUN)

thresholds:
	$(PYTHON) scripts/06_tune_thresholds.py --run $(RUN)

templates:
	$(PYTHON) scripts/07_fetch_bias_resources.py
	$(PYTHON) scripts/08_build_templates.py

audit:
	$(PYTHON) scripts/09_bias_audit.py --model baseline:$(RUN)/model.joblib \
		--thresholds $(RUN)/thresholds.json --out reports/bias/baseline
