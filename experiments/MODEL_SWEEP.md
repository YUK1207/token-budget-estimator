# Repository/text improvement and model sweep

This experiment tests the actual preflight predictor's repository and text candidates, alongside fixed RandomForest, ExtraTrees, CatBoost, ElasticNet, LightGBM, TF-IDF ridge and TF-IDF nearest-neighbor diagnostics. It does not reproduce another paper or call a solver. The product remains standard-library-only; these dependencies are optional experiment tools.

Use an isolated Python 3.12 environment:

```bash
python -m venv .venv-sweep
.venv-sweep/bin/python -m pip install -r experiments/requirements-sweep.txt
.venv-sweep/bin/python experiments/model_sweep.py --reference-dir /path/to/v2/published-results --data-dir /path/to/v2/data --output-dir /path/to/results
```

Inputs are the sanitized pinned source data described in [the original protocol](results/swe-preflight-v2.md), its frozen `history-adaptive.jsonl` and `predictions.jsonl`, plus `tasks.json` and `summaries.json`. On the authorized server these remain under `/root/autodl-tmp/SWE/v2/`; the sweep is under `/root/autodl-tmp/SWE/v4/`. Do not commit private history or credentials.

There are 71 fitting, 43 calibration and 158 already-seen benchmark tasks. Five-fold development diagnostics fit every transform on each training fold only. The product's own selector uses its fixed gate, independently of external-model scores. Numeric methods receive the same eight counts; augmented methods additionally receive fit-only repository one-hot and 256-feature TF-IDF. Text-only methods receive TF-IDF. All targets are log1p total provider input+output tokens. No final-score hyperparameter tuning is performed.

The script writes parameters, source/data hashes and versions before training, then saves fold predictions, model choices, model artifacts and predictions before joining benchmark labels. Independent log-residual calibration uses the same 43 tasks for each fixed method. `models/` contains locally produced joblib/CBM artifacts; only load artifacts you trust.

The [report](results/model-sweep-v4.md) and adjacent JSON evidence preserve unsuccessful candidates as well as successful baselines. This success-only retrospective sample from one model/harness configuration cannot establish new independent generalization, solve rates or token savings. Small ranking differences do not establish statistical superiority.

