# Reproduce the external preflight comparison

This is a retrospective fixed-benchmark comparison using the v2 data/partition bundle. The 158 test outcomes were already inspected before these external baselines were proposed. It is not a fresh independent validation.

## Dependencies and source

Use Python 3.12 and a dedicated environment. The product CLI does not require these optional packages.

```bash
python -m venv .venv-experiment
.venv-experiment/bin/python -m pip install -r experiments/requirements-external.txt
git clone https://github.com/DEFENSE-SEU/TokenCast.git /path/to/TokenCast
git -C /path/to/TokenCast checkout af947602da74b9cb6ed127850f531db6187278fa
```

Keep the upstream repository separate and preserve its license. The adapter imports its public source directly; it does not vendor that implementation or load pretrained weights. On a host without GitHub access, transfer a verified source archive from that exact commit. In this experiment the archive SHA-256 was `ee130f93f79ec430e54216852aa073b6a1143e2f47cd926130663eaec6501ec0`.

## Run

```bash
.venv-experiment/bin/python experiments/external_baselines.py \
  --reference-dir /path/to/v2/published-results \
  --data-dir /path/to/v2/data \
  --upstream /path/to/TokenCast \
  --output-dir /path/to/comparison-results
```

If running a copied script outside this repository, add `--estimator-root /path/to/token-budget-estimator`. The reference directory must contain the frozen v2 `history-adaptive.jsonl`, `partitions.json` and `predictions.jsonl`; the data directory supplies the original recorded `summaries.json`. Data acquisition/provenance is described in the v2 report and source manifests. Raw task histories are not automatically published by this repository.

## What is compared

- The frozen current selector and historical median references.
- The frozen ridge candidate, reconstructed from the same 71 fit rows; predictions must agree with the frozen reference before its calibration points are used.
- Plain LightGBM log1p-target median regression, fixed 300 boosting rounds.
- The official public `LightGBMQuantileEstimator` at `TASK_LAUNCH` with identity and log1p targets, adapted to the same eight numeric known-start features and provider input+output total labels.

The official adapter stores the eight numbers in the registered G0 `task_embedding` tuple field. This is a structured count-vector adapter, not a pretrained semantic embedding. Its early stopping uses a hash-selected validation subset inside the 71 fitting rows. It then refits all heads on all 71 fitting rows at the selected counts, retaining the upstream encoder and prediction API. This outer refit is explicitly an adaptation; its metadata records train/validation IDs, parameters and selected rounds.

All point predictors use exactly the same 43 calibration rows for a one-sided 90% log-residual budget. Official native q05/q50/q95 values are saved separately, not presented as the same kind of budget. Fitted models, source/data hashes and predictions are persisted before scoring test labels.

## Identity boundary

Call the official arm **public TokenCast task-launch LightGBM adapter**, not **full TokenCast paper reproduction**. Its native feature set, original data, and compositional/correction architecture are not reproduced. The target in some upstream experiment configurations is output tokens; this comparison explicitly uses total input+output tokens for all arms. Plain LightGBM and the official public tree implementation are related methods, not independent confirmations of an architectural claim.

The 5,000-sample paired task bootstrap is conditional on these fitted models and this success-only source subset. It does not account for retraining uncertainty, test-set reuse, repository clustering or other deployment distributions. Final test outcomes must not select hyperparameters.
