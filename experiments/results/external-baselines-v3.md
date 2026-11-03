# External preflight baselines: retrospective fixed benchmark

## Scope

This reuses the previously inspected 158-task v2 benchmark. It is not a new independent validation. The fitting set has 71 tasks and the independent calibration set has 43 tasks. All arms receive the same eight versioned known-start numeric features and provider input+output total labels. No model calls or new patch evaluation were run.

Official source: DEFENSE-SEU/TokenCast at `af947602da74b9cb6ed127850f531db6187278fa`. Runtime: Python 3.12, LightGBM 4.6.0, NumPy 1.26.4, SciPy 1.11.4. The adapter calls the public TASK_LAUNCH LightGBMQuantileEstimator; it is NOT the full paper compositional/correction pipeline or its native feature configuration.

## Results

| Method | MAE (tokens) | WAPE | Single-sided upper coverage | Mean upper budget |
|---|---:|---:|---:|---:|
| Historical median | 847,577 | 61.27% | 89.87% | 2,918,237 |
| Current selector (median fallback) | 847,577 | 61.27% | 89.87% | 2,918,237 |
| Fixed ridge reference | 850,918 | 61.51% | 91.14% | 3,298,135 |
| Plain LightGBM log-median | 911,252 | 65.87% | 91.77% | 3,703,215 |
| Official public Task Launch adapter (identity) | 850,021 | 61.44% | 89.87% | 2,960,520 |
| Official public Task Launch adapter (log1p) | 850,104 | 61.45% | 89.87% | 2,945,808 |

## Paired differences from the historical median

Positive MAE difference means worse prediction error. The following 95% paired task-bootstrap intervals use 5,000 resamples with seed 20261103 and are conditional on fixed fitted models. They do not account for retraining, repository clustering or reuse of known test outcomes.

| Method | MAE difference, mean (tokens) | Bootstrap interval |
|---|---:|---|
| Fixed ridge reference | 3,342 | [-20,818, 26,544] |
| Plain LightGBM log-median | 63,676 | [15,346, 114,383] |
| Official public Task Launch adapter (identity) | 2,445 | [-2,226, 7,253] |
| Official public Task Launch adapter (log1p) | 2,528 | [-1,854, 7,052] |

## Interpretation

The current selector retained the historical median and has no demonstrated point-quality gain beyond that baseline. Plain LightGBM was worse under this fixed small-data configuration. The official public task-launch adapters were close to the median; their paired intervals cross zero, so this comparison does not establish superiority of either method. The official public tree adapter and plain LightGBM are related tree methods, not independent confirmations of an architectural result.

All arms use the same one-sided log-residual budget calibration. Native official q05/q95 predictions are retained separately, not treated as equivalent to this upper budget. More coverage with a larger upper budget is a tradeoff, not an unconditional improvement.

The source population contains only successful historical runs from one model/harness. These fixed configurations were not tuned on the 158 outcomes; neither the result nor its bootstrap intervals establish cross-model, cross-repository or failure-task generalization. The official adapter target/features/refit protocol differs from the paper, so the table must not be described as beating TokenCast paper results.

## Reproduction

[Adapter setup and exact scope](../EXTERNAL_BASELINES.md) · [Comparison script](../external_baselines.py) · [Frozen protocol](../../docs/superpowers/plans/2026-11-03-external-baselines.md)

Official early stopping used 53 fit-only training tasks and 18 fit-only validation tasks, then all 71 fitting tasks were used for outer refit. Selected rounds: identity q05/q50/q95 = 1/6/9; log1p = 1/5/158. Hyperparameters, transforms and selection rules were fixed before scoring. Repeated execution reproduced metrics byte-for-byte; this is a reproducibility check, not another independent test.

[Upstream repository](https://github.com/DEFENSE-SEU/TokenCast) · [Source trajectories](https://huggingface.co/datasets/loong0814/openhands_trajectories)
