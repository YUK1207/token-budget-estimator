# Offline preflight evaluation, 2026-11-03

## Protocol

Public OpenHands CodeActAgent trajectories for Claude 3.7 Sonnet, first source run_4. Source dataset revision: `fa9cbb063f770df596da95af24f7af3b8f595778`. No new model calls, patches, or official patch-scoring runs were performed.

Of 500 recorded tasks, 301 were resolved with valid usage. The original 29 pilot test tasks were excluded from fitting and calibration. The first 250 source tasks supplied 114 development cases (71 fit, 43 calibration). The previously unused second 250 supplied 158 final test cases. The 199 unresolved runs could not reliably distinguish natural failure from truncation and were excluded.

Inputs include issue text, git-tree counts at the exact base commit (bounded at 12000 blobs), and an approximate initial-request/scaffold length calculated from pre-generation request text. No future usage or patch outcomes are model features. Model choices and parameters were frozen before final scoring.

## Results on the same 158 final test tasks

| Method | MAE (tokens) | WAPE | Upper coverage | Mean upper (tokens) |
|---|---:|---:|---:|---:|
| configuration_median | 847,577 | 61.27% | N/A | N/A |
| hierarchical_median | 847,577 | 61.27% | 89.87% | 2,919,000 |
| prior_ratio | 956,700 | 69.16% | 74.05% | 2,967,918 |
| raw_rules | 1,268,559 | 91.70% | 10.76% | 225,354 |
| ridge_candidate | 850,918 | 61.51% | 91.14% | 3,298,601 |
| selected_model | 847,577 | 61.27% | 89.87% | 2,919,000 |

## Interpretation

The automatic selector retained the hierarchical median on fit-only validation. Its point equals the configuration median in this declared code-repair cohort, so this experiment does not establish point-prediction gains beyond that baseline. The forced ridge candidate was slightly worse on final-test MAE and used a larger average upper budget. No final-test result was used to change the selector.

The selected model reduced point MAE by about 11.4% relative to the prior ratio comparator, but WAPE remained 61.3%. The 90% one-sided calibration target achieved 89.87% empirical upper coverage, with an average upper of 2.919 million tokens. This is not a central uncertainty interval, a per-task guarantee, or evidence of token savings.

These results are not directly comparable to the original 29-task pilot: tasks and feature availability differ. Compare methods only within the same table.

## Limits and reproduction

This is a success-only, single-model/harness, archive-order historical cohort. It does not measure natural failures, cross-model or leave-repository-out generalization, end-to-end runtime cost, or a new SWE-bench solve rate. Repeated deterministic evaluation is a reproducibility check, not another independent test.

Use `experiments/evaluate_swe_preflight.py` with the sanitized data bundle. It saves input/code hashes, task partitions, frozen predictions and per-task metrics. Dataset preparation streams the public archive and sums recorded provider input plus output; cache-read counts are not added twice. Future changes need a fresh final holdout.

[Trajectory dataset](https://huggingface.co/datasets/loong0814/openhands_trajectories) · [SWE-bench Verified tasks](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
