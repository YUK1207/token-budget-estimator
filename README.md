# Token Budget Estimator

A pure-preflight Codex skill and Python CLI for estimating total agent input + output tokens **before the solver starts**. Python 3.10+, standard library only.

Inputs: task text, lightweight local repository counts, declared execution configuration, and prior usage records. No solver execution, LLM calls, or running-task trajectory inputs. Post-run recording improves future forecasts only.

[Website](https://yuk1207.github.io/token-budget-estimator/) · [API concept](https://yuk1207.github.io/token-budget-estimator/api.html) · [Method and papers](references/preflight-method.md)

The website's HTTP endpoints and paid plans are design concepts, not implemented services.

## Changes

- English word boundaries prevent `callback` and `small` from matching `all`.
- History matches model/version, agent/version, execution configuration and baseline version. The adaptive predictor pools types; its hierarchical point retains type information.
- The default model pools configuration history and shrinks sparse task types toward the configuration median.
- Optional repository priors and fit-only TF-IDF neighbors add known-start evidence without solver or LLM calls.
- Deterministic five-fold validation inside fitting history selects a candidate only with a 5% MAE advantage and improvement in at least 80% of usable folds; otherwise hierarchy remains active.
- Known-start features include task counts, exact repository counts when supplied, and declared initial context.
- Independent calibration provides a separate one-sided 90% budget upper bound. The prior ratio/central-interval method remains available with `--predictor legacy`.
- Insufficient data explicitly falls back to rough rules. Sample count alone never establishes confidence.
- Peak single-request context is separate from cumulative tokens.
- Prompt suggestions no longer apply fixed 0.45/0.65/0.82 savings multipliers.

This baseline is inspired by quantile calibration and task-start forecasting. It is not a reproduction of TokenCast or learned CQR. Predictive accuracy and token savings remain unvalidated on a held-out benchmark.

## Install

Copy the repository into `~/.codex/skills/token-budget-estimator`, then start a new Codex chat:

```text
$token-budget-estimator Estimate the budget before fixing the failing login tests.
```

Keep the whole `scripts/` directory, including `history_model.py`, `preflight_model.py` and `task_neighbors.py`.

## CLI

From this repository root:

```bash
python scripts/estimate.py estimate --task "Fix failing login tests" --cwd /path/to/target-repo --budget 50000 --json
```

For history matching, use stable exact identifiers. The execution configuration should identify reasoning effort, context/compaction policy, prompt/tool versions and termination limits.

```bash
python scripts/estimate.py estimate --task "Fix failing login tests" --cwd /path/to/target-repo --model model-version --agent harness-version --execution-config reasoning-medium-context-v1 --instance-id repo-issue-123 --history /path/to/private-history.jsonl --json > prediction.json
```

Save this prediction **before** solving. After the run, record provider-accounted usage from the same scope:

```bash
python scripts/estimate.py record --prediction-file prediction.json --actual-tokens 52000 --instance-id repo-issue-123 --outcome completed --history /path/to/private-history.jsonl
```

The frozen prediction preserves the raw baseline, configuration and history digest. Recording does not update an already issued forecast. Do not publish private tasks or histories.

## Declared repair tasks and exact repository counts

For SWE-style issue repair use `--task-kind code_repair`, so prose without words such as "fix" is not mistaken for a simple question. Generic requests retain `--task-kind auto`.

```bash
python scripts/estimate.py estimate --task "Unexpected behavior in parser" --task-kind code_repair --model model-version --agent harness-version --execution-config config-v1 --point-model auto --json
```

The default `--point-model auto` compares hierarchy, ridge, repository shrinkage and repository-plus-text neighbors using five-fold validation inside fitting history. `median`, `ridge`, `repository` and `retrieval` are diagnostic overrides, not parameters to tune on final test outcomes. Standardization and text vocabulary use fitting rows only. Missing evidence falls back to hierarchy.

`--repository owner/repo` supplies known-start repository identity. Otherwise the CLI reads the local git origin without network access and strips URL credentials. Exact snapshot files may provide `repository` or `snapshot_id=owner/repo@commit`. Unknown repositories use the hierarchical prior. JSON includes `prediction_support` and `model_validation` to explain the evidence and fit-only selection.

`--repo-signals-file repo-counts.json` may supply exact snapshot counts instead of a local scan. Required integer fields: `file_count`, `code_file_count`, `test_file_count`, `config_file_count`. Counts must be nonnegative and components cannot exceed total files. Obtain them at the target base commit, never from gold patches or post-fix trees. Repository scans are bounded at 12000 files. Keep snapshot provenance with experiment artifacts.

## Output

| Field | Meaning |
|---|---|
| `low_tokens`, `high_tokens` | Interval; inspect `interval_method` before interpreting coverage |
| `midpoint_tokens` | Arithmetic interval midpoint retained for compatibility; **not P50** |
| `p50_tokens`, `p90_tokens` | New default: unvalidated median proxy in P50, P90 null. Legacy mode retains empirical ratio quantiles |
| `prediction_method` | `heuristic`, `hierarchical_median`, `ridge_log`, `repository_shrinkage`, or `task_similarity_shrinkage`; legacy methods remain available |
| `interval_method` | Default: `heuristic_uncalibrated`, `one_sided_uncalibrated`, or `one_sided_conformal_90` |
| `budget_upper_tokens` | Calibrated one-sided budget upper, or null when unsupported |
| `budget_target_coverage` | Marginal upper-coverage target, not central interval coverage |
| `feature_snapshot` | Versioned preflight-only feature counts used by the model |
| `model_selection` | Fit-only validation decision or explicit override |
| `interval_target_coverage` | `0.9` only for calibrated intervals; a marginal target under exchangeability assumptions, not measured accuracy |
| `fit_samples`, `calibration_samples` | Distinct eligible task counts |
| `confidence` | `low` for fallback; `unvalidated` for historical predictions |
| `baseline_midpoint_tokens` | Versioned, history-free scale for ratio fitting |
| `history_digest` | Fingerprint of selected task identities, splits and usage |
| `budget_status` | `within_budget`, `near_budget`, or `over_budget` |

Five configuration-matched fit tasks enable hierarchy. Automatic selection needs at least thirty fit tasks and four usable validation folds, each with twenty training and three validation tasks. A candidate needs a fixed 5% aggregate MAE advantage over hierarchy and wins in at least 80% of usable folds. Nineteen independent configuration-matched calibration tasks enable the one-sided budget bound. Repository-aware candidates match calibration repository-metadata availability to the query in both directions. These are operational minima, not evidence of sufficient accuracy. Neither calibration nor final test labels select the point model.

With insufficient calibration, the default displays an explicitly uncalibrated [0, scenario upper] using the fitting P90 (extended to the point if needed). `budget_upper_tokens` stays null. A one-sided [0, upper] is not a central uncertainty interval; budget decisions should use the separate calibrated upper field. Legacy mode retains scaled heuristic intervals. Total-only history cannot identify input/output composition or cache pricing: component fields and cost become `null`. Heuristic cost scenarios accept manual input/output prices and omit cache discounts.

`--budget` is advisory and exits successfully on `over_budget`. External launchers must parse JSON to enforce policy. Contracts are suggestions, not runtime enforcement.

## Context

`--context-tokens` is known initial context. `--context-window` is a single-request limit. Without a separately supplied peak estimate, context fit is `unknown`.

```bash
python scripts/estimate.py estimate --task "Fix a bug" --context-window 128000 --peak-context-tokens 64000
```

`fits` (at most 75%), `tight` (within the window), and `likely_exceeds` use only that supplied peak. The tool does not learn peak context from total usage.

## Other commands

```bash
python scripts/estimate.py compare --task "Explain this function" --task "Refactor the entire project" --json
python scripts/estimate.py optimize --task "Fix failing tests" --mode cheap --json
python scripts/estimate.py control --task "Fix failing tests" --budget 50000 --mode balanced --json
```

All accept model/agent/configuration/history options. `optimize` suggests a task contract and reports **savings not measured**; selecting a cheaper mode does not numerically reduce the forecast. `control` adds heuristic ROI, contract and context-diet suggestions. Its score is not measured economic return or success probability.

Project defaults: [.token-budget.json configuration](references/project-config.md). CLI flags take precedence.

## History and evaluation

- `--split auto` partitions canonical instance IDs deterministically into fit/calibration. Explicit `fit`, `calibration` and `test` are supported. Repeated IDs or normalized task text are grouped; no task can cross partitions.
- Use the first eligible run per task. Test groups are excluded from both fit and calibration. Keep a separate frozen training history for strict benchmark evaluation.
- Always pass `--instance-id` when predicting. Current ID and matching task text are excluded. Canonical IDs must remain consistent across prompt variants.
- Natural solver success (`completed`) and failure (`failed`) are eligible. `capped` and `environment_error` runs are recorded but excluded: their observed tokens are not natural completion costs.
- Target provider input + output across one solver run. Count cache-read input once; do not add reasoning tokens twice if included in output. Track setup, evaluator and environment-retry costs separately.
- Legacy `record --task ... --task-type ... --predicted-low ... --predicted-high ... --predicted-midpoint ... --actual-tokens ...` remains accepted as schema v1, but cannot calibrate v2 forecasts without configuration/raw baseline.
- On untouched tasks report point MAE/WAPE and bias, interval coverage **and width**, and solver success/cost for any proposed policy. Compare against raw heuristics and a same-configuration historical median.
- Three runs in one repository do not establish generalization. Interval hits and unit tests do not establish predictive accuracy.

## Tests

```bash
python -m unittest discover -s scripts -v
```

Synthetic tests verify calculations, CLI behavior, task isolation and regressions. They do not measure forecast quality.

## Offline evaluation

A [158-task held-out historical evaluation](experiments/results/swe-preflight-v2.md) achieved 89.87% one-sided upper coverage. The selector retained the median baseline; the ridge candidate did not establish a point-quality gain. Point WAPE remained 61.3%. These success-only historical results do not establish new solve rates, generalization or token savings.

A [fixed-protocol external comparison](experiments/results/external-baselines-v3.md) adds plain LightGBM and an official public TokenCast task-launch tree adapter on the same 158 previously seen tasks. The adapter was close to the median with a paired uncertainty interval spanning zero; no superiority to the full TokenCast paper is established. [Reproduction setup](experiments/EXTERNAL_BASELINES.md).

The [repository/text improvement and broader comparison](experiments/results/model-sweep-v4.md) adds six comparison methods and separates numeric-only from text/repository-aware inputs. The new candidates did not improve this retrospective benchmark; fit-only validation correctly retained hierarchy. [Reproduction setup](experiments/MODEL_SWEEP.md).

The [token and budget validation](experiments/results/budget-validation-v5.md) adds paired error comparisons, 80/90/95% budget diagnostics, and a history-size curve. It reuses the inspected benchmark and does not establish monetary savings. [Protocol and reproduction](experiments/BUDGET_VALIDATION.md).
