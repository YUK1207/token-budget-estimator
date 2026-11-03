# Adaptive pure-preflight model

Only pre-execution information is used. The prediction does not execute a solver, inspect a running trajectory or call an LLM. Post-run outcomes are labels for future tasks. The default predictor is `adaptive`; the former ratio method is retained under `legacy` for reproducible comparisons.

## Features and point model

Feature schema `preflight-counts-v1` contains log(1 + task characters), log(1 + line breaks), log(1 + code files), log(1 + test files), a repository-known indicator, log(1 + declared initial context), log(1 + path mentions), and log(1 + error-word mentions). File counts must refer to the target base commit. Initial request length is a declared pre-generation feature, not usage returned after generating a response. No future call count, test outcome, patch or actual total is a feature.

History is pooled across task types only within the exact model/harness/configuration and raw baseline version. ID/text groups are excluded and deduplicated before fitting. For type k with n_k fit records, the hierarchical log point is:

```text
w = n_k / (n_k + 10)
log point = w * log(type median actual) + (1-w) * log(configuration median actual)
```

If the type is absent, use the configuration median. At least five fit tasks are needed. Records without supported feature snapshots still support the hierarchy.

The ridge candidate uses standardized features and log(total tokens) targets. Feature means/scales use fitting data only; the fixed L2 penalty is 10. An unpenalized intercept is adjusted by the training median log residual. Standardized query values are clipped to [-5, 5] as a fixed numerical extrapolation guard. The exponentiated point is a median proxy under modeling assumptions, not a proven conditional median.

Optional repository identity is metadata alongside the unchanged eight numeric features. A repository log-median prior shrinks toward the hierarchical log point with weight `n_repo/(n_repo+10)`. Unknown repositories retain hierarchy. The text candidate builds a standard-library TF-IDF index exclusively from fitting task descriptions. It uses the five positive-cosine nearest tasks, their cosine-squared weighted median log cost, and a blend weight `min(0.5, sum(weights)/(1+sum(weights)))` toward the repository prior. Empty or unmatched text retains the prior. Neither candidate adds model calls.

For automatic selection, a stable task-ID hash assigns five folds within fitting history. Require at least thirty fit records and at least four usable folds, each with twenty training and three validation tasks. Compare hierarchy, ridge, repository and repository-plus-text candidates. A candidate must achieve aggregate validation MAE below 95% of hierarchy and improve at least `ceil(0.8 * usable_folds)` folds. Select the eligible candidate with lowest MAE; otherwise retain hierarchy. Every fold rebuilds text vocabulary, priors and ridge scaling on its training portion only. Refit the selected candidate on all fitting rows; calibration and final test labels never select it. Explicit overrides are diagnostics and must not be chosen using final test results. JSON exposes the validation report and query support counts.

## One-sided budget calibration

For the fixed selected point function p(x), use independent calibration scores `s_i = log(actual_i) - log(p(x_i))`. Select rank `ceil((n_cal + 1) * 0.90)`. Require nineteen calibration tasks; if the rank is unavailable, abstain from a calibrated upper. The budget is `ceil(exp(log(p(x)) + max(0, kth score)))`, rounded outward by the CLI. It is also conservatively extended when the declared initial request estimate is larger.

This is one-sided calibration: the event is `actual <= budget_upper`, with a 90% marginal target under exchangeability. It is not a central 90% interval, a per-task guarantee or a claim of empirical accuracy. `low_tokens=0` is a nonnegative support endpoint, not a learned lower quantile. Compare empirical upper coverage and excess budget on held-out tasks. For a ridge, calibration rows need compatible feature snapshots; missing-feature calibration cannot validate a feature-bearing model.

For repository and text candidates, calibration repository-metadata availability must match the query in both directions. This keeps the same missing-metadata prediction branch at calibration and prediction time. Fewer than nineteen matching calibration tasks means no calibrated budget. Availability matching alone does not establish conditional coverage by repository.

Without calibration, the high scenario uses the fit empirical P90 extended to at least the point, labeled `one_sided_uncalibrated`; the separate calibrated budget field remains null. The point field `p50_tokens` is retained for compatibility but `point_semantics` explicitly says median proxy. Legacy ratio P90 remains available only in legacy mode.

## Data boundaries and evaluation

Schema-v2 records preserve versioned feature snapshots and frozen predictions. Exact profile identity, task grouping, natural-termination filtering and held-out test exclusion apply configuration-wide. Completed successes and confirmed natural solver failures are eligible; caps and infrastructure failures are excluded. An unresolved benchmark label alone does not prove natural termination.

Point quality, single-sided coverage, and budget excess must be evaluated separately. The second offline experiment originally reserved the second 250 source tasks for evaluation; the original 29 test tasks remained diagnostic only. Those results have now been inspected, so subsequent comparisons on its 158 eligible successful tasks are explicitly retrospective, not a fresh holdout. Source selection still restricts the population to successful historical runs until failure termination can be established.

Run `experiments/evaluate_swe_preflight.py --data-dir <sanitized-data> --output-dir <results>` to reproduce this bounded experiment. It is not the official SWE-bench patch evaluator. Data/source hashes and frozen predictions are saved before scoring.

## Paper attribution

CQR motivates independent calibration, but this implementation does NOT implement learned conditional quantile regression. TokenCast motivates distinguishing task-start from online forecasts; no online architecture is copied. Log ridge, hierarchy, validation selection and the one-sided budget interface are this repository's implementation choices. Related sources and the prior comparator are documented below.

---

# Legacy ratio comparator (`--predictor legacy`)

## Legacy scope

Prediction inputs exist before the solver starts: task text, local repository counts, known initial context, model/version, harness/version and execution configuration. No trajectory prefixes, tool outcomes, hidden tests, gold patches or solver calls are prediction inputs. Historical outcomes are labels for later tasks.

Configuration IDs must distinguish reasoning effort, prompt/tool versions, context retention/compaction and termination limits. Changing settings requires a new ID. Matching also requires task type and raw baseline version `heuristic-v2`.

## Scale model

Let `b(x)` be a positive history-free heuristic midpoint. Fit prior actual/baseline ratios `r_i = y_i / b(x_i)` using eligible historical runs.

```text
P50 = b(x) * median(fit ratios)
P90 = b(x) * empirical-quantile(fit ratios, 0.90)
raw interval = b(x) * [q05(fit ratios), q95(fit ratios)]
```

Use nearest-rank empirical quantiles and a separate median. This is a restricted scale baseline, not learned conditional quantile regression. The heuristic captures coarse task/repository differences; ratios capture configuration-specific bias. No P50 is defined as an interval midpoint.

Five fit tasks enable historical points. Before calibration, retain the median-scaled heuristic interval extended to include empirical P90, marked uncalibrated. With fewer fit tasks, report only the original rough interval.

## Independent calibration

Require ten fit tasks and nineteen calibration tasks. From fit-only q05/q95, compute calibration scores:

```text
s_j = max(q05 - r_j, r_j - q95)
k = ceil((n_calibration + 1) * 0.90)
d = max(0, kth-smallest score)
interval = b(x) * [max(0, q05 - d), q95 + d]
```

If the rank exceeds calibration size, no finite calibrated interval is issued. Lower endpoints round down and upper endpoints round up. The CLI can extend the upper endpoint using the estimated initial request and rounds it outward. It never raises calibrated lower bounds using approximate prompt lengths.

The 90% target is marginal over exchangeable future tasks/runs under the same configuration and sampling protocol. It is not a per-task conditional guarantee, measured accuracy claim or guarantee under distribution shift. Thresholds are operational minima, not statistical sufficiency. Report coverage and width/interval score together.

## Partitions

Schema v2 stores a frozen prediction, raw baseline/version, canonical task ID, configuration, actual total usage, outcome, partition and timestamp. Never normalize against a previously adjusted point: this would cause calibration feedback.

`auto` assigns canonical groups by SHA-256 parity to fit/calibration. Explicit `fit`, `calibration` and `test` are supported. Group records linked by canonical ID or normalized task text. Select the first eligible natural-termination run per group without conditioning on usage or success. Repeats are not additional independent samples. Propagate test/current-task exclusion across groups. Consistent canonical IDs across paraphrases remain the caller's responsibility.

Fit quantiles never use calibration values. Test values fit neither stage. Choose benchmark partitions in advance, freeze history, issue predictions before solving, and evaluate untouched outcomes. Deterministic partitioning does not itself prove independence or exchangeability.

Natural solver successes (`completed`) and failures (`failed`) are eligible. Cap terminations are censored; infrastructure failures concern a different cost process. `capped` and `environment_error` are stored but excluded. The target is one naturally terminated solver run, not end-to-end cost including retries. Legacy rows lacking configuration/raw baseline are retained but excluded.

## Accounting and evaluation

Target provider input + output across all calls of one solver run. Include cache-read input once; do not double-count reasoning already included in output. Keep setup, evaluator and infrastructure retries separate. Total-only history cannot establish component splits or cached monetary costs; these fields remain unknown.

Peak single-request context needs an independent estimate, not cumulative usage. Supplied peak checks are scenarios, not learned predictions.

Use untouched tasks and fixed model/harness/configuration. Compare raw heuristic, same-configuration historical median, and normalized historical quantiles. Report MAE, WAPE, signed bias, interval coverage and width/interval score. Preserve failures and termination reasons. Savings and success tradeoffs require separate paired policy experiments including prediction overhead.

The initial three requests-repository runs motivated changes but are not an independent test of this version. Synthetic unit tests validate software behavior only.

## References and attribution

- [Romano, Patterson and Candes, Conformalized Quantile Regression, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html): independent conformal calibration of quantile intervals. This implementation uses constant ratio quantiles and split-conformal expansion, not learned conditional quantile regressors.
- [Ouyang et al., TokenCast: Forecasting Token Consumption During LLM Agent Execution, arXiv:2609.35760v2, 2026 preprint](https://arxiv.org/abs/2609.35760v2): task-start versus online forecasting and repeated context-input accounting. No online updates, LightGBM models, segment composition or pretrained weights are reproduced. Explicit repeated-context modeling remains future work.
- [Bai et al., How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks, arXiv:2604.22750, 2026 preprint](https://arxiv.org/abs/2604.22750): motivates configuration-specific accounting and treating task difficulty/self-estimation as uncertain predictors.

This project does not claim the empirical results of these papers. Future learned quantile models should retain fallback, frozen baselines and disjoint partitions, and demonstrate gains on untouched tasks before replacing this baseline.
