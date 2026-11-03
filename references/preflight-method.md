# Pure-preflight forecasting method

## Scope

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
