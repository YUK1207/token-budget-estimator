# Token forecast and budget validation

This standard-library experiment follows the user's requested focus: token prediction and budget usefulness, without cross-repository testing or changes to product parameters. Use the existing frozen v2 and v4 bundles:

```bash
python -m unittest discover -s experiments -p test_budget_validation.py -v
python experiments/budget_validation.py --reference-dir /path/to/v2/published-results --data-dir /path/to/v2/data --sweep-dir /path/to/v4/results --output-dir /path/to/v5/results
```

The script freezes its protocol and hashes before predictions, then joins the previously inspected test labels only after saving forecasts. The split remains 71 fit / 43 calibration / 158 successful tasks. All task IDs and normalized task descriptions are checked for cross-partition overlap. Input accounting, source revision and success-only selection are inherited from [v2](results/swe-preflight-v2.md).

The experiment reports point errors and conditional 5000-replicate paired task-bootstrap intervals; these omit retraining and repository clustering uncertainty. The external predictors are the unchanged frozen numeric/text arms from [v4](MODEL_SWEEP.md). Comparisons are exploratory, without multiple-comparison adjustment. One-sided upper budgets are evaluated at 80%, 90% and 95% using rank `ceil((n_cal+1)*coverage)` in sorted log residuals. Only 90% is a product output; the other targets are experimental calculations. Wilson intervals describe observed coverage under an independent-task binomial approximation, not a conditional guarantee.

History-size curves use 20 fixed seeds, nested fit-only subsets of 5/10/20/50/71 tasks, and the same 43 calibration and 158 benchmark tasks throughout. Report fit and total history separately. Resampling P10–P90 ranges are descriptive, not independent generalization confidence intervals. At size 71 all samples contain identical history, so there is no subset variation.

Candidate overrides and a no-calibration fitting-P90 scenario are labeled ablations. A constant calibration-quantile reference verifies that the current constant hierarchy's calibrated bound is exactly that reference; do not claim task-adaptive coverage from it. Configuration-matching and task-type shrinkage cannot be identified on this single-profile, single-task-type dataset. Caps and unverified termination failures remain excluded.

No new solver calls, official patch scoring, real cost savings or fresh independent holdout results are produced. The [report](results/budget-validation-v5.md) records the observations and these limits. All new artifacts stay under the established `SWE/v5` server directory or the matching local artifact directory.
