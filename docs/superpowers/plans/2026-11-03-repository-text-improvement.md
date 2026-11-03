# Improve the predictor and run broader baselines

The user clarified that improving their tool is the objective, then explicitly requested several comparison models to be run. Keep prediction strictly before execution. External methods are diagnostics, not the product objective.

## Evidence and fixed changes

The current point forecast is constant in the declared code-repair cohort. Repository-scale counts do not encode repository identity or issue semantics. Diagnostic errors vary markedly by repository, but those already-seen test outcomes cannot select new hyperparameters.

Add optional known-start repository identity and a standard-library fit-only TF-IDF task index. Repository medians shrink toward the task/configuration median with n/(n+10). Retrieval uses the five most similar fit tasks, a cosine-squared weighted median, and a blend weight capped at 0.5. New candidates never query calibration/test labels. Unseen repositories and empty vocabulary fall back safely.

Select among hierarchy, existing ridge, repository shrinkage and repository+retrieval using deterministic task-disjoint 5-fold OOF validation inside fitting history. Require at least 5% aggregate MAE improvement and improvement in at least four usable folds; otherwise retain hierarchy. Refit only the selected candidate on all fitting rows. Keep the existing independent one-sided calibration.

## Baselines and protocol

Add fixed RandomForest, ExtraTrees, CatBoost, ElasticNet, TF-IDF ridge and TF-IDF nearest-neighbor baselines. Reuse the 71 fit / 43 calibration / 158 previously seen benchmark cases with explicit retrospective labeling. All text vocabularies, numeric scalers and categorical encoders fit on fitting rows only. Report numeric-only and repository/text-aware feature conditions separately, not as if extra information were free.

No final-test-based parameter changes. Improve the actual CLI/history schema with optional repository metadata and explanations. Run functional tests, evaluate development OOF and fixed benchmark, preserve failures/limits, and publish the authorized code and evidence. Results need not show improvement; do not force an unvalidated candidate into the default.

## Tasks

- [x] Test repository metadata, fit-only text index, unknown-repository fallbacks and validation gates.
- [x] Implement the new standard-library candidates and CLI wiring.
- [x] Install optional comparison libraries in SWE/v4 and run all named baselines.
- [x] Freeze model choices before retrospective scoring, reproduce results, review code and push.

