# External preflight baseline protocol

User approved same-protocol LightGBM and TokenCast Task Start comparisons. All inference remains before solver execution, on the supplied server under SWE/v3. No paid model calls or new patch evaluation.

## Frozen protocol before scoring

- Reuse the v2 partition: 71 point-fitting records, 43 independent calibration records, 158 benchmark test records. The 158 outcomes were seen in v2, so this is a retrospective fixed benchmark, NOT a new independent test.
- Eight versioned preflight features and the same provider input+output total target for all arms. No post-launch features, pretrained checkpoints, gold patches or additional labeled tasks.
- Retain the existing median/current selector and ridge candidate predictions as fixed references.
- Add a plain LightGBM median regressor, fixed 300 rounds, log1p target. Use upstream task-launch parameters (learning rate .03, 15 leaves, minimum leaf 10, depth 6, L2=1, CPU/deterministic).
- Run the unmodified upstream LightGBMQuantileEstimator at TASK_LAUNCH for identity and log1p targets. Pack the same eight numeric features into the registered G0 task_embedding vector; this is a structured numeric adapter, not a pretrained semantic embedding.
- Official parameters are pinned from four_position_billed_v1.toml at upstream af947602da74b9cb6ed127850f531db6187278fa. Select boosting rounds on a fit-only hash validation subset (task_key % 5 == 0), then refit boosters on all 71 fitting records. Document this outer refit adaptation.
- For comparable budgets, apply the SAME one-sided 90% log-residual calibration on the 43 calibration records to every external point predictor. Official native q05/q95 outputs are retained for audit but are not compared as identical to a single-sided budget.
- Save code/data hashes, fitted models and test predictions BEFORE scoring. No changes to parameters after opening results. Pairwise task bootstrap (seed 20261103) describes test-sample uncertainty conditional on fitted models.

## Identity boundaries

The official runtime task-launch LightGBM adapter is not a reproduction of TokenCast's full paper compositional/correction pipeline, native feature set, dataset or reported scores. Do not label adapted results simply as 'TokenCast paper'. Upstream source stays separate from this repository.

## Tasks

- [x] Validate partition, feature whitelist and common upper calibration helpers with tests.
- [x] Install optional dependencies in an isolated server environment and import pinned upstream source.
- [x] Implement and review the adapter; freeze and run once without tuning test outcomes.
- [x] Reproduce outputs, document exact scope and results, push authorized experiment code/results.

Execution: 38 standard-library tests pass. Both official target transforms passed synthetic API smoke tests after correcting tuple-vector packing. Independent source review found an omitted ridge reference; it was restored with frozen-point equality checks. Source archive hash was verified on server. Fixed benchmark metrics reproduced byte-for-byte. No hyperparameters changed after scoring. Official adapters remained close to the median; paired intervals cross zero. Publication and final server sync follow.
