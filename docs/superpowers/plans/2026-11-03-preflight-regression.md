# Preflight regression and new holdout plan

Approved direction: keep all prediction before solver execution. Improve sparse-history pooling, execution-start features, point regression, and one-sided budget calibration. User requested implementation and continued testing on the supplied server.

## Fixed design

- Preserve the dependency-free CLI and v2 prediction compatibility. New feature-aware records carry a versioned preflight feature snapshot.
- Pool eligible natural-termination histories by exact execution configuration. Sparse types shrink toward the configuration-wide median instead of falling straight back to rules.
- Fit standardized ridge regression of log(total tokens) using declared preflight numeric features. Select regression only on fit-only deterministic validation against the hierarchical median; otherwise retain the median. Calibration and final test labels never select a model.
- Add a separate 90% single-sided log-residual conformal budget upper bound, calibrated on task-disjoint history. Do not call it a central 90% interval or claim conditional coverage.
- Explicit `--task-kind code_repair` identifies benchmark repair tasks without guessing from issue prose. Expose feature snapshots with safe counts, never tool results or actual future usage.
- Audit exact base-commit repository counts and approximate known initial request scaffold where available. Do not infer natural failure termination from unresolved status alone.

## Tasks

- [x] Add failing tests for hierarchy, regression, single-sided ranks, feature/label isolation, task kind, and CLI record round trips.
- [x] Implement the feature-aware model and integrate into estimate/record commands.
- [x] Update documentation and perform independent code review.
- [x] On the server, keep the original 250-task cohort as development data; freeze the model choices before evaluating the next 250 previously unused source tasks. Keep success-only limitations explicit if natural failure termination cannot be established.
- [x] Run tests, reproduce results, commit/push the authorized repository update and retain all experiment artifacts under SWE.

## Evaluation rules

No tuning after new holdout metrics are opened. Compare identical new tasks using old heuristic, prior ratio predictor, history median, hierarchical median, regression candidate and selected model. Report point errors and one-sided empirical coverage/excess separately. Any improved result is limited to this source/model/cohort.

Execution: 34 tests passed locally and on the server; independent review found no actionable defect. Acquired all 500 exact base-commit tree counts and pre-generation request length estimates. New final evaluation used 158 unseen resolved cases, with 71 fit and 43 calibration cases; original 29 test cases were excluded. The selector retained the median (forced ridge did not win). No parameters changed after final metrics were opened. Publication/push and final reproducibility verification follow this record.
