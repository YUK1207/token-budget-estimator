# Pure preflight calibration implementation plan

**Goal:** Keep forecasting entirely before solver execution, improve historical estimates, and remove unsupported savings claims.

**Architecture:** Preserve the standard-library CLI and heuristic fallback. Add a configuration-matched empirical quantile predictor over actual/heuristic-baseline ratios, with task-disjoint fit/calibration data and split-conformal interval expansion. This is a lightweight baseline inspired by quantile calibration, not a reproduction of TokenCast or learned CQR.

**Constraints:** No solver execution or online forecasting. No model calls or extra runtime dependencies. Legacy history remains readable but cannot enter configuration-specific calibration. Raw cumulative usage, peak context, and monetary cost are separate quantities. No accuracy or savings claim without held-out evidence.

## Tasks

- [x] Add regression tests for word matching, peak-context semantics, and unsupported optimizer savings; run red, implement fixes, run green.
- [x] Add `scripts/history_model.py` and tests for configuration isolation, task deduplication, test exclusion, malformed data, empirical quantiles, finite-sample calibration, and baseline-version isolation.
- [x] Integrate forecast metadata and configuration flags into all preflight commands; extend record with task identity, split, raw baseline, and configuration. Test CLI round trips.
- [x] Update README, SKILL, method references and static site claims to explain pure preflight operation, data limits and paper attribution.
- [x] Run the full unittest suite, CLI smoke checks and diff checks; review and prepare the authorized commit/push.

## Review focus

No self-training on the current instance; no mixing capped or environment-failed runs with completed solver runs; no fabricated P50 from interval midpoint; no CQR guarantee under distribution shift; no use of component token splits or prices after total-only calibration.

## Execution record

The user approved the preflight-only direction and requested implementation and push. Use the existing clean review clone on `codex/preflight-calibration`; push the validated commit to main by fast-forward only. Do not publish local histories or the screenshot.

Validation: 25 unittest cases pass; CLI smoke check and git diff --check pass. Independent review found duplicate task leakage and malformed history fields; both were reproduced with failing tests and fixed. A bounded recheck found no additional blocking issues. Calibrated lower endpoints no longer use approximate prompt lengths.
