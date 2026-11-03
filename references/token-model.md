# Token model

The uncalibrated `heuristic-v2` baseline approximates prompt, initial context, discovery, file reads, command output, retries, analysis and response tokens. Task profiles and coefficients in `scripts/estimate.py` are hand-set rough assumptions. English keywords use word boundaries.

Record the versioned raw baseline before history adjustment. It is a scale feature, not a median. Historical actual/raw-baseline ratios provide empirical quantiles; an independent partition can calibrate intervals. See [method and papers](preflight-method.md).

After total-only historical calibration, input/output components and cost are unknown. Heuristic monetary scenarios accept manual prices without cache discounts.

Cumulative usage and peak context are separate. Without a separately supplied peak, context fit is unknown.

The adaptive default pools configuration history, uses a hierarchical median or fit-validated log ridge, and calibrates a separate one-sided budget upper. The ratio-scaled model remains an explicit legacy comparator. See [adaptive method](preflight-method.md) for feature and partition details.
