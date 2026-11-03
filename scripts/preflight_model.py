"""Feature-aware, pure-preflight costs with configuration-wide budget calibration."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from statistics import mean, median

import history_model as history

FEATURE_VERSION = 'preflight-counts-v1'
FEATURE_NAMES = ('log_task_chars', 'log_lines', 'log_code_files', 'log_test_files',
                 'repo_known', 'log_initial_context', 'log_paths', 'log_errors')


@dataclass(frozen=True)
class Forecast(history.HistoricalForecast):
    point_method: str = 'heuristic'
    interval_method: str = 'heuristic_uncalibrated'
    budget_upper_tokens: int | None = None
    selection: str = 'insufficient_history'


def feature_snapshot(task: str, file_count: int, code_count: int, test_count: int,
                     repo_known: bool, initial_context: int) -> dict:
    """Only declared inputs; file_count is accepted for future schema compatibility."""
    values = [math.log1p(len(task)), math.log1p(task.count('\n')),
              math.log1p(code_count), math.log1p(test_count), float(repo_known),
              math.log1p(initial_context),
              math.log1p(len(re.findall(r'[\w./-]+\.(?:py|js|ts|go|rs|java|cpp|c|h)\b', task))),
              math.log1p(len(re.findall(r'traceback|exception|error|failed|失败|报错', task.lower())))]
    return {'version': FEATURE_VERSION, 'values': dict(zip(FEATURE_NAMES, values))}


def feature_vector(snapshot: object) -> list[float] | None:
    if not isinstance(snapshot, dict) or snapshot.get('version') != FEATURE_VERSION:
        return None
    values = snapshot.get('values')
    if not isinstance(values, dict):
        return None
    result = [values.get(name) for name in FEATURE_NAMES]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v >= 0 for v in result):
        return None
    return [float(v) for v in result]


def valid_snapshot(snapshot: object) -> bool:
    return feature_vector(snapshot) is not None


def row_snapshot(row: dict) -> dict | None:
    snapshot = row.get('feature_snapshot')
    if snapshot is None and isinstance(row.get('prediction'), dict):
        snapshot = row['prediction'].get('feature_snapshot')
    return snapshot


def hierarchy(rows: list[dict], kind: str) -> float:
    """Geometrically shrink a sparse type median toward its configuration median."""
    global_log = math.log(median(r['actual_tokens'] for r in rows))
    local = [r['actual_tokens'] for r in rows if r['task_type'] == kind]
    if not local:
        return global_log
    weight = len(local) / (len(local) + 10.0)
    return weight * math.log(median(local)) + (1 - weight) * global_log


def solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Pivoted elimination for a small positive ridge system (no extra dependency)."""
    n = len(target)
    augmented = [list(a) + [b] for a, b in zip(matrix, target)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-12:
            raise ValueError('singular ridge system')
        augmented[col] = [v / scale for v in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [v - factor * ref for v, ref in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


@dataclass(frozen=True)
class Ridge:
    means: list[float]
    scales: list[float]
    weights: list[float]
    intercept: float

    def predict_log(self, snapshot: dict) -> float:
        vector = feature_vector(snapshot)
        if vector is None:
            raise ValueError('missing or incompatible feature snapshot')
        standardized = [max(-5.0, min(5.0, (v - m) / s))
                        for v, m, s in zip(vector, self.means, self.scales)]
        return self.intercept + sum(w * x for w, x in zip(self.weights, standardized))


def fit_ridge(rows: list[dict]) -> Ridge | None:
    rows = [r for r in rows if valid_snapshot(row_snapshot(r))]
    if len(rows) < 20:
        return None
    vectors = [feature_vector(row_snapshot(r)) for r in rows]
    targets = [math.log(r['actual_tokens']) for r in rows]
    means = [mean(col) for col in zip(*vectors)]
    scales = [max(1e-8, math.sqrt(mean((v[j] - means[j]) ** 2 for v in vectors)))
              for j in range(len(FEATURE_NAMES))]
    x = [[(value - center) / scale for value, center, scale in zip(v, means, scales)] for v in vectors]
    intercept = mean(targets)
    n = len(FEATURE_NAMES)
    # Fixed regularization: no calibration/test label selects this value.
    matrix = [[sum(v[i] * v[j] for v in x) + (10.0 if i == j else 0.0)
               for j in range(n)] for i in range(n)]
    rhs = [sum(v[i] * (y - intercept) for v, y in zip(x, targets)) for i in range(n)]
    weights = solve(matrix, rhs)
    provisional = Ridge(means, scales, weights, intercept)
    correction = median(y - provisional.predict_log(row_snapshot(r)) for r, y in zip(rows, targets))
    return Ridge(means, scales, weights, intercept + correction)


def token_point(log_value: float, *, outward: bool = False) -> int:
    # Numeric guard only. These endpoints are far beyond supported task budgets.
    value = math.exp(max(0.0, min(700.0, log_value)))
    return max(1, math.ceil(value) if outward else round(value))


def select_ridge(fit: list[dict], mode: str) -> tuple[Ridge | None, str]:
    if mode == 'median':
        return None, 'requested_hierarchical_median'
    supported = [r for r in fit if valid_snapshot(row_snapshot(r))]
    if mode == 'ridge':
        model = fit_ridge(supported)
        return model, 'requested_ridge' if model else 'insufficient_feature_history'
    if mode != 'auto':
        raise ValueError('point model must be auto, median, or ridge')
    validation = [r for r in supported if int(history.task_key(r['instance_id']), 16) % 5 == 0]
    training = [r for r in supported if int(history.task_key(r['instance_id']), 16) % 5 != 0]
    if len(validation) < 5 or len(training) < 20:
        return None, 'insufficient_fit_only_validation'
    candidate = fit_ridge(training)
    ridge_error = mean(abs(token_point(candidate.predict_log(row_snapshot(r))) - r['actual_tokens']) for r in validation)
    hierarchy_error = mean(abs(token_point(hierarchy(training, r['task_type'])) - r['actual_tokens']) for r in validation)
    global_error = mean(abs(median(r['actual_tokens'] for r in training) - r['actual_tokens']) for r in validation)
    # Require a fixed 5% validation improvement over BOTH simple baselines.
    if ridge_error < 0.95 * min(hierarchy_error, global_error):
        return fit_ridge(supported), 'ridge_won_fit_only_validation'
    return None, 'median_retained_on_fit_only_validation'


def predict(history_path: Path, task_type: str, profile: dict[str, str], baseline: int,
            *, task: str = '', instance_id: str | None = None,
            features: dict | None = None, point_model: str = 'auto') -> Forecast:
    fit, calibration, digest = history.eligible_records(history_path, profile, instance_id=instance_id, task=task)
    counts = dict(fit_samples=len(fit), calibration_samples=len(calibration), history_digest=digest)
    if len(fit) < 5:
        return Forecast(**counts)
    ridge, reason = select_ridge(fit, point_model) if valid_snapshot(features) else (None, 'query_features_unavailable')
    if ridge is not None:
        # Feature availability/version must also match at calibration time.
        calibration = [r for r in calibration if valid_snapshot(row_snapshot(r))]
        counts['calibration_samples'] = len(calibration)

    def point_log(kind: str, snapshot: dict | None) -> float:
        return ridge.predict_log(snapshot) if ridge is not None else hierarchy(fit, kind)

    center = point_log(task_type, features)
    point = token_point(center)
    method = 'ridge_log' if ridge is not None else 'hierarchical_median'
    empirical_high = max(point, math.ceil(history.quantile([r['actual_tokens'] for r in fit], 0.9)))
    result = dict(method=method, point_method=method, p50_tokens=point,
                  low_tokens=0, high_tokens=empirical_high, selection=reason,
                  interval_method='one_sided_uncalibrated', **counts)
    if len(calibration) < 19:
        return Forecast(**result)
    scores = [math.log(r['actual_tokens']) - point_log(r['task_type'], row_snapshot(r)) for r in calibration]
    rank = math.ceil((len(scores) + 1) * 0.9)
    if rank > len(scores):
        return Forecast(**result)
    # Single-sided quantile, not symmetric residual expansion or interval midpoint.
    adjustment = max(0.0, sorted(scores)[rank - 1])
    upper = token_point(center + adjustment, outward=True)
    result.update(calibrated=True, target_coverage=0.9, high_tokens=upper,
                  budget_upper_tokens=upper, interval_method='one_sided_conformal_90')
    return Forecast(**result)
