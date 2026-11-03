"""Feature-aware, pure-preflight costs with configuration-wide budget calibration."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median

import history_model as history
import task_neighbors

FEATURE_VERSION = 'preflight-counts-v1'
FEATURE_NAMES = ('log_task_chars', 'log_lines', 'log_code_files', 'log_test_files',
                 'repo_known', 'log_initial_context', 'log_paths', 'log_errors')


@dataclass(frozen=True)
class Forecast(history.HistoricalForecast):
    point_method: str = 'heuristic'
    interval_method: str = 'heuristic_uncalibrated'
    budget_upper_tokens: int | None = None
    selection: str = 'insufficient_history'
    selection_report: dict | None = None
    support: dict | None = None


def feature_snapshot(task: str, file_count: int, code_count: int, test_count: int,
                     repo_known: bool, initial_context: int, repository: str | None = None) -> dict:
    """Only declared inputs; file_count is accepted for future schema compatibility."""
    values = [math.log1p(len(task)), math.log1p(task.count('\n')),
              math.log1p(code_count), math.log1p(test_count), float(repo_known),
              math.log1p(initial_context),
              math.log1p(len(re.findall(r'[\w./-]+\.(?:py|js|ts|go|rs|java|cpp|c|h)\b', task))),
              math.log1p(len(re.findall(r'traceback|exception|error|failed|失败|报错', task.lower())))]
    result = {'version': FEATURE_VERSION, 'values': dict(zip(FEATURE_NAMES, values))}
    if repository:
        result['repository'] = repository.strip()
    return result


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


@dataclass
class Candidate:
    kind: str
    rows: list[dict]
    ridge: Ridge | None = None
    text: task_neighbors.TextIndex | None = None

    @classmethod
    def fit(cls, rows, kind):
        return cls(kind, rows, fit_ridge(rows) if kind == 'ridge' else None,
                   task_neighbors.TextIndex(rows) if kind == 'retrieval' else None)

    def predict_log(self, kind, snapshot, task):
        prior = hierarchy(self.rows, kind)
        support = dict(repository_fit_samples=0, neighbor_count=0)
        if self.kind == 'ridge':
            if self.ridge is not None and valid_snapshot(snapshot):
                return self.ridge.predict_log(snapshot), support
            return prior, support
        if self.kind in {'repository', 'retrieval'}:
            prior, count = task_neighbors.repository_prior(self.rows, task_neighbors.repository(snapshot), prior)
            support['repository_fit_samples'] = count
        if self.text is not None:
            prior, text_support = self.text.predict_log(task, prior)
            support.update(text_support)
        return prior, support


_POINT_CACHE = OrderedDict()


def select_candidate(fit, mode):
    if mode not in {'auto', 'median', 'ridge', 'repository', 'retrieval'}:
        raise ValueError('point model must be auto, median, ridge, repository, or retrieval')
    signature = hashlib.sha256(json.dumps([(r['instance_id'], r['task'], r['task_type'],
                                            r['actual_tokens'], row_snapshot(r)) for r in fit],
                                           sort_keys=True).encode()).hexdigest()
    key = (signature, mode)
    if key in _POINT_CACHE:
        _POINT_CACHE.move_to_end(key)
        return _POINT_CACHE[key]
    report = {'protocol': 'fit_only_fivefold_v1', 'usable_folds': 0}
    chosen, reason = ('median', 'insufficient_fit_only_validation')
    if mode != 'auto':
        chosen, reason = mode, 'requested_' + mode
    elif len(fit) >= 30:
        kinds = ['median', 'ridge', 'repository', 'retrieval']
        errors = {k: [] for k in kinds}
        fold_errors = {k: [] for k in kinds}
        for fold in range(5):
            validation = [r for r in fit if int(history.task_key(r['instance_id']), 16) % 5 == fold]
            training = [r for r in fit if int(history.task_key(r['instance_id']), 16) % 5 != fold]
            if len(validation) < 3 or len(training) < 20:
                continue
            for kind in kinds:
                model = Candidate.fit(training, kind)
                loss = [abs(token_point(model.predict_log(r['task_type'], row_snapshot(r), r['task'])[0])
                            - r['actual_tokens']) for r in validation]
                errors[kind].extend(loss)
                fold_errors[kind].append(mean(loss))
        usable = len(fold_errors['median'])
        report['usable_folds'] = usable
        if usable >= 4:
            averages = {k: mean(v) for k, v in errors.items()}
            wins = {k: sum(a < b for a, b in zip(v, fold_errors['median'])) for k, v in fold_errors.items()}
            report.update(oof_mae=averages, fold_wins_vs_median=wins)
            eligible = [k for k in kinds[1:] if averages[k] < .95 * averages['median']
                        and wins[k] >= math.ceil(.8 * usable)]
            if eligible:
                chosen = min(eligible, key=lambda k: (averages[k], k))
                reason = chosen + '_won_fivefold_validation'
            else:
                reason = 'median_retained_on_fivefold_validation'
    model = Candidate.fit(fit, chosen)
    if chosen == 'ridge' and model.ridge is None:
        model, reason = Candidate.fit(fit, 'median'), 'insufficient_feature_history'
    result = model, reason, report
    _POINT_CACHE[key] = result
    if len(_POINT_CACHE) > 8:
        _POINT_CACHE.popitem(last=False)
    return result


def predict(history_path: Path, task_type: str, profile: dict[str, str], baseline: int,
            *, task: str = '', instance_id: str | None = None,
            features: dict | None = None, point_model: str = 'auto') -> Forecast:
    fit, calibration, digest = history.eligible_records(history_path, profile, instance_id=instance_id, task=task)
    counts = dict(fit_samples=len(fit), calibration_samples=len(calibration), history_digest=digest)
    if len(fit) < 5:
        return Forecast(**counts)
    model, reason, selection_report = select_candidate(fit, point_model)
    if model.kind == 'ridge' and not valid_snapshot(features):
        model, reason = Candidate.fit(fit, 'median'), 'query_features_unavailable'
    if model.kind == 'ridge':
        # Feature availability/version must also match at calibration time.
        calibration = [r for r in calibration if valid_snapshot(row_snapshot(r))]
        counts['calibration_samples'] = len(calibration)
    if model.kind in {'repository', 'retrieval'}:
        availability = bool(task_neighbors.repository(features))
        calibration = [r for r in calibration if bool(task_neighbors.repository(row_snapshot(r))) == availability]
        counts['calibration_samples'] = len(calibration)

    def point_log(kind, snapshot, text):
        return model.predict_log(kind, snapshot, text)[0]

    center, support = model.predict_log(task_type, features, task)
    point = token_point(center)
    method = {'ridge': 'ridge_log', 'median': 'hierarchical_median',
              'repository': 'repository_shrinkage', 'retrieval': 'task_similarity_shrinkage'}[model.kind]
    empirical_high = max(point, math.ceil(history.quantile([r['actual_tokens'] for r in fit], 0.9)))
    result = dict(method=method, point_method=method, p50_tokens=point,
                  low_tokens=0, high_tokens=empirical_high, selection=reason,
                  selection_report=selection_report, support=support,
                  interval_method='one_sided_uncalibrated', **counts)
    if len(calibration) < 19:
        return Forecast(**result)
    scores = [math.log(r['actual_tokens']) - point_log(r['task_type'], row_snapshot(r), r['task']) for r in calibration]
    rank = math.ceil((len(scores) + 1) * 0.9)
    if rank > len(scores):
        return Forecast(**result)
    # Single-sided quantile, not symmetric residual expansion or interval midpoint.
    adjustment = max(0.0, sorted(scores)[rank - 1])
    upper = token_point(center + adjustment, outward=True)
    result.update(calibrated=True, target_coverage=0.9, high_tokens=upper,
                  budget_upper_tokens=upper, interval_method='one_sided_conformal_90')
    return Forecast(**result)
