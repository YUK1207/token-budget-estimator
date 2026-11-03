"""Pure-preflight empirical ratio quantiles with split-conformal calibration.

No solver execution, model calls, or learned CQR implementation. The fixed
heuristic is a scale feature; configuration-matched history learns its ratios.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

BASELINE_VERSION = 'heuristic-v2'
PROFILE_KEYS = ('model', 'agent', 'execution_config')


@dataclass(frozen=True)
class HistoricalForecast:
    method: str = 'heuristic'
    p50_tokens: int | None = None
    p90_tokens: int | None = None
    low_tokens: int | None = None
    high_tokens: int | None = None
    fit_samples: int = 0
    calibration_samples: int = 0
    calibrated: bool = False
    target_coverage: float | None = None
    history_digest: str | None = None


def positive_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def quantile(values: list[float], probability: float) -> float:
    """Nearest-rank empirical quantile; the median is computed separately."""
    if not values or not 0 < probability <= 1:
        raise ValueError('quantile requires data and probability in (0, 1]')
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)]


def conformal_radius(scores: list[float], coverage: float) -> float | None:
    if not 0 < coverage < 1:
        raise ValueError('coverage must be in (0, 1)')
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        return None
    return max(0.0, sorted(scores)[rank - 1])


def task_key(task: str) -> str:
    return hashlib.sha256(' '.join(task.split()).encode('utf-8')).hexdigest()


def predict(history_path: Path, task_type: str, profile: dict[str, str],
            baseline: int, *, instance_id: str | None = None,
            task: str = '') -> HistoricalForecast:
    """Forecast only from prior v2 records; test/current instances never fit.

    Keep the first eligible run per linked ID/text group. Auto split uses a
    stable group-ID hash, so repeats cannot cross fit and calibration. Test
    groups are excluded across every row. The forecast target is one naturally
    terminated run, including solver failures, not runs stopped by a token cap.
    """
    if not all(isinstance(profile.get(k), str) and profile[k].strip() for k in PROFILE_KEYS):
        return HistoricalForecast()
    if not history_path.exists():
        return HistoricalForecast()
    rows = []
    for line in history_path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if (row.get('schema_version') == 2
                and row.get('baseline_version') == BASELINE_VERSION
                and row.get('task_type') == task_type
                and row.get('execution_profile') == profile
                and isinstance(row.get('instance_id'), str) and row['instance_id'].strip()
                and isinstance(row.get('task'), str) and row['task'].strip()
                and isinstance(row.get('split'), str)
                and row.get('split') in {'auto', 'fit', 'calibration', 'test'}):
            rows.append(row)
    # Connect ID aliases and exact normalized text transitively. Grouping
    # precedes exclusions so a test alias also excludes its paraphrased rows.
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for row in rows:
        left = find('id:' + row['instance_id'])
        right = find('text:' + task_key(row['task']))
        parents[max(left, right)] = min(left, right)
    excluded_groups = {find('id:' + r['instance_id']) for r in rows if r['split'] == 'test'}
    if instance_id is not None and 'id:' + instance_id in parents:
        excluded_groups.add(find('id:' + instance_id))
    if task and 'text:' + task_key(task) in parents:
        excluded_groups.add(find('text:' + task_key(task)))
    fit, calibration = [], []
    seen = set()
    used = []
    for row in rows:
        identity = find('id:' + row['instance_id'])
        if identity in seen or identity in excluded_groups:
            continue
        if not isinstance(row.get('outcome'), str) or row['outcome'] not in {'completed', 'failed'}:
            continue
        actual, raw = row.get('actual_tokens'), row.get('baseline_midpoint_tokens')
        if not positive_number(actual) or not positive_number(raw):
            continue
        ratio = actual / raw
        if not math.isfinite(ratio) or ratio <= 0:
            continue
        seen.add(identity)
        split = row['split']
        if split == 'auto':
            split = 'fit' if int(task_key(identity), 16) % 2 == 0 else 'calibration'
        (fit if split == 'fit' else calibration).append(ratio)
        used.append((identity, split, actual, raw))
    digest = hashlib.sha256(json.dumps(sorted(used)).encode('utf-8')).hexdigest()
    counts = dict(fit_samples=len(fit), calibration_samples=len(calibration), history_digest=digest)
    if len(fit) < 5:
        return HistoricalForecast(**counts)
    p50 = math.ceil(median(fit) * baseline)
    p90 = math.ceil(quantile(fit, 0.9) * baseline)
    # No statistical confidence label follows merely from a sample count.
    result = dict(method='historical_quantiles', p50_tokens=p50, p90_tokens=p90, **counts)
    if len(fit) < 10 or len(calibration) < 19:
        return HistoricalForecast(**result)
    lower, upper = quantile(fit, 0.05), quantile(fit, 0.95)
    scores = [max(lower - r, r - upper) for r in calibration]
    radius = conformal_radius(scores, 0.9)
    if radius is None:
        return HistoricalForecast(**result)
    result.update(method='historical_split_conformal', calibrated=True,
                  target_coverage=0.9, low_tokens=math.floor(max(0, lower - radius) * baseline),
                  high_tokens=math.ceil((upper + radius) * baseline))
    return HistoricalForecast(**result)
