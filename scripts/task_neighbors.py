"""Fit-only lexical evidence and repository priors, without model calls."""
from collections import Counter
import math
import re
from statistics import median

STOP = frozenset('the and for with this that from have has was were are not but can will would should please fix issue code following'.split())


def tokens(text):
    return [v for v in re.findall(r'[a-z_][a-z_0-9]{2,}|[\u4e00-\u9fff]{2,}', text.lower()) if v not in STOP]


def snapshot(row):
    value = row.get('feature_snapshot')
    if value is None and isinstance(row.get('prediction'), dict):
        value = row['prediction'].get('feature_snapshot')
    return value if isinstance(value, dict) else {}


def repository(value):
    key = value.get('repository') if isinstance(value, dict) else None
    return key.strip() if isinstance(key, str) else ''


def repository_prior(rows, key, prior):
    local = [r['actual_tokens'] for r in rows if key and repository(snapshot(r)) == key]
    if not local:
        return prior, 0
    weight = len(local) / (len(local) + 10.0)
    return weight * math.log(median(local)) + (1 - weight) * prior, len(local)


def weighted_median(pairs):
    if not pairs or not all(math.isfinite(v) and math.isfinite(w) and w > 0 for v, w in pairs):
        raise ValueError('weighted median needs finite values and positive weights')
    threshold = sum(w for _, w in pairs) / 2
    accumulated = 0.0
    for value, weight in sorted(pairs):
        accumulated += weight
        if accumulated >= threshold:
            return value
    return max(v for v, _ in pairs)


class TextIndex:
    def __init__(self, rows):
        self.rows = tuple(rows)
        counts = [Counter(tokens(r['task'])) for r in rows]
        document_frequency = Counter(t for row in counts for t in row)
        self.idf = {t: math.log((1 + len(rows)) / (1 + n)) + 1 for t, n in document_frequency.items()}
        self.vectors = tuple(self._encode(row) for row in counts)

    def _encode(self, counts):
        values = {t: (1 + math.log(n)) * self.idf[t] for t, n in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in values.values()))
        return {t: v / norm for t, v in values.items()} if norm else {}

    def predict_log(self, task, prior):
        query = self._encode(Counter(tokens(task)))
        matches = []
        if query:
            for row, vector in zip(self.rows, self.vectors):
                similarity = sum(value * vector.get(t, 0) for t, value in query.items())
                if similarity > 0:
                    matches.append((similarity, row['instance_id'], row['actual_tokens']))
        matches.sort(key=lambda r: (-r[0], r[1]))
        matches = matches[:5]
        if not matches:
            return prior, dict(neighbor_count=0, max_similarity=0.0)
        total = sum(s * s for s, _, _ in matches)
        weight = min(0.5, total / (1 + total))
        local = weighted_median([(math.log(cost), s * s) for s, _, cost in matches])
        return ((1 - weight) * prior + weight * local,
                dict(neighbor_count=len(matches), max_similarity=matches[0][0]))
