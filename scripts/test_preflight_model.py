import json
import math
from pathlib import Path
import tempfile
import unittest

import history_model
import preflight_model as model

PROFILE = {'model': 'm', 'agent': 'a', 'execution_config': 'c'}


def row(i, actual=10000, kind='debugging', split='fit', features=None):
    return dict(schema_version=2, baseline_version=history_model.BASELINE_VERSION,
                instance_id=f'issue-{i}', task=f'Fix issue {i}', task_type=kind,
                execution_profile=PROFILE, baseline_midpoint_tokens=10000,
                actual_tokens=actual, split=split, outcome='completed',
                feature_snapshot=features)


class AdaptiveTests(unittest.TestCase):
    def test_repository_calibration_matches_availability_in_both_directions(self):
        rows = [row(i, features={'repository': 'a/repo'}) for i in range(10)]
        rows += [row(100+i, 10000, split='calibration', features={'repository': 'a/repo'}) for i in range(20)]
        rows += [row(200+i, 100000, split='calibration') for i in range(19)]
        known = self.forecast(rows, features={'repository': 'a/repo'}, point_model='repository')
        missing = self.forecast(rows, point_model='repository')
        self.assertEqual(known.calibration_samples, 20)
        self.assertEqual(missing.calibration_samples, 19)
        self.assertTrue(known.calibrated)
        self.assertTrue(missing.calibrated)

    def test_repository_candidate_is_selected_only_with_oof_evidence(self):
        rows = []
        for i in range(100):
            repo = 'a/repo' if i < 50 else 'b/repo'
            features = model.feature_snapshot('Generic task', 0, 0, 0, False, 0, repository=repo)
            rows.append(row(i, 10000 if i < 50 else 1000000, features=features))
        result = self.forecast(rows, features=model.feature_snapshot('Generic task', 0, 0, 0, False, 0, repository='a/repo'))
        self.assertIn(result.point_method, {'repository_shrinkage', 'task_similarity_shrinkage'})
        self.assertGreaterEqual(result.selection_report['usable_folds'], 4)
        self.assertLess(result.p50_tokens, 100000)

    def test_repository_and_text_candidates_use_only_fit_support(self):
        rows = [row(i, 20000, features={'repository': 'a/repo'}) for i in range(10)]
        rows += [row(30+i, 100000, features={'repository': 'b/repo'}) for i in range(10)]
        result = self.forecast(rows, features={'repository': 'a/repo'}, point_model='repository')
        self.assertEqual(result.point_method, 'repository_shrinkage')
        self.assertEqual(result.support['repository_fit_samples'], 10)
        self.assertLess(result.p50_tokens, 60000)
        absent = self.forecast(rows, features={'repository': 'unknown/repo'}, point_model='repository')
        self.assertEqual(absent.support['repository_fit_samples'], 0)
        self.assertEqual(absent.p50_tokens, 60000)

    def test_history_digest_covers_task_text_for_semantic_models(self):
        rows = [row(i) for i in range(8)]
        before = self.forecast(rows, point_model='retrieval')
        rows[0]['task'] = 'A completely different task description'
        after = self.forecast(rows, point_model='retrieval')
        self.assertNotEqual(before.history_digest, after.history_digest)

    def forecast(self, rows, **options):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history'
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            return model.predict(path, 'test_repair', PROFILE, 10000, **options)

    def test_sparse_types_pool_same_configuration(self):
        rows = [row(i, 20000) for i in range(10)] + [row(11, 100000, 'test_repair')]
        result = self.forecast(rows, point_model='median')
        self.assertEqual(result.method, 'hierarchical_median')
        self.assertGreater(result.p50_tokens, 20000)
        self.assertLess(result.p50_tokens, 100000)
        self.assertEqual(result.fit_samples, 11)

    def test_one_sided_log_calibration_has_no_symmetric_expansion(self):
        rows = [row(i, 10000) for i in range(10)]
        rows += [row(100+i, 30000, split='calibration') for i in range(19)]
        result = self.forecast(rows, point_model='median')
        self.assertEqual(result.low_tokens, 0)
        self.assertAlmostEqual(result.budget_upper_tokens, 30000, delta=1)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.interval_method, 'one_sided_conformal_90')

    def test_current_and_test_exclusion_applies_across_types(self):
        rows = [row(i) for i in range(8)]
        rows += [row(0, 999999, 'test_repair', split='test')]
        result = self.forecast(rows, instance_id='issue-1')
        self.assertEqual(result.fit_samples, 6)

    def test_features_do_not_include_future_usage(self):
        snapshot = model.feature_snapshot('Fix a.py traceback error', 100, 5, 3, True, 1000)
        self.assertEqual(set(snapshot['values']), set(model.FEATURE_NAMES))
        self.assertTrue(model.valid_snapshot(snapshot))
        self.assertFalse(model.valid_snapshot({'version': 'future', 'values': snapshot['values']}))
        self.assertIsNone(model.feature_vector({'version': model.FEATURE_VERSION, 'values': {'actual_tokens': 1}}))

    def test_ridge_learns_feature_signal_with_fit_only_selection(self):
        rows = []
        for i in range(100):
            snapshot = model.feature_snapshot('x' * (100 + i * 40), 0, 0, 0, False, 0)
            rows.append(row(i, 10 * (100 + i * 40), features=snapshot))
        query = model.feature_snapshot('x' * 2100, 0, 0, 0, False, 0)
        result = self.forecast(rows, features=query)
        self.assertEqual(result.point_method, 'ridge_log')
        self.assertLess(abs(result.p50_tokens - 21000), 4000)
        changed = rows + [row(200+i, 9999999, split='calibration', features=query) for i in range(20)]
        again = self.forecast(changed, features=query)
        self.assertEqual(again.p50_tokens, result.p50_tokens)
        self.assertEqual(again.point_method, result.point_method)
        self.assertEqual(again.selection, result.selection)

    def test_ridge_rejects_missing_and_nonfinite_feature_values(self):
        snapshot = model.feature_snapshot('task', 0, 0, 0, False, 0)
        snapshot['values'][model.FEATURE_NAMES[0]] = float('nan')
        self.assertFalse(model.valid_snapshot(snapshot))
        self.assertIsNone(model.fit_ridge([row(i) for i in range(30)]))

    def test_uncalibrated_upper_is_not_claimed_as_budget_bound(self):
        result = self.forecast([row(i) for i in range(5)])
        self.assertIsNone(result.budget_upper_tokens)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.interval_method, 'one_sided_uncalibrated')
