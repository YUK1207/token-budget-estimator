import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import estimate as estimator


class PreflightRegressionTests(unittest.TestCase):
    def test_repository_identity_is_sanitized_and_recorded_as_preflight_metadata(self):
        self.assertEqual(estimator.normalize_repository('https://token:secret@github.com/Owner/Repo.git'), 'Owner/Repo')
        with tempfile.TemporaryDirectory() as tmp:
            result = estimator.estimate('Fix parser', tmp, 0, Path(tmp) / 'absent', repository='owner/repo')
            self.assertEqual(result.feature_snapshot['repository'], 'owner/repo')

    def test_declared_repair_kind_and_feature_record_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                estimator.main(['estimate', '--task', 'A surprising behavior', '--cwd', tmp,
                                '--task-kind', 'code_repair', '--point-model', 'median',
                                '--model', 'm', '--agent', 'a', '--execution-config', 'c', '--json'])
            result = json.loads(output.getvalue())
            self.assertEqual(result['task_type'], 'code_repair')
            self.assertIn('feature_snapshot', result)
            self.assertEqual(result['feature_snapshot']['version'], 'preflight-counts-v1')
            self.assertIsNone(result['budget_upper_tokens'])

    def test_declared_repository_counts_validate_and_replace_local_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'repo.json'
            path.write_text(json.dumps(dict(file_count=100, code_file_count=60, test_file_count=30, config_file_count=2)))
            signals = estimator.load_repo_signals(path)
            self.assertEqual(signals.repo_size, 'medium')
            self.assertEqual(signals.code_file_count, 60)
            path.write_text(json.dumps(dict(file_count=100, code_file_count=-1, test_file_count=30, config_file_count=2)))
            with self.assertRaises(ValueError):
                estimator.load_repo_signals(path)

    def test_advisory_reports_disclose_uncalibrated_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = estimator.build_preflight_controller('Fix a bug', tmp, 0, Path(tmp) / 'missing')
            self.assertIn('heuristic_uncalibrated', estimator.render_controller_markdown(result))
            self.assertIn('heuristic_uncalibrated', estimator.render_compare_markdown([result.forecast]))

    def test_calibrated_interval_does_not_raise_lower_bound_using_approximate_prompt(self):
        historical = estimator.history_model.HistoricalForecast(
            method='historical_split_conformal', p50_tokens=100, p90_tokens=150,
            low_tokens=0, high_tokens=200, calibrated=True, target_coverage=0.9)
        with tempfile.TemporaryDirectory() as tmp, patch.object(estimator.history_model, 'predict', return_value=historical):
            result = estimator.estimate('Fix a bug', tmp, 1000, Path(tmp) / 'missing')
            self.assertEqual(result.low_tokens, 0)

    def test_all_preflight_commands_forward_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            for command in ['compare', 'optimize', 'control']:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    estimator.main([command, '--task', 'Fix a bug', '--cwd', tmp,
                                    '--history', str(Path(tmp) / 'absent'), '--model', 'm',
                                    '--agent', 'a', '--execution-config', 'c',
                                    '--peak-context-tokens', '2000', '--context-window', '8000', '--json'])
                result = json.loads(out.getvalue())
                forecast = result[0] if command == 'compare' else result['forecast'] if command == 'control' else result['original_estimate']
                self.assertEqual(forecast['execution_profile']['model'], 'm')
                self.assertEqual(forecast['context_fit'], 'fits')

    def test_record_rejects_profile_mismatch_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction = estimator.estimate('Fix a bug', tmp, 0, Path(tmp) / 'absent',
                                           execution_profile={'model': 'm', 'agent': 'a', 'execution_config': 'c'})
            path = self.write_prediction(tmp, estimator.asdict(prediction))
            target = Path(tmp) / 'history.jsonl'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                estimator.main(['record', '--prediction-file', path, '--actual-tokens', '100',
                                '--instance-id', 'i1', '--model', 'wrong', '--history', str(target)])
            self.assertFalse(target.exists())

    def test_history_changes_total_without_fabricating_component_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = {'model': 'm', 'agent': 'a', 'execution_config': 'c'}
            path = Path(tmp) / 'history.jsonl'
            initial = estimator.estimate('Fix a bug', tmp, 0, path)
            rows = [dict(schema_version=2, baseline_version=initial.baseline_version,
                         instance_id=f'issue-{i}', task=f'Other bug {i}', task_type='debugging',
                         execution_profile=profile, baseline_midpoint_tokens=initial.baseline_midpoint_tokens,
                         actual_tokens=10000, split='fit', outcome='completed') for i in range(5)]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            result = estimator.estimate('Fix a bug', tmp, 0, path, execution_profile=profile,
                                        input_price_per_million=1, output_price_per_million=5)
            self.assertEqual(result.p50_tokens, 10000)
            self.assertIsNone(result.input_high_tokens)
            self.assertIsNone(result.cost)
            self.assertEqual(result.prediction_method, 'hierarchical_median')

    def test_substrings_do_not_make_tasks_broad(self):
        self.assertEqual(estimator.classify_task('Fix callback error in function'), 'debugging')
        self.assertEqual(estimator.classify_task('Fix error in small function'), 'debugging')
        self.assertEqual(estimator.classify_task('Refactor the entire project'), 'large_project_task')

    def test_total_usage_is_not_peak_context(self):
        result = estimator.estimate('Fix a bug', '.', 0, Path('missing-history'), context_window=128000)
        self.assertEqual(result.context_fit, 'unknown')
        result = estimator.estimate('Fix a bug', '.', 0, Path('missing-history'),
                                    context_window=128000, peak_context_tokens=140000)
        self.assertEqual(result.context_fit, 'likely_exceeds')

    def test_optimizer_does_not_invent_savings(self):
        result = estimator.optimize_task('Refactor the entire project', '.', 0, Path('missing-history'), mode='cheap')
        self.assertEqual(result.optimized_estimate, result.original_estimate)
        self.assertIn('not measured', result.savings_summary)

    def test_record_and_estimate_keep_configuration_and_raw_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = str(Path(tmp) / 'history.jsonl')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                estimator.main(['estimate', '--task', 'Fix a bug', '--cwd', tmp,
                                '--history', history, '--model', 'model-v1', '--agent', 'mini-v1',
                                '--execution-config', 'reasoning-medium-context-default', '--json'])
            prediction = json.loads(out.getvalue())
            self.assertEqual(prediction['prediction_method'], 'heuristic')
            self.assertEqual(prediction['execution_profile']['model'], 'model-v1')
            self.assertIsNone(prediction['p50_tokens'])
            with contextlib.redirect_stdout(io.StringIO()):
                estimator.main(['record', '--prediction-file', self.write_prediction(tmp, prediction),
                                '--actual-tokens', '20000', '--instance-id', 'repo-issue-1',
                                '--history', history])
            row = json.loads(Path(history).read_text(encoding='utf-8'))
            self.assertEqual(row['baseline_midpoint_tokens'], prediction['baseline_midpoint_tokens'])
            self.assertEqual(row['execution_profile'], prediction['execution_profile'])
            self.assertEqual(row['instance_id'], 'repo-issue-1')

    @staticmethod
    def write_prediction(tmp, prediction):
        path = Path(tmp) / 'prediction.json'
        path.write_text(json.dumps(prediction), encoding='utf-8')
        return str(path)
