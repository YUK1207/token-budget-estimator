"""Retrospective fixed-benchmark comparison with separately installed upstream code.

No TokenCast implementation is vendored. Its public task_launch tree runtime is
adapted to common numeric features and total input+output labels; this is not a
reproduction of the full paper's compositional forecasting architecture.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

FEATURE_NAMES = ('log_task_chars', 'log_lines', 'log_code_files', 'log_test_files',
                 'repo_known', 'log_initial_context', 'log_paths', 'log_errors')
UPSTREAM_COMMIT = 'af947602da74b9cb6ed127850f531db6187278fa'
SEED = 20260719
PARAMETERS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=10,
                  max_depth=6, lambda_l2=1.0, feature_fraction=1.0,
                  bagging_fraction=1.0, bagging_freq=0, num_threads=1,
                  device_type='cpu', deterministic=True, force_col_wise=True,
                  verbosity=-1, seed=SEED)


def vector(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get('version') != 'preflight-counts-v1':
        raise ValueError('unsupported preflight feature version')
    values = snapshot.get('values', {})
    result = [values.get(k) for k in FEATURE_NAMES]
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
               and math.isfinite(v) and v >= 0 for v in result):
        raise ValueError('features must be finite known-start counts')
    return [float(v) for v in result]


def validate_partitions(fit, calibration, test):
    groups = [set(fit), set(calibration), set(test)]
    if any(len(g) != len(ids) for g, ids in zip(groups, [fit, calibration, test])):
        raise ValueError('duplicate task IDs')
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('task leakage across partitions')


def official_features(features):
    # Upstream vector columns deliberately require immutable tuples.
    return {'task_embedding': tuple(features)}


def upper_bounds(calibration_points, calibration_actual, query_points):
    if len(calibration_points) != len(calibration_actual) or len(calibration_actual) < 19:
        raise ValueError('need at least nineteen paired calibration tasks')
    numbers = [*calibration_points, *calibration_actual, *query_points]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and v > 0 for v in numbers):
        raise ValueError('costs and points must be positive and finite')
    scores = sorted(math.log(y) - math.log(p) for p, y in zip(calibration_points, calibration_actual))
    rank = math.ceil((len(scores) + 1) * 0.9)
    adjustment = max(0.0, scores[rank - 1])
    return [math.ceil(math.exp(math.log(p) + adjustment)) for p in query_points]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def official_model(fit_rows, upstream, transform, model_dir):
    """Use upstream fit/start/predict APIs, plus an explicit all-fit outer refit."""
    sys.path.insert(0, str(upstream / 'src'))
    import lightgbm as lgb
    import numpy as np
    from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
    from token_prediction.estimators.base import FitContext, TrainingExample, TrainingView, RunContext
    from token_prediction.estimators.lightgbm import LightGBMQuantileEstimator

    target = PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS
    position = PredictionPosition.TASK_LAUNCH
    dataset, contract, condition, unit = 'swe-fixed-v3', 'preflight-counts-v1', 'claude37-openhands-run4', 'provider-input-plus-output-v1'

    def point(identity, features):
        return PredictionPoint(point_id='launch:' + identity, source_event_id='start:' + identity,
                               task_id=identity, trajectory_id='trace:' + identity, run_id='run4',
                               prediction_context_id='context:' + identity, condition_id=condition,
                               logical_call_id=None, attempt_id=None, cutoff_event_seq=0,
                               position=position, target=target,
                               features=official_features(features), known_offset_tokens=0,
                               token_unit_contract_id=unit)

    def view(rows):
        return TrainingView(dataset_id=dataset, position=position, target=target,
                            input_contract_id=contract, token_unit_contract_id=unit,
                            examples=tuple(TrainingExample(point(r['instance_id'], vector(r['feature_snapshot'])),
                                                           r['actual_tokens'], 1.0) for r in rows))

    inner = [r for r in fit_rows if int(hashlib.sha256(' '.join(r['instance_id'].split()).encode()).hexdigest(), 16) % 5 != 0]
    validation = [r for r in fit_rows if r not in inner]
    if len(inner) < 20 or len(validation) < 5:
        raise ValueError('insufficient fit-only train/validation tasks')
    estimator = LightGBMQuantileEstimator(num_boost_round=300, early_stopping_rounds=30,
                                          learning_rate=.03, num_leaves=15, min_data_in_leaf=10,
                                          max_depth=6, lambda_l2=1., target_transform=transform)
    fitted = estimator.fit(view(inner), view(validation), FitContext(seed=SEED, fold=0))
    # Stopping selection never sees the calibration or test labels. Refit every
    # head on all 71 fitting records at the selected count, with the fitted
    # upstream encoder, so each method has the same labeled fitting budget.
    full = view(fit_rows)
    encoded = fitted.encoder.transform(tuple(e.point for e in full.examples))
    labels = np.asarray([r['actual_tokens'] for r in fit_rows], dtype=float)
    if transform == 'log1p':
        labels = np.log1p(labels)
    refit = {}
    for quantile, old_booster in fitted.boosters.items():
        refit[quantile] = lgb.train(dict(old_booster.params),
                                   lgb.Dataset(encoded.matrix, label=labels,
                                               feature_name=list(encoded.feature_names),
                                               categorical_feature=list(encoded.categorical_indices)),
                                   num_boost_round=fitted.best_iterations[quantile])
    fitted = replace(fitted, boosters=refit)
    model_dir.mkdir(exist_ok=True)
    for key, text in fitted.model_strings().items():
        (model_dir / (key + '.txt')).write_text(text, encoding='utf-8')
    metadata = dict(upstream_commit=UPSTREAM_COMMIT, adapter='numeric_G0_vector_task_launch_total',
                    target_transform=transform, full_fit_count=len(fit_rows),
                    inner_train_ids=[r['instance_id'] for r in inner],
                    validation_ids=[r['instance_id'] for r in validation],
                    best_iterations={str(k): v for k, v in fitted.best_iterations.items()},
                    parameters={str(q.quantile): dict(q.parameters) for q in fitted.fit_report.quantiles},
                    native_interval='uncalibrated q05/q50/q95; common one-sided calibration is applied separately',
                    refit='all fitting rows, fixed validation-selected rounds; upstream encoder and runtime retained')
    dump(model_dir / 'adapter.json', metadata)

    def predict(identity, features):
        context = RunContext(task_id=identity, trajectory_id='trace:' + identity, run_id='run4',
                             dataset_id=dataset, condition_id=condition, target=target,
                             input_contract_id=contract, target_unit_contract_id=unit)
        forecast = fitted.start(context).predict(point(identity, features))
        return max(1., forecast.point), dict(q05=forecast.raw_lower, q50=forecast.raw_point, q95=forecast.raw_upper)

    return predict, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--estimator-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    import lightgbm as lgb
    import numpy as np
    import scipy
    sys.path.insert(0, str(args.estimator_root / 'scripts'))
    import preflight_model

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    models = output / 'models'
    models.mkdir(exist_ok=True)
    reference = args.reference_dir
    history = [json.loads(line) for line in (reference / 'history-adaptive.jsonl').read_text().splitlines()]
    fit = [r for r in history if r['split'] == 'fit']
    calibration = [r for r in history if r['split'] == 'calibration']
    query = [json.loads(line) for line in (reference / 'predictions.jsonl').read_text().splitlines()]
    ids = [r['instance_id'] for r in query]
    validate_partitions([r['instance_id'] for r in fit], [r['instance_id'] for r in calibration], ids)
    if (len(fit), len(calibration), len(query)) != (71, 43, 158):
        raise ValueError('unexpected reference protocol')
    train_x = np.asarray([vector(r['feature_snapshot']) for r in fit])
    cal_x = [vector(r['feature_snapshot']) for r in calibration]
    test_x = [vector(r['arms']['selected_model']['feature_snapshot']) for r in query]
    y = np.asarray([r['actual_tokens'] for r in fit], dtype=float)
    # Protocol and all hyperparameters fixed before model fitting and scoring.
    protocol = dict(status='retrospective_fixed_benchmark_not_new_holdout',
                    counts=dict(fit=71, calibration=43, test=158), seed=SEED,
                    bootstrap_seed=20261103, bootstrap_iterations=5000,
                    features=FEATURE_NAMES, target='provider input + output total per resolved run',
                    lightgbm_version=lgb.__version__, numpy_version=np.__version__, scipy_version=scipy.__version__,
                    upstream_commit=UPSTREAM_COMMIT, upstream_source='https://github.com/DEFENSE-SEU/TokenCast',
                    plain_tree=dict(PARAMETERS, objective='quantile', alpha=.5, rounds=300, transform='log1p'),
                    official_scope='public task_launch LightGBM same-feature/target adapter; NOT full paper TokenCast',
                    file_hashes={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), reference / 'history-adaptive.jsonl', reference / 'predictions.jsonl',
                                  args.upstream / 'src/token_prediction/estimators/lightgbm.py',
                                  args.upstream / 'src/token_prediction/estimators/tabular_encoder.py',
                                  args.estimator_root / 'scripts/preflight_model.py',
                                  args.data_dir / 'summaries.json', reference / 'partitions.json']})
    dump(output / 'frozen-protocol.json', protocol)
    plain = lgb.train(dict(PARAMETERS, objective='quantile', metric='quantile', alpha=.5),
                      lgb.Dataset(train_x, label=np.log1p(y), feature_name=list(FEATURE_NAMES)), num_boost_round=300)
    plain.save_model(str(models / 'plain-lightgbm-log-median.txt'))
    official = {}
    for transform in ['identity', 'log1p']:
        name = 'tokencast_public_task_launch_' + transform + '_adapter'
        official[name], _ = official_model(fit, args.upstream, transform, models / name)
    center = float(median(y))
    if not all(r['task_type'] == 'code_repair' for r in fit):
        raise ValueError('reference median calibration assumes the declared code-repair cohort')
    if not all(r['arms']['selected_model']['point_method'] == 'hierarchical_median'
               and abs(r['arms']['selected_model']['p50_tokens'] - center) <= 1 for r in query):
        raise ValueError('current selector reference is not the documented constant median')
    forecasts = {}
    forecasts['history_median'] = ([center] * len(calibration), [center] * len(query))
    forecasts['current_selected'] = ([center] * len(calibration), [r['arms']['selected_model']['p50_tokens'] for r in query])
    ridge = preflight_model.fit_ridge(fit)
    if ridge is None:
        raise ValueError('reference ridge model cannot be reconstructed')
    ridge_test = [preflight_model.token_point(ridge.predict_log(r['arms']['ridge_candidate']['feature_snapshot'])) for r in query]
    if not all(abs(value - r['arms']['ridge_candidate']['p50_tokens']) <= 1 for value, r in zip(ridge_test, query)):
        raise ValueError('reconstructed ridge does not match the frozen reference')
    dump(models / 'fixed-reference-ridge.json', asdict(ridge))
    forecasts['fixed_reference_ridge'] = (
        [preflight_model.token_point(ridge.predict_log(r['feature_snapshot'])) for r in calibration],
        [r['arms']['ridge_candidate']['p50_tokens'] for r in query])
    forecasts['plain_lightgbm_log_median'] = (
        np.maximum(1, np.expm1(plain.predict(np.asarray(cal_x), num_threads=1))).tolist(),
        np.maximum(1, np.expm1(plain.predict(np.asarray(test_x), num_threads=1))).tolist())
    native_quantiles = {}
    for name, predictor in official.items():
        cal = [predictor(r['instance_id'], x)[0] for r, x in zip(calibration, cal_x)]
        rows = [predictor(identity, x) for identity, x in zip(ids, test_x)]
        forecasts[name] = (cal, [r[0] for r in rows])
        native_quantiles[name] = [r[1] for r in rows]
    predictions = []
    for name, (cal, points) in forecasts.items():
        uppers = upper_bounds(cal, [r['actual_tokens'] for r in calibration], points)
        for index, identity in enumerate(ids):
            predictions.append(dict(instance_id=identity, method=name, point=points[index], upper=uppers[index],
                                    native_quantiles=native_quantiles[name][index] if name in native_quantiles else None))
    (output / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in predictions) + '\n', encoding='utf-8')
    # Test actual consumption is joined only after all predictions are frozen.
    summaries = json.loads((args.data_dir / 'summaries.json').read_text())
    truth = {r['instance_id']: r['actual_tokens'] for r in summaries}
    metrics = {}
    baseline_errors = np.asarray([abs(center - truth[identity]) for identity in ids])
    bootstrap = np.random.default_rng(20261103).integers(0, len(ids), size=(5000, len(ids)))
    for name in forecasts:
        rows = [r for r in predictions if r['method'] == name]
        actual = np.asarray([truth[r['instance_id']] for r in rows])
        points = np.asarray([r['point'] for r in rows])
        uppers = np.asarray([r['upper'] for r in rows])
        errors = np.abs(points - actual)
        differences = (errors - baseline_errors)[bootstrap].mean(axis=1)
        metrics[name] = dict(n=len(rows), mae=float(errors.mean()), wape=float(errors.sum() / actual.sum()),
                             signed_bias=float((points - actual).mean()),
                             upper_coverage=float((actual <= uppers).mean()), mean_upper=float(uppers.mean()),
                             mean_positive_budget_excess=float(np.maximum(uppers - actual, 0).mean()),
                             mae_difference_vs_history_median_bootstrap95=np.quantile(differences, [.025, .975]).tolist())
    dump(output / 'metrics.json', dict(protocol_status=protocol['status'], metrics=metrics,
                                      caution='Same 158 previously seen tasks; single-source success subset; bootstrap is conditional on fitted models.'))
    lines = ['# LightGBM 与官方 Task Launch 适配对照', '',
             '这是前一轮 158 题上的固定协议回顾性对照，不是新的独立测试。71 题拟合、43 题校准；统一八项执行前特征和输入＋输出总量目标。', '',
             '| 方法 | MAE（token） | WAPE | 单侧上界覆盖率 | 平均预算上界 |', '|---|---:|---:|---:|---:|']
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['mae']:,.0f} | {m['wape']:.1%} | {m['upper_coverage']:.1%} | {m['mean_upper']:,.0f} |")
    lines += ['', '官方适配对照调用单独保存的 TokenCast 公开 LightGBMQuantileEstimator（TASK_LAUNCH），使用未修改的上游源代码和显式全拟合集重训适配。它不是完整论文组合预测管线的复现，也不使用论文原生特征或预训练权重。', '',
              'identity 和 log1p 为事先固定的目标变换对照。所有方法使用同一组 43 题做单侧 log 残差预算校准；官方原生 q05/q95 输出另存审计，不当作同一种预算上界。', '',
              '公开官方分支与普通 LightGBM 属于同一树模型家族，结果不构成对独立架构的重复胜出。样本为单一模型/执行端的成功轨迹，不能推断失败任务、跨模型泛化或节省效果。', '',
              '原始分数、配对 bootstrap 区间、冻结协议和模型文件见 metrics.json、frozen-protocol.json、predictions.jsonl、models/。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
