"""Fixed retrospective token forecasting, budget and history-size diagnostics.

No solver calls, no model tuning, no cross-repository split. The previously
inspected 158 cases are reused explicitly as retrospective diagnostics.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import sys


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def percentile(values, probability):
    return sorted(values)[min(len(values) - 1, max(0, math.ceil(len(values) * probability) - 1))]


def calibrated_upper(cal_points, cal_actual, points, coverage):
    if len(cal_points) != len(cal_actual) or not 0 < coverage < 1:
        raise ValueError('invalid calibration pairs or coverage')
    rank = math.ceil((len(cal_actual) + 1) * coverage)
    if rank > len(cal_actual) or not cal_actual:
        return None
    if any(not math.isfinite(x) or x <= 0 for x in [*cal_points, *cal_actual, *points]):
        raise ValueError('positive finite token counts required')
    scores = sorted(math.log(y) - math.log(p) for p, y in zip(cal_points, cal_actual))
    ratio = math.exp(max(0, scores[rank - 1]))
    # Subtract a small tolerance before outward rounding to avoid a one-token
    # floating-point artifact at an exactly integral reconstructed cost.
    return [math.ceil(p * ratio - 1e-8) for p in points]


def point_metrics(points, actual):
    if len(points) != len(actual) or not points:
        raise ValueError('paired nonempty predictions required')
    errors = [p - y for p, y in zip(points, actual)]
    return dict(n=len(points), mae=mean(abs(e) for e in errors),
                wape=math.fsum(abs(e) for e in errors) / math.fsum(actual), signed_bias=mean(errors),
                underprediction_rate=mean(p < y for p, y in zip(points, actual)),
                median_absolute_percentage_error=median(abs(e) / y for e, y in zip(errors, actual)),
                within_factor_two=mean(.5 * y <= p <= 2 * y for p, y in zip(points, actual)))


def wilson(hits, total):
    z = 1.959963984540054
    p = hits / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - radius, center + radius]


def budget_metrics(uppers, actual):
    hits = sum(y <= u for y, u in zip(actual, uppers))
    return dict(coverage=hits / len(actual), coverage_wilson95=wilson(hits, len(actual)),
                mean_upper=mean(uppers), reserved_to_actual=sum(uppers) / sum(actual),
                mean_unused_reservation=mean(max(0, u - y) for u, y in zip(uppers, actual)),
                mean_overrun=mean(max(0, y - u) for u, y in zip(uppers, actual)),
                overrun_frequency=1 - hits / len(actual))


def paired_mae_ci(points, reference, actual, rounds=5000):
    differences = [abs(p - y) - abs(r - y) for p, r, y in zip(points, reference, actual)]
    rng = random.Random(20261103)
    draws = [mean(rng.choices(differences, k=len(differences))) for _ in range(rounds)]
    return dict(delta_mae=mean(differences), paired_task_bootstrap95=[percentile(draws, .025), percentile(draws, .975)],
                interpretation='negative favors comparator; conditional on fitted models; no clustering/retraining uncertainty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--sweep-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    import preflight_model as own
    import history_model
    from external_baselines import validate_partitions
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    lines = lambda path: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    history = lines(args.reference_dir / 'history-adaptive.jsonl')
    tasks = load(args.data_dir / 'tasks.json')
    queries = lines(args.reference_dir / 'predictions.jsonl')
    for row in history:
        row['feature_snapshot'] = dict(row['feature_snapshot'], repository=tasks[row['instance_id']]['repo'])
    fit = [r for r in history if r['split'] == 'fit']
    cal = [r for r in history if r['split'] == 'calibration']
    test = [dict(instance_id=r['instance_id'], task=tasks[r['instance_id']]['problem_statement'], task_type='code_repair',
                 feature_snapshot=dict(r['arms']['selected_model']['feature_snapshot'], repository=tasks[r['instance_id']]['repo'])) for r in queries]
    validate_partitions([r['instance_id'] for r in fit], [r['instance_id'] for r in cal], [r['instance_id'] for r in test])
    normalized_texts = [{history_model.task_key(r['task']) for r in rows} for rows in [fit, cal, test]]
    assert not (normalized_texts[0] & normalized_texts[1] or normalized_texts[0] & normalized_texts[2] or normalized_texts[1] & normalized_texts[2])
    assert (len(fit), len(cal), len(test)) == (71, 43, 158)
    protocol = dict(status='retrospective_seen_tasks_not_independent_validation', seed=20261103,
                    fit=71, calibration=43, test=158, targets=[.8, .9, .95],
                    fit_sizes=[5, 10, 20, 50, 71], repeats=20, bootstrap_rounds=5000,
                    product_revision='785c582', product_changes=False,
                    learning_curve='nested fit subsets per seed; fixed 43 calibration and fixed 158 test; total history is fit size + 43',
                    unavailable=['fresh independent tasks', 'configuration matching ablation: only one profile',
                                 'natural solver failures: termination unknown', 'real monetary savings or new solver success'],
                    hashes={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__), root / 'scripts/preflight_model.py', root / 'scripts/task_neighbors.py']},
                    data_hashes={label: hashlib.sha256(p.read_bytes()).hexdigest() for label, p in
                                 [('reference/history-adaptive.jsonl', args.reference_dir / 'history-adaptive.jsonl'),
                                  ('reference/predictions.jsonl', args.reference_dir / 'predictions.jsonl'),
                                  ('data/tasks.json', args.data_dir / 'tasks.json'),
                                  ('data/summaries.json', args.data_dir / 'summaries.json'),
                                  ('sweep/predictions.jsonl', args.sweep_dir / 'predictions.jsonl')]})
    write(out / 'protocol.json', protocol)
    def predict(model, rows):
        return [own.token_point(model.predict_log(r['task_type'], r['feature_snapshot'], r['task'])[0]) for r in rows]
    forecasts = {}
    selection = {}
    for mode in ['auto', 'median', 'ridge', 'repository', 'retrieval']:
        model, reason, report = own.select_candidate(fit, mode)
        forecasts[mode] = dict(cal=predict(model, cal), test=predict(model, test))
        selection[mode] = dict(kind=model.kind, reason=reason, report=report)
    frozen_curve = []
    for seed in range(20):
        shuffled = list(fit)
        random.Random(20261103 + seed).shuffle(shuffled)
        for size in protocol['fit_sizes']:
            subset = sorted(shuffled[:size], key=lambda r: r['instance_id'])
            for mode in ['auto', 'median']:
                model, reason, _ = own.select_candidate(subset, mode)
                cal_points, test_points = predict(model, cal), predict(model, test)
                frozen_curve.append(dict(seed=seed, fit_size=size, calibration_size=len(cal), total_history=size+len(cal),
                                         mode=mode, selected=model.kind, reason=reason,
                                         fit_ids=[r['instance_id'] for r in subset], point=test_points,
                                         upper90=calibrated_upper(cal_points, [r['actual_tokens'] for r in cal], test_points, .9)))
        print('History-size repeat complete', seed + 1, '/ 20', flush=True)
    # Freeze all forecasts before joining retrospective test labels.
    write(out / 'frozen-predictions.json', dict(test_ids=[r['instance_id'] for r in test], models=forecasts, learning=frozen_curve, selection=selection))
    truth = {r['instance_id']: r['actual_tokens'] for r in load(args.data_dir / 'summaries.json')}
    actual = [truth[r['instance_id']] for r in test]
    cal_actual = [r['actual_tokens'] for r in cal]
    comparisons = {mode: values['test'] for mode, values in forecasts.items()}
    sweep = lines(args.sweep_dir / 'predictions.jsonl')
    for method in ['elastic_net_numeric', 'random_forest_numeric', 'lightgbm_numeric', 'catboost_numeric', 'tfidf_ridge']:
        rows = {r['instance_id']: r for r in sweep if r['method'] == method}
        comparisons[method] = [rows[r['instance_id']]['point'] for r in test]
    metrics = {name: dict(**point_metrics(points, actual), comparison_to_auto=paired_mae_ci(points, comparisons['auto'], actual))
               for name, points in comparisons.items()}
    budgets = {}
    for mode, values in forecasts.items():
        budgets[mode] = {str(target): budget_metrics(calibrated_upper(values['cal'], cal_actual, values['test'], target), actual)
                         for target in protocol['targets']}
    # No-calibration ablation uses fit-only empirical P90, matching the
    # product's uncalibrated high scenario (not a nominal coverage guarantee).
    fit_high = max(percentile([r['actual_tokens'] for r in fit], .9), max(forecasts['auto']['test']))
    budgets['no_calibration_fit_p90'] = budget_metrics([fit_high] * len(test), actual)
    constant_calibration_upper = sorted(cal_actual)[math.ceil((len(cal_actual) + 1) * .9) - 1]
    budgets['constant_calibration_quantile90'] = budget_metrics([constant_calibration_upper] * len(test), actual)
    assert calibrated_upper(forecasts['median']['cal'], cal_actual, forecasts['median']['test'], .9) == [constant_calibration_upper] * len(test)
    external_budgets90 = {}
    for method in comparisons:
        rows = {r['instance_id']: r for r in sweep if r['method'] == method}
        if rows:
            external_budgets90[method] = budget_metrics([rows[r['instance_id']]['upper'] for r in test], actual)
    curve = []
    for size in protocol['fit_sizes']:
        for mode in ['auto', 'median']:
            rows = [r for r in frozen_curve if r['fit_size'] == size and r['mode'] == mode]
            scored = [dict(**point_metrics(r['point'], actual), **budget_metrics(r['upper90'], actual)) for r in rows]
            curve.append(dict(fit_size=size, calibration_size=43, total_history=size+43, mode=mode,
                              selections=dict(Counter(r['selected'] for r in rows)),
                              mae_mean=mean(r['mae'] for r in scored),
                              mae_resampling_p10_p90=[percentile([r['mae'] for r in scored], p) for p in [.1, .9]],
                              coverage_mean=mean(r['coverage'] for r in scored), mean_upper=mean(r['mean_upper'] for r in scored)))
    write(out / 'metrics.json', dict(protocol=protocol, point=metrics, budget=budgets, external_budget90=external_budgets90, learning=curve, selection=selection))
    report = ['# Token 预估与预算验证', '',
              '固定 71 条拟合、43 条校准、158 道已看过结果的成功任务。纯前置；未改变产品参数，未运行解题模型。', '',
              '## 预测误差及配对比较', '',
              '差值为该方法 MAE 减去默认方法 MAE；负值更好。置信区间为固定拟合模型下的 5000 次任务配对 bootstrap，不包含重新训练或仓库聚类不确定性。', '',
              '| 方法 | MAE | WAPE | 平均有符号误差 | MAE 差值 95% 区间 |', '|---|---:|---:|---:|---|']
    for name, m in metrics.items():
        ci = m['comparison_to_auto']['paired_task_bootstrap95']
        report.append(f"| {name} | {m['mae']:,.0f} | {m['wape']:.1%} | {m['signed_bias']:,.0f} | [{ci[0]:,.0f}, {ci[1]:,.0f}] |")
    report += ['', '## 默认方法预算上界', '',
               '80% 和 95% 是实验校准目标；产品 CLI 仍固定输出 90%。覆盖区间为二项 Wilson 近似，依赖任务独立假设。', '',
               '| 目标覆盖 | 实际覆盖 | 覆盖率 95% 区间 | 平均上界 tokens | 预留总量/实际总量 |', '|---|---:|---|---:|---:|']
    for target, b in budgets['auto'].items():
        ci = b['coverage_wilson95']
        report.append(f"| {float(target):.0%} | {b['coverage']:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] | {b['mean_upper']:,.0f} | {b['reserved_to_actual']:.2f} |")
    b = budgets['no_calibration_fit_p90']
    report += ['', f"不校准、仅使用拟合历史 P90 场景：覆盖 {b['coverage']:.1%}，平均上界 {b['mean_upper']:,.0f} tokens。它没有标称覆盖保证。", '',
               '当前默认点预测在所有 code_repair 任务上为常数；其单侧校准上界因此等价于校准集有限样本修正的经验分位数。这个基准已经显式核验，不能把它作为额外的任务自适应效果。', '',
               '## 90% 预算对照', '', '| 方法 | 实际覆盖 | 平均上界 tokens | 预留总量/实际总量 |', '|---|---:|---:|---:|']
    for name, b in [('auto', budgets['auto']['0.9']), *[(k, budgets[k]['0.9']) for k in ['ridge', 'repository', 'retrieval']], *external_budgets90.items()]:
        report.append(f"| {name} | {b['coverage']:.1%} | {b['mean_upper']:,.0f} | {b['reserved_to_actual']:.2f} |")
    report += ['',
               '## 历史量学习曲线', '',
               '每档 20 次固定随机种子抽样，种子内嵌套拟合子集；校准历史始终为 43 条。区间是抽样结果的 P10–P90，不是独立测试置信区间。', '',
               '| 拟合历史 | 总历史（含43条校准） | 默认 MAE 均值 | MAE 抽样 P10–P90 | 90% 上界覆盖均值 | 自动选择 |', '|---:|---:|---:|---|---:|---|']
    for r in curve:
        if r['mode'] == 'auto':
            lo, hi = r['mae_resampling_p10_p90']
            report.append(f"| {r['fit_size']} | {r['total_history']} | {r['mae_mean']:,.0f} | [{lo:,.0f}, {hi:,.0f}] | {r['coverage_mean']:.1%} | {r['selections']} |")
    report += ['', '## 解释边界与未完成证据', '',
               '候选消融见点误差和预算 JSON；只有单一执行配置，因此无法估计配置匹配的贡献。所有任务类型均为 code_repair，因此层级类型收缩与配置中位数在本数据上等价。', '',
               '当前结果不含新独立任务、可确认的自然失败轨迹、真实费用节省或新增解决率。更低预算预留不等于真实消耗减少。历史量曲线保留 43 条校准，不能称为仅 5 条总历史即可可靠校准。', '',
               '在看过结果后没有调整超参数或选择新默认模型；下一次独立验证必须冻结实现并使用另一个未参与开发的数据批次。']
    (out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(dict(auto=metrics['auto'], budget=budgets['auto'], learning=[r for r in curve if r['mode'] == 'auto']), indent=2), flush=True)


if __name__ == '__main__':
    main()
