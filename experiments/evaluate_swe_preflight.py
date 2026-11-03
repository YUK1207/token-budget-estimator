"""Reproduce the bounded offline SWE cost experiment from sanitized source files.

Inputs are prepared from public data; never execute source code or patches.
This evaluator is not the official SWE-bench patch-scoring harness.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--estimator-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.estimator_root / 'scripts'))
    import estimate
    data, output = args.data_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((data / name).read_text(encoding='utf-8'))
    tasks, summaries, counts = load('tasks.json'), load('summaries.json'), load('repo-signals.json')
    profile_source = load('source-profile.json')
    diagnostic_only = set(load('old-test-ids.json'))
    profile = dict(model=profile_source['llm']['model'],
                   agent='OpenHands-CodeActAgent@' + profile_source['git_commit'],
                   execution_config=profile_source['execution_config'])
    assert len(summaries) == 500 and len({r['instance_id'] for r in summaries}) == 500
    # Verify recorded usage totals without using them as prediction features.
    evidence = collections.defaultdict(list)
    for line in (data / 'usage-evidence.jsonl').read_text().splitlines():
        row = json.loads(line)
        evidence[row['instance_id']].append(row)
    for row in summaries:
        assert row['actual_tokens'] == sum(c['input_tokens'] + c['output_tokens'] for c in evidence[row['instance_id']])
    development = [r for r in summaries[:250] if r['valid'] and r['resolved'] and r['instance_id'] not in diagnostic_only]
    # No hyperparameter or model selection uses these source tasks/outcomes.
    final_test = [r for r in summaries[250:] if r['valid'] and r['resolved']]
    assert not {r['instance_id'] for r in development} & {r['instance_id'] for r in final_test}
    assert not diagnostic_only & {r['instance_id'] for r in development + final_test}
    missing = data / 'intentionally-absent-history'
    assert not missing.exists()
    source_records = {r['instance_id']: r for r in summaries}

    def inputs(identity):
        task = tasks[identity]['problem_statement']
        entry = counts[identity]
        assert 'error' not in entry
        assert entry['snapshot_id'] == tasks[identity]['repo'] + '@' + tasks[identity]['base_commit']
        c = entry['code_file_count']
        size = 'small' if c < 50 else 'medium' if c < 500 else 'large' if c < 2000 else 'very_large'
        signals = estimate.RepoSignals(*(entry[k] for k in
                                        ['file_count', 'code_file_count', 'test_file_count', 'config_file_count']), size)
        initial = source_records[identity].get('initial_request_estimate')
        assert initial is not None
        # The initial system/user request and tools exist before generation.
        # This is a character-count approximation, not provider usage feedback.
        context = max(0, initial - estimate.approx_tokens(task))
        return dict(task=task, cwd=str(data / 'no-checkout-needed'), context_tokens=context,
                    execution_profile=profile, instance_id=identity, repo_signals=signals)

    new_history, legacy_history, partitions = [], [], []
    for row in development:
        identity = row['instance_id']
        split = 'fit' if int(hashlib.sha256(('v2-development:' + identity).encode()).hexdigest(), 16) % 10 < 6 else 'calibration'
        partitions.append(dict(instance_id=identity, split=split))
        for target, kind in [(new_history, 'code_repair'), (legacy_history, 'auto')]:
            pred = estimate.estimate(**inputs(identity), history_path=missing, task_kind=kind)
            target.append(dict(schema_version=2, baseline_version=pred.baseline_version,
                               instance_id=identity, task=pred.task_text, task_type=pred.task_type,
                               execution_profile=profile, actual_tokens=row['actual_tokens'],
                               baseline_midpoint_tokens=pred.baseline_midpoint_tokens,
                               feature_snapshot=pred.feature_snapshot, split=split, outcome='completed'))
    fit = [r for r in new_history if r['split'] == 'fit']
    for name, rows in [('history-adaptive.jsonl', new_history), ('history-legacy.jsonl', legacy_history)]:
        (output / name).write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    dump(output / 'partitions.json', dict(development=partitions, diagnostic_only=sorted(diagnostic_only),
                                         final_test=[r['instance_id'] for r in final_test]))
    # Freeze code/data/protocol BEFORE constructing final test predictions.
    dump(output / 'frozen-design.json', dict(
        protocol='first-250-development-minus-original-29-diagnostics; second-250-final-test; resolved-valid-only',
        ridge_penalty=10, selection_margin=0.05, feature_version=estimate.preflight_model.FEATURE_VERSION,
        code_hashes={str(p.relative_to(args.estimator_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [args.estimator_root / 'scripts' / name for name in
                               ['estimate.py', 'history_model.py', 'preflight_model.py']]},
        data_hashes={name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in
                     ['tasks.json', 'summaries.json', 'repo-signals.json', 'usage-evidence.jsonl', 'old-test-ids.json']},
        profile=profile))
    frozen_predictions = []
    global_median = median(r['actual_tokens'] for r in fit)
    for index, row in enumerate(final_test, 1):
        identity = row['instance_id']
        base = inputs(identity)
        arms = {'raw_rules': estimate.estimate(**base, history_path=missing, predictor='legacy'),
                'prior_ratio': estimate.estimate(**base, history_path=output / 'history-legacy.jsonl', predictor='legacy')}
        for name, model in [('hierarchical_median', 'median'), ('ridge_candidate', 'ridge'), ('selected_model', 'auto')]:
            arms[name] = estimate.estimate(**base, history_path=output / 'history-adaptive.jsonl',
                                           task_kind='code_repair', point_model=model)
        frozen_predictions.append(dict(instance_id=identity, history_median=global_median,
                                       arms={name: estimate.asdict(pred) for name, pred in arms.items()}))
        if index % 25 == 0:
            print('Frozen predictions', index, '/', len(final_test), flush=True)
    (output / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in frozen_predictions) + '\n', encoding='utf-8')
    # All final predictions are frozen before joining actual test labels.
    truth = {r['instance_id']: r['actual_tokens'] for r in final_test}
    scored = []
    for item in frozen_predictions:
        identity = item['instance_id']
        for arm, prediction in item['arms'].items():
            point = prediction['p50_tokens'] if prediction['p50_tokens'] is not None else prediction['midpoint_tokens']
            scored.append(dict(instance_id=identity, arm=arm, point=point, actual=truth[identity],
                               upper=prediction['high_tokens'], method=prediction['point_method'],
                               interval_method=prediction['interval_method'], selection=prediction['model_selection']))
        scored.append(dict(instance_id=identity, arm='configuration_median', point=item['history_median'],
                           actual=truth[identity], upper=None, method='historical_median', interval_method=None, selection=None))
    dump(output / 'scored-tasks.json', scored)
    metrics = {}
    for arm in sorted({r['arm'] for r in scored}):
        rows = [r for r in scored if r['arm'] == arm]
        m = dict(n=len(rows), mae=mean(abs(r['point'] - r['actual']) for r in rows),
                 wape=sum(abs(r['point'] - r['actual']) for r in rows) / sum(r['actual'] for r in rows),
                 bias=mean(r['point'] - r['actual'] for r in rows),
                 point_methods=dict(collections.Counter(r['method'] for r in rows)),
                 selections=dict(collections.Counter(str(r['selection']) for r in rows)))
        if rows[0]['upper'] is not None:
            m.update(upper_coverage=mean(r['actual'] <= r['upper'] for r in rows),
                     mean_upper=mean(r['upper'] for r in rows),
                     mean_positive_budget_excess=mean(max(r['upper'] - r['actual'], 0) for r in rows),
                     interval_methods=dict(collections.Counter(r['interval_method'] for r in rows)))
        metrics[arm] = m
    report = dict(counts=dict(source_tasks=len(summaries), development=len(development),
                              development_partitions=dict(collections.Counter(r['split'] for r in partitions)),
                              diagnostic_only=len(diagnostic_only), final_test=len(final_test),
                              unresolved_termination_unknown=sum(not r['resolved'] for r in summaries)),
                  metrics=metrics,
                  limitations=['Success-only public historical runs, one model/harness; not new solver runs or patch evaluation.',
                               'Fixed archive-order development/test cohorts, not a random or leave-repository-out generalization study.',
                               'Exact base-commit git-tree counts are capped at 12000 blobs; scaffold length is approximate.',
                               'Unresolved runs cannot reliably separate natural failure and truncation, so are excluded.',
                               'All model selection used fit-only validation; no changes may be tuned on this final test.'])
    dump(output / 'metrics.json', report)
    lines = ['# 前置预算预测：第二轮未见题目评估', '',
             f"开发题目 {len(development)}（划分 {report['counts']['development_partitions']}）；上一轮 29 道测试题仅保留为诊断；新测试 {len(final_test)} 道。", '',
             '| 方法 | MAE（token） | WAPE | 上界覆盖率 | 平均预算上界 |', '|---|---:|---:|---:|---:|']
    for arm, m in metrics.items():
        coverage = f"{m['upper_coverage']:.1%}" if 'upper_coverage' in m else '—'
        upper = f"{m['mean_upper']:,.0f}" if 'mean_upper' in m else '—'
        lines.append(f"| {arm} | {m['mae']:,.0f} | {m['wape']:.1%} | {coverage} | {upper} |")
    lines += ['', '## 范围与复现', '',
              '全部为成功解题轨迹上的离线消耗预测；没有调用模型或重跑官方评分。新增精确 base_commit 文件统计和开跑前初始请求的字符长度估计。', '',
              '新版报告的是单侧预算上界；旧版是双侧区间的高端点，二者不能当成同一种区间宽度比较。覆盖目标依赖交换性假设，实际覆盖率单独报告。', '',
              '参见 frozen-design.json、partitions.json、predictions.jsonl、scored-tasks.json 和 metrics.json。本轮测试结果打开后，不再用于本轮调参。', '',
              '来源：https://huggingface.co/datasets/loong0814/openhands_trajectories',
              '题目：https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
