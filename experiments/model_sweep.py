"""Development OOF diagnostics and a retrospective fixed-model benchmark."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--estimator-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path[:0] = [str(args.estimator_root / 'scripts'), str(Path(__file__).parent)]
    import preflight_model as own
    import history_model
    import task_neighbors
    from external_baselines import upper_bounds, validate_partitions, vector
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    import catboost
    from catboost import CatBoostRegressor
    import lightgbm as lgb
    import joblib

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    models = out / 'models'
    models.mkdir(exist_ok=True)
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    tasks = load(args.data_dir / 'tasks.json')
    histories = [json.loads(line) for line in (args.reference_dir / 'history-adaptive.jsonl').read_text().splitlines()]
    query = [json.loads(line) for line in (args.reference_dir / 'predictions.jsonl').read_text().splitlines()]

    def augment(row):
        row = dict(row)
        snap = dict(row['feature_snapshot'])
        snap['repository'] = tasks[row['instance_id']]['repo']
        row['feature_snapshot'] = snap
        return row

    fit = [augment(r) for r in histories if r['split'] == 'fit']
    cal = [augment(r) for r in histories if r['split'] == 'calibration']
    test = [augment(dict(instance_id=r['instance_id'], task=tasks[r['instance_id']]['problem_statement'],
                         task_type='code_repair', feature_snapshot=r['arms']['selected_model']['feature_snapshot'])) for r in query]
    validate_partitions([r['instance_id'] for r in fit], [r['instance_id'] for r in cal], [r['instance_id'] for r in test])
    assert (len(fit), len(cal), len(test)) == (71, 43, 158)
    # All transforms below fit only these rows, never query/calibration data.
    class Features:
        def __init__(self, rows):
            self.text = TfidfVectorizer(max_features=256, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.text.fit([r['task'] for r in rows])
            self.repo = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.repo.fit([[r['feature_snapshot']['repository']] for r in rows])

        def transform(self, rows, mode):
            numeric = np.asarray([vector(r['feature_snapshot']) for r in rows])
            if mode == 'numeric':
                return numeric
            text = self.text.transform([r['task'] for r in rows]).toarray()
            if mode == 'text':
                return text
            repos = self.repo.transform([[r['feature_snapshot']['repository']] for r in rows])
            return np.concatenate([numeric, repos, text], axis=1)

    names = [(f'{method}_{mode}', method, mode) for method in
             ['random_forest', 'extra_trees', 'catboost', 'elastic_net', 'lightgbm'] for mode in ['numeric', 'augmented']]
    names += [('tfidf_ridge', 'text_ridge', 'text'), ('tfidf_knn', 'text_knn', 'text')]
    own_modes = [('ours_hierarchy', 'median'), ('ours_ridge', 'ridge'),
                 ('ours_repository', 'repository'), ('ours_repo_text', 'retrieval'), ('ours_auto', 'auto')]
    all_names = [n for n, _, _ in names] + [n for n, _ in own_modes]
    dump(out / 'frozen-protocol.json', dict(
        status='retrospective_seen_158_benchmark; development_selection_uses_fit_only',
        fit=71, calibration=43, benchmark=158, seed=20261103,
        models=all_names, feature_conditions=dict(numeric='same original 8 counts', augmented='8 counts + fit-only repository onehot + fit-only TFIDF-256',
                                                  text='fit-only TFIDF-256; no repository identity'),
        versions=dict(sklearn=sklearn.__version__, catboost=catboost.__version__, lightgbm=lgb.__version__, numpy=np.__version__),
        fixed_parameters=dict(trees=200, tree_depth=4, tree_min_leaf=5, catboost_iterations=300,
                              catboost_depth=4, catboost_l2=10, elastic_alpha=.1, elastic_l1_ratio=.5,
                              text_ridge_alpha=10, knn_k=5, tfidf_max_features=256,
                              lightgbm_rounds=300, lightgbm_leaves=15, lightgbm_min_leaf=10),
        input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__), args.reference_dir / 'history-adaptive.jsonl', args.reference_dir / 'predictions.jsonl',
                       args.data_dir / 'tasks.json', args.estimator_root / 'scripts/preflight_model.py',
                       args.estimator_root / 'scripts/task_neighbors.py']}))

    def fit_external(method, mode, rows):
        features = Features(rows)
        x = features.transform(rows, mode)
        y = np.log1p([r['actual_tokens'] for r in rows])
        scaler = None
        if method == 'random_forest':
            model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', max_depth=4,
                                           min_samples_leaf=5, max_features=.8, random_state=20261103, n_jobs=1)
        elif method == 'extra_trees':
            model = ExtraTreesRegressor(n_estimators=200, criterion='absolute_error', max_depth=4,
                                        min_samples_leaf=5, max_features=.8, random_state=20261103, n_jobs=1)
        elif method == 'catboost':
            model = CatBoostRegressor(iterations=300, depth=4, learning_rate=.03, loss_function='MAE',
                                       l2_leaf_reg=10, bootstrap_type='No', random_seed=20261103,
                                       thread_count=1, verbose=False, allow_writing_files=False)
        elif method == 'elastic_net':
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            model = ElasticNet(alpha=.1, l1_ratio=.5, max_iter=10000, random_state=20261103)
        elif method == 'lightgbm':
            model = lgb.LGBMRegressor(objective='quantile', alpha=.5, n_estimators=300, learning_rate=.03,
                                      num_leaves=15, min_child_samples=10, max_depth=6, reg_lambda=1,
                                      deterministic=True, force_col_wise=True, random_state=20261103, n_jobs=1, verbosity=-1)
        elif method == 'text_ridge':
            model = Ridge(alpha=10, solver='lsqr')
        else:
            model = None
        if model is not None:
            model.fit(x, y)

        def predict(queries):
            matrix = features.transform(queries, mode)
            if method == 'text_knn':
                similarities = matrix @ x.T
                values = []
                for sim in similarities:
                    indices = sorted(range(len(rows)), key=lambda i: (-sim[i], rows[i]['instance_id']))[:5]
                    pairs = [(float(y[i]), float(sim[i]) ** 2) for i in indices if sim[i] > 0]
                    values.append(task_neighbors.weighted_median(pairs) if pairs else float(median(y)))
                pred = values
            else:
                if scaler is not None:
                    matrix = scaler.transform(matrix)
                pred = model.predict(matrix)
            return np.maximum(1, np.expm1(np.clip(pred, 0, 30))).tolist()
        return predict, (features.text, features.repo, scaler, model, x if model is None else None, y if model is None else None)

    # Development diagnostics: each validation text/vocabulary/repository model
    # is fitted exclusively on the remaining fit tasks (nested for ours_auto).
    oof = {name: [] for name in all_names}
    for fold in range(5):
        validation = [r for r in fit if int(history_model.task_key(r['instance_id']), 16) % 5 == fold]
        training = [r for r in fit if r not in validation]
        for name, method, mode in names:
            predictor, _ = fit_external(method, mode, training)
            oof[name].extend(dict(instance_id=r['instance_id'], prediction=p, actual=r['actual_tokens'])
                             for r, p in zip(validation, predictor(validation)))
        for name, mode in own_modes:
            model, _, _ = own.select_candidate(training, mode)
            oof[name].extend(dict(instance_id=r['instance_id'], prediction=own.token_point(model.predict_log(
                r['task_type'], r['feature_snapshot'], r['task'])[0]), actual=r['actual_tokens']) for r in validation)
        print('OOF fold complete', fold, flush=True)
    oof_metrics = {name: dict(n=len(rows), mae=mean(abs(r['prediction'] - r['actual']) for r in rows),
                               wape=sum(abs(r['prediction'] - r['actual']) for r in rows) / sum(r['actual'] for r in rows))
                   for name, rows in oof.items()}
    dump(out / 'development-oof.json', dict(metrics=oof_metrics, predictions=oof))
    # No additional winner is selected from the large external sweep. The
    # product's own preregistered selector alone chooses among its four modes.
    forecasts = {}
    for name, method, mode in names:
        predictor, fitted = fit_external(method, mode, fit)
        forecasts[name] = (predictor(cal), predictor(test))
        text, repo, scaler, model, train_x, train_y = fitted
        state = dict(text=text, repository=repo, scaler=scaler)
        if method == 'catboost':
            model.save_model(str(models / (name + '.cbm')))
        else:
            state.update(model=model, train_x=train_x, train_y=train_y)
        joblib.dump(state, models / (name + '.joblib'))
        print('Benchmark predictions frozen', name, flush=True)
    choices = {}
    for name, mode in own_modes:
        model, reason, report = own.select_candidate(fit, mode)
        predict = lambda rows: [own.token_point(model.predict_log(r['task_type'], r['feature_snapshot'], r['task'])[0]) for r in rows]
        forecasts[name] = (predict(cal), predict(test))
        choices[name] = dict(selected=model.kind, reason=reason, validation=report)
    dump(out / 'own-selection.json', choices)
    predictions = []
    for name, (cal_points, test_points) in forecasts.items():
        uppers = upper_bounds(cal_points, [r['actual_tokens'] for r in cal], test_points)
        predictions.extend(dict(instance_id=r['instance_id'], method=name, point=p, upper=u)
                           for r, p, u in zip(test, test_points, uppers))
    (out / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in predictions) + '\n', encoding='utf-8')
    # Join the already-seen retrospective benchmark labels only after freezing.
    summaries = load(args.data_dir / 'summaries.json')
    truth = {r['instance_id']: r['actual_tokens'] for r in summaries}
    metrics = {}
    for name in all_names:
        rows = [r for r in predictions if r['method'] == name]
        actual = np.asarray([truth[r['instance_id']] for r in rows])
        points = np.asarray([r['point'] for r in rows])
        upper = np.asarray([r['upper'] for r in rows])
        metrics[name] = dict(n=len(rows), mae=float(np.abs(points - actual).mean()),
                             wape=float(np.abs(points - actual).sum() / actual.sum()),
                             upper_coverage=float((actual <= upper).mean()), mean_upper=float(upper.mean()))
    dump(out / 'metrics.json', dict(status='retrospective_not_fresh_holdout', metrics=metrics, own_choices=choices))
    lines = ['# 预测器改进与多模型对照', '',
             '71 题拟合、43 题校准、158 题为已看过结果的固定回顾性基准。开发 OOF 在 71 题内部进行；没有用基准分数选择默认模型。', '',
             '| 方法 | 开发 OOF MAE | 基准 MAE | WAPE | 上界覆盖率 |', '|---|---:|---:|---:|---:|']
    for name in all_names:
        m = metrics[name]
        lines.append(f"| {name} | {oof_metrics[name]['mae']:,.0f} | {m['mae']:,.0f} | {m['wape']:.1%} | {m['upper_coverage']:.1%} |")
    lines += ['', '## 信息条件', '',
              'numeric：原八项计数。augmented：计数＋仓库身份 one-hot＋拟合数据建立的 TF-IDF。text：仅 TF-IDF。不同信息条件分开报告；不能把额外文本信息带来的变化全归因于模型结构。', '',
              'ours_repository 使用仓库先验收缩；ours_repo_text 加入拟合数据的文本邻居证据；ours_auto 只有五折 OOF 达到固定门槛才启用候选。外部对照用于诊断，不按基准分数挑赢家替换默认。', '',
              '全部数据为单一模型/执行端的成功历史轨迹。没有新调用模型或运行官方评分，不能据此声称新的独立泛化成绩或节省效果。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(metrics=metrics, own_choices=choices), indent=2), flush=True)


if __name__ == '__main__':
    main()
