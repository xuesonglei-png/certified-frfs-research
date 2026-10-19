"""All15/all45 outer folds at three fixed, nested training budgets."""
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest
from screen_multiknot_forest import evaluate, METRICS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.multiknot_forest import fit_multiknot_forest
from review_box_iterations import paired

FOLDER = ROOT/'results/forest-sample-depth-diagnosis'
FRACTIONS = (.3, .6, 1.)


def training_budgets(train, labels, fold):
    rng = np.random.default_rng(20261031+fold)
    orders = [rng.permutation(train[labels[train] == c]) for c in np.unique(labels[train])]
    return [np.sort(np.concatenate([order[:max(1, int(np.ceil(fraction*len(order))))] for order in orders]))
            for fraction in FRACTIONS]


def run(name):
    threadpool_limits(1)
    X, y = load_dataset(name)
    paths = [Path(__file__), ROOT/'scripts/screen_multiknot_forest.py',
             ROOT/'src/certified_frfs/multiknot_forest.py', ROOT/'src/certified_frfs/fuzzy_interaction_tree.py',
             ROOT/'src/certified_frfs/validated_pool.py', ROOT/'src/certified_frfs/datasets.py',
             ROOT/'src/certified_frfs/linear_retention.py', ROOT/'research/forest-sample-depth-diagnosis-protocol.md',
             ROOT/'results/development-sample-depth-audit.json', ROOT/'data/processed'/(name+'.npz')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    out, records = FOLDER/(name+'.json'), []
    if out.exists():
        old = json.loads(out.read_text())
        assert old['fingerprints'] == hashes
        if old['complete']:
            return
        records = old['records']
    splits = list(StratifiedKFold(3, shuffle=True, random_state=101).split(X, y))
    for fold, (train, test) in enumerate(splits):
        priorpath = ROOT/'results/predictive-iteration41'/(name+'-s101-f'+str(fold)+'.json')
        prior = json.loads(priorpath.read_text())
        assert prior['train_indices'] == train.tolist() and prior['test_indices'] == test.tolist()
        hashes[priorpath.relative_to(ROOT).as_posix()] = digest(priorpath)
        for fraction, selected in zip(FRACTIONS, training_budgets(train, y, fold)):
            index = len(FRACTIONS)*fold+FRACTIONS.index(fraction)
            if index < len(records):
                continue
            A, B = preprocess_fit(X[selected], X[test])
            start = time.perf_counter()
            tree = interaction_tree(A, y[selected])
            tree_seconds = time.perf_counter()-start
            models, evaluations, costs = {}, {}, {}
            for knot_count in (3, 5):
                key = str(knot_count)
                start = time.perf_counter()
                fit = fit_multiknot_forest(A, y[selected], list(range(A.shape[1])), tree['edges'], knot_count=knot_count)
                costs[key] = {'fit_seconds': time.perf_counter()-start}
                models[key] = fit
                start = time.perf_counter()
                evaluations[key] = evaluate(B, y[test], fit)
                costs[key]['evaluation_with_certificate_seconds'] = time.perf_counter()-start
            records.append({'outer': fold, 'fraction': fraction, 'outer_train_indices': train.tolist(),
                'reference_indices': selected.tolist(), 'validation_indices': test.tolist(), 'tree': tree,
                'tree_seconds': tree_seconds, 'models': models, 'evaluations': evaluations, 'costs': costs})
            out.write_text(json.dumps({'dataset': name, 'raw_n': len(X), 'd': X.shape[1], 'records': records,
                'fingerprints': hashes, 'complete': len(records) == 9, 'development_query_labels_scored': True,
                'query_labels_used_for_selection': False,
                'scope': 'fixed_budget_learning_curve_diagnosis_not_reduct_or_SOTA'}, separators=(',', ':')), encoding='utf-8')
            gain = 100*(evaluations['5']['metrics']['clean_BA']-evaluations['3']['metrics']['clean_BA'])
            print(name, fold, fraction, 'train_n', len(selected), 'BAgainpp', round(gain, 6), flush=True)


def summarize(names):
    rows, counts = [], Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        counts['datasets'] += 1
        counts['models'] += sum(len(r['models']) for r in saved['records'])
        counts['optimizer_caps'] += sum(not fit['optimizer_converged'] for r in saved['records'] for fit in r['models'].values())
        metrics = {}
        for fraction in FRACTIONS:
            records = [r for r in saved['records'] if r['fraction'] == fraction]
            assert len(records) == 3
            metrics[str(fraction)] = {m: {k: float(np.mean([r['evaluations'][m]['metrics'][k] for r in records]))
                                         for k in METRICS} for m in ('3', '5')}
        gain = {f: metric['5']['clean_BA']-metric['3']['clean_BA'] for f, metric in metrics.items()}
        rows.append({'dataset': name, 'metrics': metrics, 'capacity_gain_by_fraction': gain,
                     'capacity_gain_full_minus_small': gain['1.0']-gain['0.3']})
    assert counts['datasets'] == 15 and counts['models'] == 270
    means = {str(f): {m: {k: float(np.mean([r['metrics'][str(f)][m][k] for r in rows])) for k in METRICS}
                      for m in ('3', '5')} for f in FRACTIONS}
    descriptions = {str(f): paired([r['capacity_gain_by_fraction'][str(f)] for r in rows]) for f in FRACTIONS}
    primary = paired([r['capacity_gain_full_minus_small'] for r in rows])
    result = {'rows': rows, 'means': means, 'counts': dict(counts),
              'primary_sample_depth_interaction': primary, 'capacity_gain_by_fraction': descriptions,
              'goal_achieved': False, 'independent_audit_required': True, 'support_selectors_evaluated': False,
              'scope': 'exposed_development_sample_size_diagnosis_not_confirmatory_or_old_gate_reversal'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((ROOT/'results/development-sample-depth-audit.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
