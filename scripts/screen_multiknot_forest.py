"""Frozen five-knot model mechanism on all15 first-reference internal folds."""
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, source_verified, digest
from screen_bagged_fixed_forest import FOLDER as INPUT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.multiknot_forest import fit_multiknot_forest, scores, box_margins
from review_box_iterations import paired

FOLDER = ROOT/'results/multiknot-forest-mechanism-screen'
METRICS = ('clean_BA', 'certified_BA', 'nominal_logloss')


def evaluate(X, y, fit):
    value = scores(X, fit)
    classes = np.array(fit['classes'])
    own = np.searchsorted(classes, y)
    pred = classes[value.argmax(axis=1)]
    oracle = box_margins(X, y, fit, eps=.01, witnesses=True)
    pair = oracle['pair_margins'].copy()
    pair[np.arange(len(X)), own] = 0.
    losses = logsumexp(value, axis=1)-value[np.arange(len(X)), own]
    def balanced(array):
        return float(np.mean([np.mean(array[y == c]) for c in np.unique(y)]))
    return {'scores': value.tolist(), 'predictions': pred.tolist(), 'pair_margins': pair.tolist(),
            'metrics': {'clean_BA': balanced(pred == y), 'certified_BA': balanced(oracle['margins'] > 1e-10),
                        'nominal_logloss': balanced(losses)}}


def run(name):
    threadpool_limits(1)
    source_verified(name)
    upstream = json.loads((INPUT/(name+'.json')).read_text())
    audit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert audit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed']
    hashes = dict(upstream['fingerprints'])
    for path in (Path(__file__), ROOT/'src/certified_frfs/multiknot_forest.py',
                 ROOT/'research/multiknot-forest-mechanism-protocol.md',
                 INPUT/(name+'.json'), INPUT/(name+'-audit.json')):
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha
    out = FOLDER/(name+'.json')
    folds = []
    if out.exists():
        prior = json.loads(out.read_text())
        assert prior['fingerprints'] == hashes
        if prior['complete']:
            return
        folds = prior['folds']
    first = upstream['rows'][0]
    assert first['outer'] == first['inner'] == 0
    ri, a = np.array(first['outer_reference_indices']), np.array(first['reference_indices'])
    X, y = load_dataset(name)
    raw, labels = X[ri[a]], y[ri[a]]
    for fi, fold in enumerate(first['folds']):
        if fi < len(folds):
            continue
        ca, cb = np.array(fold['reference_indices']), np.array(fold['validation_indices'])
        A, B = preprocess_fit(raw[ca], raw[cb])
        start = time.perf_counter()
        control = {**fold['single'], 'knots': [0., .5, 1.]}
        try:
            fitted = fit_multiknot_forest(A, labels[ca], list(range(A.shape[1])), fold['tree']['edges'],
                                         knot_count=5, C=1., pair_penalty=1.)
            failed = not fitted['optimizer_converged']
        except (ValueError, FloatingPointError) as error:
            fitted = {'failed': True, 'message': str(error)}
            failed = True
        deployed = control if failed else fitted
        evaluations = {'three': evaluate(B, labels[cb], control), 'five': evaluate(B, labels[cb], deployed)}
        folds.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'tree': fold['tree'],
                      'control': control, 'fit': fitted, 'ordinary_fallback': failed,
                      'evaluations': evaluations, 'seconds': time.perf_counter()-start})
        out.write_text(json.dumps({'dataset': name, 'fingerprints': hashes,
            'outer_reference_indices': ri.tolist(), 'reference_indices': a.tolist(),
            'folds': folds, 'complete': len(folds) == len(first['folds']), 'archived_query_labels_used': False,
            'scope': 'fold_isolated_full_support_model_capacity_development_not_SOTA'}, separators=(',', ':')), encoding='utf-8')
        print(name, fi, 'BAgainpp', round(100*(evaluations['five']['metrics']['clean_BA']-evaluations['three']['metrics']['clean_BA']), 6),
              'failed', failed, flush=True)


def summarize(names):
    rows, counts = [], Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        assert saved['complete'] and not saved['archived_query_labels_used']
        metrics = {m: {k: float(np.mean([f['evaluations'][m]['metrics'][k] for f in saved['folds']]))
                       for k in METRICS} for m in ('three', 'five')}
        rows.append({'dataset': name, 'metrics': metrics})
        counts['datasets'] += 1
        counts['folds'] += len(saved['folds'])
        counts['new_failures'] += sum(f['ordinary_fallback'] for f in saved['folds'])
        counts['ordinary_caps'] += sum(not f['control']['optimizer_converged'] for f in saved['folds'])
        counts['positive_clean_datasets'] += metrics['five']['clean_BA'] > metrics['three']['clean_BA']+1e-12
    assert counts['datasets'] == 15 and counts['folds'] == 44
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in ('three', 'five')}
    gain = 100*(means['five']['clean_BA']-means['three']['clean_BA'])
    loggain = means['three']['nominal_logloss']-means['five']['nominal_logloss']
    gate = gain >= .5 and loggain >= 0 and counts['positive_clean_datasets'] >= 8 and counts['new_failures'] == counts['ordinary_caps'] == 0
    result = {'rows': rows, 'means': means, 'counts': dict(counts), 'clean_BA_gain_pp': gain,
              'logloss_gain': loggain, 'numeric_resource_gate': bool(gate),
              'paired_development_description': paired([r['metrics']['five']['clean_BA']-r['metrics']['three']['clean_BA'] for r in rows]),
              'goal_achieved': False, 'archived_query_labels_used': False, 'independent_audit_required': True,
              'scope': 'exposed_internal_validation_full_classifier_not_reduct_or_recent_baseline_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((INPUT/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
