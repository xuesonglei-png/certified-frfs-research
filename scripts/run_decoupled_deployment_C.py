"""Reference-only teacher choice followed by independently tuned deployment C."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic
from run_common_fuzzy_classifier import score
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-balanced-teacher-backward'
FOLDER = ROOT/'results/decoupled-deployment-C'
CS = (.1, 1., 10.)
FRACTIONS = ('0.5', '0.65', '0.8', '1.0')


def refit(raw, y, stage, C):
    A, _ = preprocess_fit(raw, raw)
    d = A.shape[1]
    W, b, classes = stage['teacher_coef'], stage['teacher_intercept'], stage['teacher_classes']
    full = fit_constrained_logistic(A, y, W, b, classes, list(range(d)), C=C, class_balanced=True)
    actual_full = {**full, 'classes': classes} if full['feasible'] else {
        'coef': W, 'intercept': b, 'classes': classes, 'scope': 'explicit_balanced_teacher_fallback'}
    cache = {tuple(range(d)): full}
    records = {}
    for fraction in FRACTIONS:
        candidate = stage['policies'][fraction]['candidate']
        fs = candidate['features']
        identity = tuple(fs)
        if identity not in cache:
            cache[identity] = fit_constrained_logistic(A, y, W, b, classes, fs, C=C, class_balanced=True)
        fitted = cache[identity]
        records[fraction] = {'candidate': candidate, 'requested_features': fs,
            'actual_features': fs if fitted['feasible'] else list(range(d)), 'solver_fit': fitted,
            'fit': {**fitted, 'classes': classes} if fitted['feasible'] else actual_full,
            'fallback': not fitted['feasible']}
    return {'teacher_C': stage['teacher_C'], 'teacher_coef': W, 'teacher_intercept': b,
            'teacher_classes': classes, 'd': d, 'deployment_C': C, 'policies': records}


def metrics(raw, query, truth, fitted, *, seed=None):
    _, B = preprocess_fit(raw, query)
    queries = [B]
    if seed is not None:
        rng = np.random.default_rng(seed)
        queries += [np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
    out = {}
    for fraction, record in fitted['policies'].items():
        values = [score(Q, truth, record['fit']) for Q in queries]
        out[fraction] = {'clean': values[0], 'retained': len(record['actual_features'])/fitted['d'],
                         'query_noise01': float(np.mean(values[1:])) if seed is not None else None}
    return out


def select(inner):
    options = []
    for C in CS:
        for fraction in FRACTIONS:
            scores = [r['by_C'][str(C)]['metrics'][fraction] for r in inner]
            options.append({'C': C, 'fraction': fraction,
                'score': float(np.mean([r['clean'] for r in scores])),
                'retained': float(np.mean([r['retained'] for r in scores]))})
    best = max(r['score'] for r in options)
    eligible = [r for r in options if r['score'] >= best-.005-1e-12]
    proposed = min(eligible, key=lambda r: (r['retained'], -r['score'], r['C'], float(r['fraction'])))
    full = min((r for r in options if r['fraction'] == '1.0'), key=lambda r: (-r['score'], r['C']))
    return proposed, full, {'options': options, 'eligible': eligible, 'tolerance': .005}


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json')
    saved = json.loads(source.read_text())
    X, y = load_dataset(name)
    records = []
    for index, split in enumerate(saved['splits']):
        ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
        raw, labels = X[ri], y[ri]
        # Teacher C uses ONLY full inner scores from the frozen source pipeline.
        teacher_C = split['full_config']['C']
        inner = []
        for j, previous in enumerate(split['inner']):
            a, b = np.array(previous['reference_indices']), np.array(previous['validation_indices'])
            stage = previous['by_C'][str(teacher_C)]['stage']
            byC = {}
            for C in CS:
                fitted = refit(raw[a], labels[a], stage, C)
                byC[str(C)] = {'stage': fitted, 'metrics': metrics(raw[a], raw[b], labels[b], fitted)}
            inner.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(), 'by_C': byC})
            print(name, index, 'inner', j, 'deployment C grid complete', flush=True)
        proposed, full, selection = select(inner)
        source_stage = split['outer_stages'][str(teacher_C)]
        outer = {str(C): refit(raw, labels, source_stage, C) for C in sorted({proposed['C'], full['C']})}
        values = {C: metrics(raw, X[vi], y[vi], stage, seed=144+index) for C, stage in outer.items()}
        records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'teacher_C': teacher_C, 'inner': inner, 'selection': selection, 'selected_config': proposed,
            'full_config': full, 'outer_stages': outer, 'metrics': values[str(proposed['C'])][proposed['fraction']],
            'full_metrics': values[str(full['C'])]['1.0']})
        print(name, index, 'selected teacher/deployment C', teacher_C, proposed['C'], flush=True)
    files = [source, Path(__file__), ROOT/'research/decoupled-deployment-C-protocol.md',
             ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py', ROOT/'data/processed'/(name+'.npz')]
    (FOLDER/(name+'.json')).write_text(json.dumps({'dataset': name, 'splits': records,
        'fingerprints': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        rows.append({'dataset': name, 'clean': float(np.mean([r['metrics']['clean'] for r in saved['splits']])),
            'full': float(np.mean([r['full_metrics']['clean'] for r in saved['splits']])),
            'removed': float(np.mean([1-r['metrics']['retained'] for r in saved['splits']])),
            'noise': float(np.mean([r['metrics']['query_noise01'] for r in saved['splits']])),
            'full_noise': float(np.mean([r['full_metrics']['query_noise01'] for r in saved['splits']]))})
    report = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
              'mean_full': float(np.mean([r['full'] for r in rows])),
              'mean_removed': float(np.mean([r['removed'] for r in rows])),
              'vs_matched_full': paired([r['clean']-r['full'] for r in rows]),
              'vs_matched_full_noise': paired([r['noise']-r['full_noise'] for r in rows]),
              'scope': 'hierarchical_reference_only_teacher_C_then_independent_deployment_C_and_support_exposed_development'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
