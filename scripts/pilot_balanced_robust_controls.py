"""Frozen exact-count supports; identical C, teachers and balanced objectives."""
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

SOURCE = ROOT/'results/nested-joint-group'
SUPPORTS = ROOT/'results/nested-joint-group-common-budget'
FOLDER = ROOT/'results/balanced-robust-controls-pilot'


def run(name):
    threadpool_limits(1)
    paths = [SOURCE/(name+'.json'), SUPPORTS/(name+'.json')]
    saved, controls = [json.loads(p.read_text()) for p in paths]
    out = FOLDER/(name+'.json')
    if out.exists():
        raise FileExistsError(out)
    X, y = load_dataset(name)
    records = []
    for i, (split, control) in enumerate(zip(saved['splits'], controls['splits'])):
        assert split['reference_indices'] == control['reference_indices']
        assert split['validation_indices'] == control['validation_indices']
        ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
        A, B = preprocess_fit(X[ri], X[vi])
        stage = split['stage']
        C = stage['teacher_C']
        teacher = {'coef': stage['teacher_coef'], 'intercept': stage['teacher_intercept'],
                   'classes': stage['teacher_classes']}
        supports = {method: r['features'] for method, r in control['results'].items()}
        size = len(supports['proposed'])
        assert all(len(fs) == size for fs in supports.values())
        supports['full'] = list(range(X.shape[1]))
        results = {}
        for balanced in (False, True):
            mode = 'balanced' if balanced else 'ordinary'
            full = fit_constrained_logistic(A, y[ri], teacher['coef'], teacher['intercept'],
                teacher['classes'], supports['full'], C=C, class_balanced=balanced)
            fullfit = {**full, 'classes': teacher['classes']} if full['feasible'] else teacher
            cache = {tuple(supports['full']): full}
            outcomes = {}
            for method, fs in supports.items():
                identity = tuple(fs)
                if identity not in cache:
                    cache[identity] = fit_constrained_logistic(A, y[ri], teacher['coef'], teacher['intercept'],
                        teacher['classes'], fs, C=C, class_balanced=balanced)
                fitted = cache[identity]
                actual = {**fitted, 'classes': teacher['classes']} if fitted['feasible'] else fullfit
                actual_fs = fs if fitted['feasible'] else supports['full']
                rng = np.random.default_rng(144+i)
                noise = [score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], actual) for _ in range(3)]
                outcomes[method] = {'requested_features': fs, 'actual_features': actual_fs,
                    'solver_fit': fitted, 'fit': actual, 'fallback': not fitted['feasible'],
                    'clean': score(B, y[vi], actual), 'noise': float(np.mean(noise))}
            results[mode] = outcomes
        records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'teacher_C': C, 'teacher': teacher, 'requested_size': size, 'results': results})
        print(name, i, 'ordinary and balanced shared-training controls scored', flush=True)
    paths += [Path(__file__), ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py',
              ROOT/'research/balanced-robust-controls-protocol.md', ROOT/'data/processed'/(name+'.npz')]
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': records,
        'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        methods = {mode: {method: {'clean': float(np.mean([s['results'][mode][method]['clean'] for s in saved['splits']])),
            'removed': float(np.mean([1-len(s['results'][mode][method]['actual_features'])/saved['d'] for s in saved['splits']])),
            'fallbacks': sum(s['results'][mode][method]['fallback'] for s in saved['splits'])}
            for method in ('proposed', 'fgs2023', 'sfrip2025', 'full')}
            for mode in ('ordinary', 'balanced')}
        rows.append({'dataset': name, 'methods': methods})
    report = {'rows': rows,
        'means': {mode:{m:float(np.mean([r['methods'][mode][m]['clean'] for r in rows]))
                       for m in rows[0]['methods'][mode]} for mode in ('ordinary', 'balanced')},
        'balanced_vs_ordinary': {m:paired([r['methods']['balanced'][m]['clean']-r['methods']['ordinary'][m]['clean'] for r in rows])
                                for m in rows[0]['methods']['ordinary']},
        'within_balanced': {m:paired([r['methods']['balanced']['proposed']['clean']-r['methods']['balanced'][m]['clean'] for r in rows])
                            for m in ('fgs2023', 'sfrip2025', 'full')},
        'scope': 'all15_all44_frozen_supports_shared_C_reference_teacher_exact_margin_balanced_objective_diagnostic',
        'limitations': 'Initial raw counts match; failed models deploy full with actual counts reported. '
                      'Baseline paths and proposed supports were selected previously under other objectives. '
                      'FGS/SFRIP are explicit budget adaptations; classification-margin wrapper is not native algorithm. '
                      'No fresh nested balancing/support selection or independent confirmation.'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
