"""Frozen certified supports; direct convex logistic score refit diagnostic."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic
from run_common_fuzzy_classifier import score
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-triangular-repair'
FOLDER = ROOT/'results/constrained-logistic-pilot'


def run(name):
    threadpool_limits(1)
    source = SOURCE/(name+'.json')
    out = FOLDER/(name+'.json')
    if out.exists():
        raise FileExistsError(out)
    saved = json.loads(source.read_text())
    X, y = load_dataset(name)
    splits = []
    for i, split in enumerate(saved['splits']):
        ri, vi = np.array(split['reference_indices']), np.array(split['validation_indices'])
        A, B = preprocess_fit(X[ri], X[vi])
        stage = split['stage']
        teacher = {'coef': stage['teacher_coef'], 'intercept': stage['teacher_intercept'],
                   'classes': stage['teacher_classes']}
        candidate = stage['policies'][split['chosen']]['candidate']
        supports = {'proposed': candidate['features'], 'full': list(range(X.shape[1]))}
        results, cache = {}, {}
        for method, fs in supports.items():
            identity = tuple(fs)
            if identity not in cache:
                cache[identity] = fit_constrained_logistic(A, y[ri], teacher['coef'], teacher['intercept'],
                    teacher['classes'], fs, C=stage['teacher_C'])
            fitted = cache[identity]
            if fitted['feasible']:
                fit, actual_features = {**fitted, 'classes': teacher['classes']}, fs
            else:
                fit, actual_features = teacher, supports['full']
            rng = np.random.default_rng(144+i)
            noise = [score(np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1), y[vi], fit) for _ in range(3)]
            results[method] = {'requested_features': fs, 'actual_features': actual_features,
                'solver_fit': fitted, 'fit': fit, 'fallback': not fitted['feasible'],
                'clean': score(B, y[vi], fit), 'noise': float(np.mean(noise))}
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'teacher_C': stage['teacher_C'], 'teacher': teacher, 'candidate': candidate,
            'results': results, 'anchor_metrics': split['metrics']})
        print(name, i, 'direct constrained logistic', results['proposed']['clean'], flush=True)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py',
             ROOT/'research/constrained-logistic-pilot-protocol.md']
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': splits,
        'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []
    for name in names:
        p = json.loads((FOLDER/(name+'.json')).read_text())
        splits = p['splits']
        rows.append({'dataset': name,
            'clean': float(np.mean([s['results']['proposed']['clean'] for s in splits])),
            'full': float(np.mean([s['results']['full']['clean'] for s in splits])),
            'anchor': float(np.mean([s['anchor_metrics']['clean'] for s in splits])),
            'removed': float(np.mean([1-len(s['results']['proposed']['actual_features'])/p['d'] for s in splits])),
            'fallbacks': sum(s['results']['proposed']['fallback'] for s in splits)})
    result = {'rows': rows, 'mean_clean': float(np.mean([r['clean'] for r in rows])),
        'mean_removed': float(np.mean([r['removed'] for r in rows])),
        'vs_matched_full': paired([r['clean']-r['full'] for r in rows]),
        'vs_anchor': paired([r['clean']-r['anchor'] for r in rows]),
        'scope': 'all15_all44_frozen_audited_LP_supports_and_C_classifier_diagnostic_not_nested_reselection'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
