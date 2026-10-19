"""Fixed group penalty and three complete budget curves, no test selection."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_decision_iteration import ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_group_fuzzy import joint_group_scores
from certified_frfs.constrained_fuzzy_logistic import fit_constrained_logistic
from certified_frfs.certificate_repair import repair_support
from run_common_fuzzy_classifier import train, score
from review_box_iterations import paired

SOURCE = ROOT/'results/nested-fuzzy-logistic'
FOLDER = ROOT/'results/joint-group-fuzzy-pilot'


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
        C = split['selected_config']['C']
        teacher = train(A, y[ri], list(range(X.shape[1])), C)
        joint = joint_group_scores(A, y[ri], teacher['coef'], teacher['intercept'], teacher['classes'], C=C)
        full_solve = fit_constrained_logistic(A, y[ri], teacher['coef'], teacher['intercept'],
                                             teacher['classes'], list(range(X.shape[1])), C=C)
        full_fit = {**full_solve, 'classes': teacher['classes']} if full_solve['feasible'] else teacher
        ranking = joint['ranking'] if joint['feasible'] else list(range(X.shape[1]))
        policies = {}
        cache = {tuple(range(X.shape[1])): full_solve}
        for fraction in (.5, .65, .8):
            initial = sorted(ranking[:int(np.ceil(fraction*X.shape[1]))])
            candidate = repair_support(A, y[ri], teacher['coef'], initial)
            fs = candidate['features']
            key = tuple(fs)
            if key not in cache:
                cache[key] = fit_constrained_logistic(A, y[ri], teacher['coef'], teacher['intercept'],
                                                      teacher['classes'], fs, C=C)
            result = cache[key]
            # Failed grouped training is a full fallback, never an index-ranked algorithm.
            success = joint['feasible'] and result['feasible']
            fit = {**result, 'classes': teacher['classes']} if success else full_fit
            actual = fs if success else list(range(X.shape[1]))
            policies[str(fraction)] = {'initial_features': initial, 'candidate': candidate,
                'solver_fit': result, 'fit': fit, 'actual_features': actual, 'fallback': not success,
                'clean': score(B, y[vi], fit)}
        splits.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'teacher_C': C, 'teacher': teacher, 'joint': joint, 'full_solver_fit': full_solve,
            'full_fit': full_fit, 'full_clean': score(B, y[vi], full_fit),
            'teacher_clean': score(B, y[vi], teacher), 'policies': policies})
        print(name, i, 'joint group budgets evaluated', flush=True)
    files = [source, Path(__file__), ROOT/'src/certified_frfs/joint_group_fuzzy.py',
             ROOT/'src/certified_frfs/constrained_fuzzy_logistic.py', ROOT/'research/joint-group-fuzzy-pilot-protocol.md']
    out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': splits,
        'fingerprints': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files}}, indent=2), encoding='utf-8')


def summarize(names):
    rows = []
    for name in names:
        p = json.loads((FOLDER/(name+'.json')).read_text())
        s = p['splits']
        methods = {key: {'clean': float(np.mean([r['policies'][key]['clean'] for r in s])),
                        'removed': float(np.mean([1-len(r['policies'][key]['actual_features'])/p['d'] for r in s]))}
                   for key in ('0.5', '0.65', '0.8')}
        rows.append({'dataset': name, 'methods': methods,
                     'full': float(np.mean([r['full_clean'] for r in s])),
                     'teacher': float(np.mean([r['teacher_clean'] for r in s]))})
    result = {'rows': rows, 'budgets': {key: {
        'mean_clean': float(np.mean([r['methods'][key]['clean'] for r in rows])),
        'mean_removed': float(np.mean([r['methods'][key]['removed'] for r in rows])),
        'vs_full': paired([r['methods'][key]['clean']-r['full'] for r in rows]),
        'vs_teacher': paired([r['methods'][key]['clean']-r['teacher'] for r in rows])}
        for key in ('0.5', '0.65', '0.8')},
        'scope': 'all15_all44_fixed_group_penalty_budget_curve_inherited_teacher_C_development_pilot'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['budgets'])


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            future.result()
    summarize(names)
