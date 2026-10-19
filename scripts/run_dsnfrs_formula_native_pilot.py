"""ALL15 formula-controlled recent FSS baseline with fold-local selection."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_multiscale_certified_metric_knn_pool import ROOT, FOLDER as PARENT, digest
from dsnfrs_formula_baseline import DirectedSoftNeighborhood
from run_hardaware_soft5_knn_pilot import final_evaluation
from run_certified_metric_knn_pool import evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT/'results/dsnfrs-formula-native-pilot'
GRID = tuple({'k': k, 'beta': beta, 'delta': delta}
             for k in (3, 5, 7) for beta in (1.6, 1.8) for delta in (.02, .04))


def fingerprints(name):
    paths = [Path(__file__), ROOT/'scripts/dsnfrs_formula_baseline.py',
             ROOT/'scripts/independent_dsnfrs_formula.py', ROOT/'scripts/audit_dsnfrs_formula_native_pilot.py',
             ROOT/'scripts/run_hardaware_soft5_knn_pilot.py', ROOT/'scripts/run_certified_metric_knn_pool.py',
             ROOT/'research/dsnfrs-formula-native-pilot-protocol.md',
             PARENT/(name+'.json'), PARENT/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    paths += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def fit(A, y, recipe):
    model = DirectedSoftNeighborhood(A, y, k=recipe['k'], beta=recipe['beta'], variance_ddof=1,
                                    zero_variance_lambda=1e-6, singleton_variance='zero')
    selected, search = model.forward(delta=recipe['delta'])
    fs = sorted(selected); w = (np.ones(len(fs))*np.sqrt(A.shape[1])/len(fs)).tolist() if fs else []
    return {'features': fs, 'weights': w, 'selected_order': selected,
            'recipe': recipe, 'search': search, 'actual_count': len(fs)}


def choose(rows):
    return min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_clean_BA'], 12),
        round(rows[i]['mean_balanced_logloss'], 12), rows[i]['mean_actual_count'], i))


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    parent_path = PARENT/(name+'.json'); parent_audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['artifact_hashes'][name] == digest(parent_path)
    parent = json.loads(parent_path.read_text())
    for rel, sha in parent['fingerprints'].items(): assert digest(ROOT/rel) == sha
    hashes = fingerprints(name); path = FOLDER/(name+'.json')
    if path.exists():
        saved = json.loads(path.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
    X, y = load_dataset(name); ri, vi = np.array(parent['reference_indices']), np.array(parent['validation_indices'])
    raw, labels = X[ri], y[ri]; inner = []
    for bi, block in enumerate(parent['candidate_reference_blocks'][:-1]):
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        A, B = preprocess_fit(raw[ca], raw[cb]); options = []
        for recipe in GRID:
            fitted = fit(A, labels[ca], recipe)
            options.append({**fitted, 'evaluation': evaluate(A, labels[ca], B, labels[cb], fitted['features'], fitted['weights'])})
        inner.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'options': options})
        print(name, 'inner', bi, 'native12 fits complete', flush=True)
    rows = []
    for i, recipe in enumerate(GRID):
        scores = [b['options'][i]['evaluation']['metrics'] for b in inner]
        rows.append({'recipe': recipe, 'mean_clean_BA': float(np.mean([r['clean_BA'] for r in scores])),
                     'mean_balanced_logloss': float(np.mean([r['balanced_logloss'] for r in scores])),
                     'mean_actual_count': float(np.mean([b['options'][i]['actual_count'] for b in inner]))})
    selected = choose(rows); A, B = preprocess_fit(raw, X[vi]); fitted = fit(A, labels, GRID[selected])
    outer = {**fitted, **final_evaluation(A, labels, B, y[vi], fitted)}
    assert hashes == fingerprints(name)
    saved = {'dataset': name, 'd': X.shape[1], 'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
             'fingerprints': hashes, 'inner': inner, 'selection': {'options': rows, 'selected': selected},
             'outer': outer, 'complete': True, 'query_labels_used_for_selection': False,
             'scope': 'all15_exposed_development_formula_native_DSNFRS_not_author_tables_or_confirmation'}
    path.write_text(json.dumps(saved, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', 'BA', outer['evaluation']['metrics']['clean_BA'], 'count', len(fitted['features']), flush=True)


def summarize(names):
    rows = []
    for name in sorted(names):
        r = json.loads((FOLDER/(name+'.json')).read_text()); assert r['complete']
        rows.append({'dataset': name, 'd': r['d'], 'actual_count': r['outer']['actual_count'],
                     'recipe': r['outer']['recipe'], 'metrics': r['outer']['evaluation']['metrics']})
    result = {'rows': rows, 'means': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
              'mean_removed': float(np.mean([1-r['actual_count']/r['d'] for r in rows])),
              'goal_achieved': False, 'independent_audit_required': True,
              'author_table_reproduction_established': False, 'scope': 'all15_development_formula_controlled_recent_baseline'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = sorted(json.loads((PARENT/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(run, name) for name in names]): future.result()
    summarize(names)
