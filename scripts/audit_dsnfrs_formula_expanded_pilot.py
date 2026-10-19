"""Independent native search, inner choice, scalar5NN and query replay."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_dsnfrs_formula_expanded_pilot import ROOT, FOLDER, PARENT, GRID, digest
from independent_dsnfrs_formula import check_fit
from audit_certified_metric_knn_pool import independent_evaluate, query_certificate
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); path = FOLDER/(name+'.json'); r = json.loads(path.read_text())
    assert r['complete'] and not r['query_labels_used_for_selection']
    for rel, sha in r['fingerprints'].items(): assert digest(ROOT/rel) == sha
    parent = json.loads((PARENT/(name+'.json')).read_text()); pa = json.loads((PARENT/'complete-audit.json').read_text())
    assert pa['passed'] and pa['artifact_hashes'][name] == digest(PARENT/(name+'.json'))
    assert parent['reference_indices'] == r['reference_indices'] and parent['validation_indices'] == r['validation_indices']
    X, y = load_dataset(name); ri, vi = np.array(r['reference_indices']), np.array(r['validation_indices'])
    assert not set(ri)&set(vi); raw, labels = X[ri], y[ri]; counts = Counter(datasets=1)
    assert len(r['inner']) == len(parent['candidate_reference_blocks'])-1
    for source, block in zip(parent['candidate_reference_blocks'][:-1], r['inner'], strict=True):
        assert block['reference_indices'] == source['reference_indices'] and block['validation_indices'] == source['validation_indices']
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices'])
        assert not set(ca)&set(cb); A, B = preprocess_fit(raw[ca], raw[cb])
        assert len(block['options']) == len(GRID)
        for recipe, stored in zip(GRID, block['options'], strict=True):
            equivalent(recipe, stored['recipe']); check_fit(A, labels[ca], stored, counts)
            fresh = independent_evaluate(A, labels[ca], B, labels[cb], stored['features'], stored['weights'])
            equivalent(fresh, stored['evaluation']); counts['inner_prediction_replays'] += 1
        counts['inner_folds'] += 1
        print(name, 'inner', counts['inner_folds'], 'all24 native searches and votes independently replayed', flush=True)
    rows = r['selection']['options']; assert len(rows) == len(GRID)
    for i, row in enumerate(rows):
        equivalent(row['recipe'], GRID[i]); metrics = [b['options'][i]['evaluation']['metrics'] for b in r['inner']]
        equivalent(row['mean_clean_BA'], float(np.mean([q['clean_BA'] for q in metrics])))
        equivalent(row['mean_balanced_logloss'], float(np.mean([q['balanced_logloss'] for q in metrics])))
        equivalent(row['mean_actual_count'], float(np.mean([b['options'][i]['actual_count'] for b in r['inner']])))
    selected = min(range(len(GRID)), key=lambda i: (-round(rows[i]['mean_clean_BA'],12),
        round(rows[i]['mean_balanced_logloss'],12), rows[i]['mean_actual_count'], i))
    assert r['selection']['selected'] == selected
    outer = r['outer']; equivalent(outer['recipe'], GRID[selected]); A, B = preprocess_fit(raw, X[vi])
    check_fit(A, labels, outer, counts); fs, w = outer['features'], outer['weights']
    report = independent_evaluate(A, labels, B, y[vi], fs, w)
    cert = query_certificate(A, labels, B, np.array(report['predictions']), fs, w)
    equivalent(cert, outer['query_certificate'])
    correct = (np.array(report['predictions']) == y[vi]) & np.array(cert['certified'])
    report['metrics']['query_certified_BA'] = float(np.mean([np.mean(correct[y[vi] == c]) for c in np.unique(y[vi])]))
    scores = []; assert len(outer['perturbation_evaluations']) == 16
    for draw, stored in enumerate(outer['perturbation_evaluations']):
        rng = np.random.default_rng(20261060+draw)
        noisy_A = np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        noisy_B = np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        fresh = independent_evaluate(noisy_A, labels, noisy_B, y[vi], fs, w)
        equivalent(fresh, stored)
        assert not np.any((np.array(fresh['predictions']) != report['predictions']) & np.array(cert['certified']))
        scores.append(fresh['metrics']['clean_BA']); counts['perturbation_prediction_replays'] += 1
    report['metrics']['mean_perturbed_BA'] = float(np.mean(scores)); equivalent(report, outer['evaluation'])
    counts['outer_prediction_replays'] += 1
    row = {'dataset': name, 'd': r['d'], 'actual_count': outer['actual_count'], 'recipe': outer['recipe'], 'metrics': report['metrics']}
    print(name, 'COMPLETE independently audited formula-native DSNFRS', flush=True)
    return {'dataset': name, 'row': row, 'counts': dict(counts), 'artifact_sha256': digest(path)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); reports, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(audit, row['dataset']) for row in summary['rows']]):
            result = future.result(); reports.append(result); counts.update(result['counts'])
    rows = sorted([r['row'] for r in reports], key=lambda r:r['dataset']); equivalent(rows, summary['rows'])
    assert counts['datasets'] == 15 and counts['inner_folds'] == 44 and counts['inner_prediction_replays'] == 44*len(GRID)
    assert counts['outer_prediction_replays'] == 15 and counts['perturbation_prediction_replays'] == 240
    assert counts['independent_DSN_native_fits'] == 44*len(GRID)+15
    means = {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
    equivalent(means, summary['means']); equivalent(float(np.mean([1-r['actual_count']/r['d'] for r in rows])), summary['mean_removed'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'author_table_reproduction_established': False, 'scope': 'all15_independent_formula_search_cv_votes_audit_not_author_code_or_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8'); print(output,flush=True)
