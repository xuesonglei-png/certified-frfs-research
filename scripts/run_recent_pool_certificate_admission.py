"""All15 training-only feasibility of a diverse same-cardinality support pool."""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from certified_frfs.safeguarded_crossfit_stage import safeguarded_metric_repair
from run_flexible_majority_repair_admission import digest

BASE = ROOT/'results/adaptive-multiknot-recent-pilot'
OOF = ROOT/'results/crossfit-label-loss-recent-pilot'
FOLDER = ROOT/'results/recent-pool-certificate-admission'


def verified_parent(folder, name):
    report = json.loads((folder/'complete-audit.json').read_text())
    artifact = folder/(name+'.json')
    assert report['passed'] and report['artifact_hashes'][name] == digest(artifact)
    saved = json.loads(artifact.read_text())
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    return saved


def run(name):
    threadpool_limits(1)
    baseline, own = verified_parent(BASE, name), verified_parent(OOF, name)
    assert baseline['reference_indices'] == own['reference_indices']
    sources = [Path(__file__), ROOT/'research/recent-pool-certificate-admission-protocol.md',
               BASE/(name+'.json'), BASE/'complete-audit.json', OOF/(name+'.json'), OOF/'complete-audit.json']
    sources += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    output = FOLDER/(name+'.json')
    if output.exists():
        prior = json.loads(output.read_text())
        assert prior['fingerprints'] == hashes
        if prior['complete']:
            return prior
    X, y = load_dataset(name)
    ri = np.array(baseline['reference_indices'])
    context = GeometryStageContext(X[ri], y[ri])
    d, target = context.d, max(1, int(np.ceil(.8*context.d)))
    pool = {}
    def register(fs, origin):
        key = tuple(sorted(fs))
        assert len(set(key)) == len(key)
        pool.setdefault(key, []).append(origin)
    for method in ('fgs2023', 'sfrip2025'):
        for index, path in enumerate(baseline['outer']['paths'][method]):
            assert sorted(path['ranking']) == list(range(d))
            register(path['ranking'][:target], {'method': method, 'config_index': index})
    register(own['outer']['stage']['candidate']['features'], {'method': 'OOF_ordinary_endpoint'})
    register(list(range(d)), {'method': 'full_fallback'})
    records = []
    for fs, origins in pool.items():
        start = time.perf_counter()
        ordinary = context.candidate(fs)
        classes = classwise_dependency_certificate(y[ri], ordinary['certificate'])
        accepted, metric = ordinary['certificate']['feasible'] and classes['feasible_all_classes'], ordinary
        repair = None
        if not accepted:
            weights, repair = safeguarded_metric_repair(context.A, y[ri], list(fs), time_limit=10.)
            if weights is not None:
                embedded = np.zeros(d)
                embedded[list(fs)] = weights
                metric = {'features': list(fs), 'weights': embedded.tolist(), 'certificate': repair['certificate']}
                classes = repair['classwise_dependency_certificate']
                accepted = True
        records.append({'features': list(fs), 'origins': origins, 'ordinary': ordinary,
                        'ordinary_class_certificate': classwise_dependency_certificate(y[ri], ordinary['certificate']),
                        'repair': repair, 'accepted': bool(accepted),
                        'selected_metric': metric if accepted else None,
                        'selected_class_certificate': classes if accepted else None,
                        'wall_seconds': time.perf_counter()-start})
        print(name, 'pool', len(records), '/', len(pool), 'count', len(fs), 'accepted', accepted, flush=True)
    assert any(r['accepted'] and len(r['features']) == d for r in records)
    result = {'dataset': name, 'd': d, 'target_count': target, 'reference_indices': ri.tolist(),
              'fingerprints': hashes, 'pool': records, 'complete': True, 'query_labels_or_accuracy_used': False,
              'scope': 'all15_reference_certified_prior_mixture_candidate_pool_not_predictive_success'}
    assert all(digest(ROOT/rel) == sha for rel, sha in hashes.items())
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((BASE/'complete-audit.json').read_text())['artifact_hashes'])
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            results.append(future.result())
    rows = [{'dataset': r['dataset'], 'd': r['d'], 'target': r['target_count'],
             'distinct_supports': len(r['pool']), 'accepted_supports': sum(p['accepted'] for p in r['pool']),
             'accepted_nonfull': sum(p['accepted'] and len(p['features']) < r['d'] for p in r['pool']),
             'metric_solves': sum(p['repair'] is not None for p in r['pool'])} for r in results]
    report = {'rows': sorted(rows, key=lambda r: r['dataset']), 'goal_achieved': False,
              'independent_audit_required': True, 'query_labels_or_accuracy_used': False,
              'scope': 'all15_training_pool_feasibility_diversity_not_test_accuracy'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)
