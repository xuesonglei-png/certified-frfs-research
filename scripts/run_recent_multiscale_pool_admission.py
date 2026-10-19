"""ALL15 training admission at cap-compatible50/65/80percent cardinalities."""
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
from run_recent_pool_certificate_admission import BASE, OOF, FOLDER as CACHE, digest, verified_parent

FOLDER = ROOT/'results/recent-multiscale-pool-admission'
FRACTIONS = (.5, .65, .8)


def recipes(paths, ordinary, d):
    out = []
    for method in ('fgs2023', 'sfrip2025'):
        for index, path in enumerate(paths[method]):
            for fraction in FRACTIONS:
                target = max(1, int(np.ceil(fraction*d)))
                out.append({'features': sorted(path['ranking'][:target]), 'origin': {'method': method, 'config_index': index, 'fraction': fraction}})
    out += [{'features': sorted(ordinary), 'origin': {'method': 'OOF_ordinary_endpoint'}},
            {'features': list(range(d)), 'origin': {'method': 'full_fallback'}}]
    assert len(out) == 47
    return out


def admission(raw, labels, candidates, cached=None):
    context = GeometryStageContext(raw, labels)
    supports = {}
    for recipe in candidates: supports.setdefault(tuple(recipe['features']), []).append(recipe['origin'])
    cache = {} if cached is None else {tuple(r['features']): r for r in cached}
    records = []
    for fs, origins in supports.items():
        if fs in cache:
            previous = cache[fs]
            # Verify preprocessing/ordinary metric identity before geometry reuse.
            assert context.candidate(fs) == previous['ordinary']
            records.append({**previous, 'origins': origins, 'cache_reused': True})
            continue
        start = time.perf_counter(); ordinary = context.candidate(fs)
        classes = classwise_dependency_certificate(labels, ordinary['certificate'])
        accepted = ordinary['certificate']['feasible'] and classes['feasible_all_classes']
        metric, repair = ordinary, None
        if not accepted:
            weights, repair = safeguarded_metric_repair(context.A, labels, list(fs), time_limit=10.)
            if weights is not None:
                wide = np.zeros(context.d); wide[list(fs)] = weights
                metric = {'features': list(fs), 'weights': wide.tolist(), 'certificate': repair['certificate']}
                classes = repair['classwise_dependency_certificate']; accepted = True
        records.append({'features': list(fs), 'origins': origins, 'ordinary': ordinary,
                        'ordinary_class_certificate': classwise_dependency_certificate(labels, ordinary['certificate']),
                        'repair': repair, 'accepted': bool(accepted), 'selected_metric': metric if accepted else None,
                        'selected_class_certificate': classes if accepted else None,
                        'wall_seconds': time.perf_counter()-start, 'cache_reused': False})
    assert any(r['accepted'] and len(r['features']) == context.d for r in records)
    return records


def run(name):
    threadpool_limits(1)
    base, own = verified_parent(BASE, name), verified_parent(OOF, name)
    audit = json.loads((CACHE/'complete-audit.json').read_text()); artifact = CACHE/(name+'.json')
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(artifact)
    cached = json.loads(artifact.read_text()); assert cached['reference_indices'] == base['reference_indices'] == own['reference_indices']
    for rel, sha in cached['fingerprints'].items(): assert digest(ROOT/rel) == sha
    sources = [Path(__file__), ROOT/'research/recent-multiscale-pool-admission-protocol.md',
               BASE/(name+'.json'), BASE/'complete-audit.json', OOF/(name+'.json'), OOF/'complete-audit.json', artifact, CACHE/'complete-audit.json', ROOT/'data/processed'/(name+'.npz')]
    sources += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}; output = FOLDER/(name+'.json')
    if output.exists():
        prior = json.loads(output.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return prior
    X, y = load_dataset(name); ri = np.array(base['reference_indices'])
    candidates = recipes(base['outer']['paths'], own['outer']['stage']['candidate']['features'], X.shape[1])
    records = admission(X[ri], y[ri], candidates, cached['pool'])
    result = {'dataset': name, 'd': X.shape[1], 'fractions': list(FRACTIONS), 'recipes': candidates,
              'reference_indices': ri.tolist(), 'fingerprints': hashes, 'pool': records, 'complete': True,
              'query_labels_or_accuracy_used': False, 'scope': 'all15_training_multiscale_admission_not_prediction'}
    assert all(digest(ROOT/rel) == sha for rel, sha in hashes.items())
    output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', 'distinct', len(records), 'nonfull admitted', sum(r['accepted'] and len(r['features']) < X.shape[1] for r in records), flush=True)
    return result


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = sorted(json.loads((BASE/'complete-audit.json').read_text())['artifact_hashes']); reports = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for f in as_completed([ex.submit(run, name) for name in names]): reports.append(f.result())
    rows = [{'dataset': r['dataset'], 'd': r['d'], 'distinct_supports': len(r['pool']), 'accepted_supports': sum(p['accepted'] for p in r['pool']),
             'accepted_nonfull': sum(p['accepted'] and len(p['features']) < r['d'] for p in r['pool']),
             'new_metric_solves': sum(not p['cache_reused'] and p['repair'] is not None for p in r['pool']),
             'cache_reused': sum(p['cache_reused'] for p in r['pool']),
             'minimum_admitted_count': min(len(p['features']) for p in r['pool'] if p['accepted'])} for r in reports]
    report = {'rows': sorted(rows, key=lambda r: r['dataset']), 'goal_achieved': False, 'independent_audit_required': True,
              'query_labels_or_accuracy_used': False, 'scope': 'all15_training_multiscale_feasibility_not_predictive_success'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8'); print(report, flush=True)
