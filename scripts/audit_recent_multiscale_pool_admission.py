"""Independently replay47recipes, cache identity, FRS and all class bounds."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_recent_multiscale_pool_admission import ROOT, FOLDER, CACHE, BASE, OOF, digest, verified_parent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_certified_diverse_pool_pilot import check_pool
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    base, own = verified_parent(BASE, name), verified_parent(OOF, name)
    assert saved['reference_indices'] == base['reference_indices'] == own['reference_indices']
    recipes = []; d = saved['d']; assert saved['fractions'] == [.5, .65, .8]
    for method in ('fgs2023', 'sfrip2025'):
        for index, path in enumerate(base['outer']['paths'][method]):
            for fraction in (.5, .65, .8):
                recipes.append({'features': sorted(path['ranking'][:int(np.ceil(fraction*d))]),
                                'origin': {'method': method, 'config_index': index, 'fraction': fraction}})
    recipes += [{'features': sorted(own['outer']['stage']['candidate']['features']), 'origin': {'method': 'OOF_ordinary_endpoint'}},
                {'features': list(range(d)), 'origin': {'method': 'full_fallback'}}]
    assert len(recipes) == 47; equivalent(recipes, saved['recipes'])
    X, y = load_dataset(name); ri = np.array(saved['reference_indices']); A, _ = preprocess_fit(X[ri], X[ri])
    counts = Counter(datasets=1)
    accepted = check_pool(A, y[ri], recipes, saved['pool'], counts)
    cache_path = CACHE/(name+'.json'); prior = json.loads(cache_path.read_text())
    cache_audit = json.loads((CACHE/'complete-audit.json').read_text())
    assert cache_audit['passed'] and cache_audit['artifact_hashes'][name] == digest(cache_path)
    cache = {tuple(r['features']): r for r in prior['pool']}
    for record in saved['pool']:
        fs = tuple(record['features']); assert record['cache_reused'] == (fs in cache)
        if fs in cache:
            expected = {**cache[fs], 'origins': record['origins'], 'cache_reused': True}
            equivalent(record, expected)
        counts['cache_reused'] += record['cache_reused']
        counts['new_metric_solves'] += not record['cache_reused'] and record['repair'] is not None
        counts['distinct_supports'] += 1
        counts['accepted_supports'] += accepted[fs]
        counts['accepted_nonfull'] += accepted[fs] and len(fs) < d
    summary_row = {'dataset': name, 'd': d, 'distinct_supports': counts['distinct_supports'], 'accepted_supports': counts['accepted_supports'],
                   'accepted_nonfull': counts['accepted_nonfull'], 'new_metric_solves': counts['new_metric_solves'], 'cache_reused': counts['cache_reused'],
                   'minimum_admitted_count': min(len(fs) for fs, passed in accepted.items() if passed)}
    print(name, 'multiscale pool independently audited', flush=True)
    return {'row': summary_row, 'counts': dict(counts), 'artifact_sha256': digest(artifact)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for future in as_completed([ex.submit(audit, r['dataset']) for r in summary['rows']]):
            result = future.result(); reports.append(result); counts.update(result['counts'])
    equivalent(sorted([r['row'] for r in reports], key=lambda r: r['dataset']), summary['rows'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'scope': 'independent_multiscale_recipe_cache_original_FRS_and_classwise_admission_not_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8'); print(output, flush=True)
