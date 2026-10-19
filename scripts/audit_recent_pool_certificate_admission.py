"""Replay pool origins, original contracts, and independent class aggregation."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_recent_pool_certificate_admission import ROOT, BASE, OOF, FOLDER, digest, verified_parent
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate
from audit_flexible_majority_repair_admission import independent_protected, check_repair
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1)
    saved = json.loads((FOLDER/(name+'.json')).read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    base, own = verified_parent(BASE, name), verified_parent(OOF, name)
    assert saved['reference_indices'] == base['reference_indices'] == own['reference_indices']
    X, y = load_dataset(name)
    ri = np.array(saved['reference_indices'])
    ctx = GeometryStageContext(X[ri], y[ri])
    protected = independent_protected(ctx.A, y[ri], ctx.original_weights)
    np.testing.assert_array_equal(protected, ctx.protected)
    expected = {}
    def register(fs, origin):
        expected.setdefault(tuple(sorted(fs)), []).append(origin)
    for method in ('fgs2023', 'sfrip2025'):
        for i, path in enumerate(base['outer']['paths'][method]):
            register(path['ranking'][:saved['target_count']], {'method': method, 'config_index': i})
    register(own['outer']['stage']['candidate']['features'], {'method': 'OOF_ordinary_endpoint'})
    register(range(ctx.d), {'method': 'full_fallback'})
    assert [tuple(r['features']) for r in saved['pool']] == list(expected)
    counts = Counter(datasets=1)
    for record in saved['pool']:
        fs = record['features']
        assert record['origins'] == expected[tuple(fs)]
        equivalent(ctx.candidate(fs), record['ordinary'])
        classes = independent_class_certificate(y[ri], record['ordinary']['certificate'])
        equivalent(classes, record['ordinary_class_certificate'])
        ordinary_pass = record['ordinary']['certificate']['feasible'] and classes['feasible_all_classes']
        info = record['repair']
        accepted = ordinary_pass
        metric = record['ordinary']
        if info is not None:
            assert not ordinary_pass and info['time_limit'] == 10.
            counts['metric_solves'] += 1
            if 'weights' in info:
                fresh = weight_certificate(ctx.A, y[ri], fs, ctx.original_weights, np.array(info['weights']), protected=protected)
                equivalent(fresh, info['certificate'])
                classes = independent_class_certificate(y[ri], fresh)
                equivalent(classes, info['classwise_dependency_certificate'])
                accepted = fresh['feasible'] and classes['feasible_all_classes']
                if fresh['feasible']:
                    check_repair(ctx.A, y[ri], info, counts)
                wide = np.zeros(ctx.d); wide[fs] = info['weights']
                metric = {'features': fs, 'weights': wide.tolist(), 'certificate': fresh}
            else:
                assert info['status'] in ('no_incumbent', 'incumbent_constraints_failed', 'normalized_cap_failed')
                accepted = False
        else:
            assert ordinary_pass
        assert record['accepted'] == accepted
        if accepted:
            equivalent(metric, record['selected_metric'])
            equivalent(classes, record['selected_class_certificate'])
        else:
            assert record['selected_metric'] is record['selected_class_certificate'] is None
        counts['distinct_supports'] += 1
        counts['accepted_supports'] += accepted
        counts['accepted_nonfull'] += accepted and len(fs) < ctx.d
    assert any(r['accepted'] and len(r['features']) == ctx.d for r in saved['pool'])
    return counts


if __name__ == '__main__':
    report = json.loads((FOLDER/'complete.json').read_text())
    counts = Counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(audit, r['dataset']) for r in report['rows']]):
            counts.update(result.result())
    for key in ('distinct_supports', 'accepted_supports', 'accepted_nonfull', 'metric_solves'):
        assert counts[key] == sum(r[key] for r in report['rows'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['dataset']: digest(FOLDER/(r['dataset']+'.json')) for r in report['rows']},
              'auditor_sha256': digest(Path(__file__)), 'summary_sha256': digest(FOLDER/'complete.json'), 'goal_achieved': False,
              'scope': 'independent_pool_origins_class_aggregation_original_contract_replay_not_predictive_success'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(output, flush=True)
