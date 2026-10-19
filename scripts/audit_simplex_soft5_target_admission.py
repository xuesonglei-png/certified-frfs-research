"""Independent endpoint geometry, objective and fresh original guard replay."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from run_simplex_soft5_target_admission import ROOT, FOLDER, PARENT, GRID, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected, friend_enemy_bounds
from audit_classwise_repair_admission import independent_class_certificate
from audit_forest_compact_transfer import equivalent


from independent_soft_topk_objective import independent_objective


from independent_canonical_polytope import independent_polytope


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_or_accuracy_used']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    parent = json.loads((PARENT/(name+'.json')).read_text())
    pa = json.loads((PARENT/'complete-audit.json').read_text())
    assert pa['passed'] and pa['artifact_hashes'][name] == digest(PARENT/(name+'.json'))
    assert saved['reference_indices'] == parent['reference_indices']
    assert len(saved['records']) == len(parent['records'])
    X, y = load_dataset(name); ri = np.array(saved['reference_indices']); labels = y[ri]
    A, _ = preprocess_fit(X[ri], X[ri]); mass = np.sqrt(A.shape[1]); full = np.ones(A.shape[1])/mass
    protected = independent_protected(A, labels, full); counts = Counter(datasets=1)
    old_total, new_total, nontrivial = 0, 0, 0
    for r, old in zip(saved['records'], parent['records']):
        fs, fit = r['features'], r['fit']; assert fs == old['features']
        equivalent(r['anchor'], old['anchor']); anchor, target = np.array(r['anchor']), np.array(r['target'])
        equivalent(fit['weights'], r['target']); np.testing.assert_allclose(target.sum(), mass, atol=1e-10)
        assert target.min() >= 0 and target.max()/mass <= 1+1e-10
        assert fit['temperature'] == .05 and fit['reg'] == .01
        assert fit['cap_normalized'] == 1 and fit['max_iter'] == 200 and fit['constrained']
        if fit['polytope'] is not None and fit['status'] not in ('unique_simplex', 'no_LOO_supervision'):
            assert fit['polytope']['protection_reference'] == 'original_full'
            C, rhs = independent_polytope(A[:, fs], labels, anchor, protected, fit['polytope'])
            counts['independent_witness_rows'] += len(C)
            equivalent(independent_objective(A[:, fs], labels, anchor/mass), fit['training_objective_initial'])
            if fit['status'] == 'local_simplex_soft5_target':
                assert fit['optimizer_success'] and fit['linear_constraints_verified']
                assert not len(C) or np.all(C@target >= rhs)
                value = independent_objective(A[:, fs], labels, target/mass)
                equivalent(value, fit['training_objective_final']); assert value <= fit['training_objective_initial']+1e-8
                counts['independent_objective_replays'] += 1
            else:
                assert 'fallback' in fit['status']; equivalent(target.tolist(), anchor.tolist())
        else:
            assert fit['status'] in ('unique_simplex', 'no_LOO_supervision') and fit['optimizer_success']
            if fit['status'] == 'unique_simplex': assert len(fs) == 1
            else: assert all(n == 1 for n in np.unique(labels, return_counts=True)[1])
            equivalent(target.tolist(), anchor.tolist())
        assert [m['interpolation'] for m in r['mixtures']] == list(GRID)
        previous = sum(m['accepted'] for m in old['mixtures'])
        assert r['previous_admitted_mixtures'] == previous
        assert r['previous_endpoint_accepted'] == old['mixtures'][-1]['accepted']
        counts['previous_admitted_mixtures'] += previous; old_total += previous
        for m in r['mixtures']:
            w = (1-m['interpolation'])*anchor+m['interpolation']*target
            equivalent(w.tolist(), m['weights'])
            cert = weight_certificate(A, labels, fs, full, w, protected=protected)
            equivalent(cert, m['certificate']); classes = independent_class_certificate(labels, cert)
            equivalent(classes, m['class_certificate']); accepted = bool(cert['feasible'] and classes['feasible_all_classes'])
            assert accepted == m['accepted']; shift = float(abs(w-anchor).sum()); equivalent(shift, m['weight_shift_l1'])
            counts['mixtures'] += 1; counts['admitted_mixtures'] += accepted; new_total += accepted
            if accepted and shift > 1e-8: counts['admitted_nontrivial_mixtures'] += 1; nontrivial += 1
        endpoint = r['mixtures'][-1]['accepted']
        counts['new_endpoint_admissions'] += endpoint and not old['mixtures'][-1]['accepted']
        counts['lost_endpoint_admissions'] += not endpoint and old['mixtures'][-1]['accepted']
        counts['supports'] += 1; counts['fit_'+fit['status']] += 1; counts['optimizer_failures'] += not fit['optimizer_success']
    row = {'dataset': name, 'supports': len(saved['records']), 'admitted_nontrivial_mixtures': nontrivial,
           'previous_admitted_mixtures': old_total, 'admitted_mixtures': new_total}
    print(name, 'independent constrained target audit passed', flush=True)
    return {'row': row, 'counts': dict(counts), 'artifact_sha256': digest(artifact)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); counts, reports = Counter(), []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    equivalent(sorted([r['row'] for r in reports], key=lambda r: r['dataset']), summary['rows'])
    assert counts['datasets'] == 15 and counts['supports'] == 86
    equivalent({k: v for k, v in counts.items() if k not in ('datasets', 'independent_witness_rows', 'independent_objective_replays')}, summary['counts'])
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_fixed_witness_and_scalarBrent_soft5_objective_fresh_guards_not_optimizer_refit_or_prediction'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(output, flush=True)
