"""Nominal fuzzy-rough loss witnesses for all fixed V4 training candidates.

The nominal table belongs to the radius-.01 box. A nominal violation proves
failure of that loss constraint; absence of one never proves robustness.
No validation or outer test observations enter these computations.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import positive_region
from diagnose_repair_bottlenecks_v7 import ROOT, NAMES, digest, blockers

FOLDER = ROOT / 'results/nominal-dependency-headroom-v7'


def membership(A, labels, features, weights):
    """Direct scalar-coordinate distance sum, independent of cdist primitive."""
    dist = np.zeros((len(A), len(A)))
    for a, weight in zip(features, weights, strict=True):
        dist += weight * abs(A[:, a, None] - A[None, :, a])
    dist[labels[:, None] == labels[None, :]] = np.inf
    return np.minimum(1., dist.min(axis=1))


def run(name):
    path = FOLDER / (name + '.json')
    if path.exists():
        raise FileExistsError(path)
    source = ROOT / 'results/repair-relation-selector-v4' / (name + '.json')
    source_sha = digest(source)
    parent = json.loads(source.read_text()); assert parent['complete']
    X, y = load_dataset(name)
    records, total = [], Counter()
    start = time.perf_counter()
    for panel in parent['rows']:
        ri = np.array(panel['train_indices'], int)
        for fold, block in enumerate(panel['fit']['inner']):
            ii = np.array(block['fit_indices'], int)
            A, _ = preprocess_fit(X[ri[ii]], X[ri[ii]])
            labels = y[ri[ii]]; classes = np.unique(labels)
            d = A.shape[1]; cache = {}
            def get(fs, weights):
                key = tuple(fs), tuple(weights)
                if key not in cache:
                    cache[key] = membership(A, labels, fs, weights)
                    full = np.zeros(d); full[list(fs)] = weights
                    np.testing.assert_allclose(cache[key], positive_region(A, labels, fs, full),
                                               rtol=1e-10, atol=1e-12)
                    total['independent_membership_replays'] += 1
                return cache[key]
            for family, part in block['families'].items():
                candidates = {}
                for recipe in part['candidates']:
                    for c in recipe['trace']:
                        ref = c['family_certificate']['dependency_certificate'].get('full_weights', c['weights'])
                        candidates[(tuple(c['features']), tuple(c['weights']), tuple(ref))] = c
                for candidate in candidates.values():
                    total['unique_candidates'] += 1
                    failure = blockers(candidate)
                    reduced = get(candidate['features'], candidate['weights'])
                    sides = {}
                    for side, ck, cl in (('original', 'certificate', 'classwise'),
                                          ('learned', 'family_certificate', 'family_classwise')):
                        dep = candidate[ck]['dependency_certificate']
                        wf = dep.get('full_weights', candidate['weights'])
                        reference = get(range(d), wf)
                        signed = reference - reduced
                        objbound = np.asarray(dep.get('redistributed_object_bounds', np.zeros(len(labels))))
                        assert np.max(np.maximum(signed, 0.) - objbound) <= 1e-9
                        class_loss = [float(signed[labels == c].mean()) for c in classes]
                        class_positive = [float(np.maximum(signed[labels == c], 0.).mean()) for c in classes]
                        cb = candidate[cl]['bounds']
                        assert all(a <= b + 1e-9 for a, b in zip(class_positive, cb, strict=True))
                        sides[side] = {'nominal_signed_loss': float(signed.mean()),
                            'nominal_positive_object_loss': float(np.maximum(signed, 0.).mean()),
                            'nominal_class_signed_losses': class_loss,
                            'nominal_class_positive_object_losses': class_positive,
                            'saved_global_upper_bound': dep['upper_bound'], 'saved_class_upper_bounds': cb,
                            'nominal_global_violation': float(signed.mean()) > .1 + 1e-10,
                            'nominal_any_class_violation': max(class_loss) > .1 + 1e-10,
                            'global_bound_minus_nominal_signed': dep['upper_bound'] - float(signed.mean()),
                            'max_class_bound_minus_nominal_signed': max(b-a for a,b in zip(class_loss,cb,strict=True))}
                    dependency_blocked = any('dependency' in s for s in failure)
                    actual_violation = any(s['nominal_global_violation'] or s['nominal_any_class_violation']
                                           for s in sides.values())
                    assert not actual_violation or dependency_blocked
                    total['dependency_blocked'] += int(dependency_blocked)
                    total['nominal_violation_witnesses'] += int(actual_violation)
                    total['dependency_blocked_without_nominal_violation'] += int(dependency_blocked and not actual_violation)
                    records.append({'panel': panel['fold'], 'inner_fold': fold, 'family': family,
                        'features': candidate['features'], 'blockers': failure,
                        'dependency_blocked': dependency_blocked, 'nominal_violation_witness': actual_violation,
                        'sides': sides})
            print(name, panel['fold'], fold, dict(total), flush=True)
    assert digest(source) == source_sha
    result = {'dataset': name, 'complete': True, 'counts': dict(total), 'records': records,
        'source_sha256': source_sha, 'runner_sha256': digest(__file__), 'seconds': time.perf_counter()-start,
        'outer_or_inner_validation_observations_used': False, 'new_prediction_scores': False,
        'goal_achieved': False,
        'scope': 'nominal_train_table_violation_witness_not_robustness_admission_or_generalization',
        'limitations': ['No nominal violation does not imply robust feasibility or sufficient-bound looseness.',
                       'Counts repeat subjects across folds and relation families.',
                       'Saved bound values are consistency-checked, not independently recomputed over perturbations.',
                       'Single-object positive loss differs from signed mean dependency decrease.']}
    path.write_text(json.dumps(result, separators=(',', ':')), encoding='utf8')
    print(name, 'complete', result['counts'], flush=True)


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in (
        Path(__file__), ROOT/'scripts/diagnose_repair_bottlenecks_v7.py',
        ROOT/'src/certified_frfs/core.py', ROOT/'src/certified_frfs/validated_pool.py',
        ROOT/'src/certified_frfs/datasets.py', ROOT/'data/manifest.json')}
    for name in NAMES:
        p = ROOT/'results/repair-relation-selector-v4'/(name+'.json')
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
        hashes['data/processed/'+name+'.npz'] = digest(ROOT/'data/processed'/(name+'.npz'))
    freeze = FOLDER/'source-freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['hashes'] == hashes
    else:
        freeze.write_text(json.dumps({'hashes': hashes, 'created_unix': time.time(),
            'all_six_datasets_required': True, 'no_new_classifier_scoring': True}, indent=2))
    for name in NAMES:
        run(name)
    for rel, value in hashes.items():
        assert digest(ROOT/rel) == value
    rows = [json.loads((FOLDER/(n+'.json')).read_text()) for n in NAMES]
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    summary = {'complete': True, 'datasets': len(rows), 'counts': dict(counts),
               'dataset_counts': {r['dataset']: r['counts'] for r in rows},
               'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in NAMES},
               'source_freeze_sha256': digest(freeze), 'goal_achieved': False,
               'robust_feasibility_established': False}
    (FOLDER/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
