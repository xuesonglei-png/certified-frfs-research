"""Independent targets/objectives, guard replay,5NN and fold selection audit."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, FOLDER, PARENT, METHODS, COUNTS, WEIGHTED, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_predictive_target_metric_admission import objective
from audit_certified_metric_knn_pool import independent_evaluate, query_certificate
from audit_multiscale_certified_metric_knn_pool import policies
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired
from run_nested_joint_path import holm

from converged_crossgroup_stage import GRID
from audit_predictive_target_knn_pilot import check_stage as old_check_stage
from independent_canonical_polytope import independent_polytope
from independent_crossgroup_soft5 import independent_objective,independent_group_info
from independent_canonical_majority import friend_enemy_bounds


def check_stage(A, y, block, counts):
    old_check_stage(A, y, {'learners': [], 'certificate_checks': block['certificate_checks']}, counts)
    d, mass = A.shape[1], np.sqrt(A.shape[1]); full = np.ones(d)/mass
    original = independent_protected(A, y, full); keys = set()
    for r in block['learners']:
        key = (tuple(r['features']), tuple(r['anchor']), r['family'])
        assert key not in keys; keys.add(key)
        assert r['family'] in ('crossgroup_soft5_unconstrained', 'crossgroup_soft5_constrained')
        fs, anchor, target, fit = r['features'], np.array(r['anchor']), np.array(r['target']), r['fit']
        constrained = r['family'] == 'crossgroup_soft5_constrained'
        assert fit['constrained'] == constrained
        assert fit['crossgroup']==independent_group_info(y)
        for w in (anchor, target):
            assert w.min() >= 0 and w.max()/mass <= 1+1e-10
            np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
        equivalent(fit['weights'], target.tolist())
        assert fit['temperature'] == .05 and fit['reg'] == .01 and fit['max_iter'] == 1000
        assert fit['neighborhood_k'] == 5 and fit['cap_normalized'] == 1
        if fit['status'] in ('unique_simplex', 'no_crossgroup_supervision'):
            assert fit['optimizer_success'] and fit['optimizer_iterations'] == 0
            if fit['status'] == 'unique_simplex': assert len(fs) == 1
            else: assert not fit['crossgroup']['query_indices']
            equivalent(anchor.tolist(), target.tolist())
        else:
            C, rhs = np.empty((0, len(fs))), np.empty(0)
            if constrained:
                protected = original; own = np.zeros(len(A), bool)
                for i in range(len(A)):
                    values = friend_enemy_bounds(A[:, fs], y, anchor, i)
                    if values is not None:
                        raw = np.partition(values, 2)[2]; own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
                original_ok = all(sum((y == c) & original & own) >= int(np.ceil(.9*sum((y == c) & original)-1e-12)) for c in np.unique(y))
                expected = 'original_full' if original_ok else 'own_anchor'
                assert fit['polytope']['protection_reference'] == expected
                if not original_ok: protected = own
                C, rhs = independent_polytope(A[:, fs], y, anchor, protected, fit['polytope'])
                counts['independent_fixed_witness_rows'] += len(C)
            else: assert fit['polytope'] is None
            initial = independent_objective(A[:, fs], y, anchor/mass)
            equivalent(initial, fit['training_objective_initial'])
            if fit['status'] == 'local_crossgroup_soft5_target':
                assert fit['optimizer_success'] and fit['optimizer_status'] == 0 and fit['linear_constraints_verified']
                assert not len(C) or np.all(C@target >= rhs)
                final = independent_objective(A[:, fs], y, target/mass)
                equivalent(final, fit['training_objective_final']); assert final <= initial+1e-8
            else:
                assert 'fallback' in fit['status']; equivalent(anchor.tolist(), target.tolist())
        counts['independent_target_objective_replays'] += 1
        counts['target_optimizer_failures'] += not fit['optimizer_success']


def check_metric(metric, base, block, method, target_index, counts):
    fs, anchor = base['features'], np.array(base['weights'])
    assert metric['features'] == fs and metric['base_origin'] == base['origin'] and metric['target_index'] == target_index
    if method not in WEIGHTED:
        assert target_index == 0 and metric['learner_id'] is metric['guard_id'] is metric['deployed_guard_id'] is None
        assert not metric['guard_fallback'] and metric['interpolation'] == 0.
        equivalent(metric['weights'], anchor.tolist()); return
    learner = block['learners'][metric['learner_id']]
    assert learner['features'] == fs; equivalent(learner['anchor'], anchor.tolist())
    family, t = GRID[target_index]
    assert metric['interpolation'] == t and metric['target_family'] == family == learner['family']
    requested = (1-t)*anchor+t*np.array(learner['target'])
    equivalent(requested.tolist(), metric['requested_weights'])
    if method == 'proposed':
        check = block['certificate_checks'][metric['guard_id']]
        assert check['features'] == fs; equivalent(check['weights'], requested.tolist())
        fallback = not check['accepted']; assert fallback == metric['guard_fallback']
        deployed = block['certificate_checks'][metric['deployed_guard_id']]
        assert deployed['accepted'] and deployed['features'] == fs
        final = anchor if fallback else requested
        equivalent(deployed['weights'], final.tolist()); equivalent(metric['weights'], final.tolist())
        counts['proposed_fallback_options'] += fallback
    else:
        assert metric['guard_id'] is metric['deployed_guard_id'] is None and not metric['guard_fallback']
        equivalent(metric['weights'], requested.tolist())

