"""Independent scalar soft loss/LOO votes and constrained witness replay."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, FOLDER, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from independent_soft_topk_objective import independent_objective
from independent_canonical_polytope import independent_polytope
from independent_canonical_majority import friend_enemy_bounds
from audit_forest_compact_transfer import equivalent


def independent_loo(A, y, fs, w):
    D = np.zeros((len(A), len(A)))
    for a, weight in zip(fs, w, strict=True): D += abs(A[:,a,None]*weight-A[None,:,a]*weight)
    np.fill_diagonal(D, np.inf)
    ids = np.argsort(D, axis=1, kind='stable')[:,:5]
    classes = np.unique(y)
    votes = np.array([[sum(y[z] == c for z in row) for c in classes] for row in ids])
    predictions = classes[votes.argmax(axis=1)]
    return float(np.mean([np.mean(predictions[y == c] == c) for c in classes]))


def main():
    threadpool_limits(1)
    path = ROOT/'research/simplex-source-budget-diagnostic.json'
    saved = json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_used'] and len(saved['rows']) == 7
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    total_rows = 0
    for row in saved['rows']:
        name = row['dataset']; source = json.loads((FOLDER/(name+'.json')).read_text())
        block = source['inner'][row['block']]; learner = block['learners'][row['learner']]
        X, y = load_dataset(name); ri = np.array(source['reference_indices']); ca = np.array(block['reference_indices'])
        raw, labels = X[ri][ca], y[ri][ca]; A, _ = preprocess_fit(raw, raw[:1])
        fs, anchor = learner['features'], np.array(learner['anchor'])
        fit = row['retry_fit']; w = np.array(fit['weights']); mass = np.sqrt(A.shape[1])
        assert fit['max_iter'] == 1000 and learner['fit']['max_iter'] == 200
        for vector in (anchor, w):
            assert vector.min() >= 0 and abs(vector.sum()-mass) <= 1e-8
        initial = independent_objective(A[:,fs], labels, anchor/mass)
        final = independent_objective(A[:,fs], labels, w/mass)
        equivalent(initial, row['objective_initial_recomputed'])
        equivalent(final, row['objective_final_recomputed'])
        equivalent(initial, fit['training_objective_initial'])
        if fit['status'] == 'local_simplex_soft5_target':
            assert fit['optimizer_success'] and final <= initial+1e-8
            equivalent(final, fit['training_objective_final'])
        else:
            assert 'fallback' in fit['status']; equivalent(w.tolist(), anchor.tolist())
        full = np.ones(A.shape[1])/mass; protected = independent_protected(A, labels, full)
        fresh = weight_certificate(A, labels, fs, full, w, protected=protected)
        classes = independent_class_certificate(labels, fresh)
        equivalent(fresh, row['original_full_guard']['certificate'])
        equivalent(classes, row['original_full_guard']['class_certificate'])
        assert row['original_full_guard']['accepted'] == bool(fresh['feasible'] and classes['feasible_all_classes'])
        if fit['constrained']:
            own = np.zeros(len(A),bool)
            for i in range(len(A)):
                values = friend_enemy_bounds(A[:,fs], labels, anchor, i)
                if values is not None:
                    raw_bound = np.partition(values,2)[2]
                    own[i] = raw_bound-1e-10*(1+abs(raw_bound)) > 1e-10
            original_ok = all(sum((labels == c)&protected&own) >= int(np.ceil(.9*sum((labels == c)&protected)-1e-12))
                              for c in np.unique(labels))
            assert fit['polytope']['protection_reference'] == ('original_full' if original_ok else 'own_anchor')
            C, rhs = independent_polytope(A[:,fs], labels, anchor, protected if original_ok else own, fit['polytope'])
            assert np.all(C@w >= rhs); total_rows += len(C)
            equivalent(float((C@w-rhs).min()) if len(C) else None, row['minimum_anchor_witness_slack'])
        equivalent(independent_loo(A, labels, fs, anchor), row['hard_training_LOO_BA_anchor'])
        equivalent(independent_loo(A, labels, fs, w), row['hard_training_LOO_BA_retry'])
        print(name, row['block'], row['learner'], 'independently replayed', flush=True)
    result = {'passed': True, 'diagnostic_sha256': digest(path), 'auditor_sha256': digest(Path(__file__)),
              'cases': 7, 'scalar_objectives': 14, 'scalar_LOO_votes': 14,
              'independent_guard_replays': 7, 'independent_constraint_rows': total_rows,
              'goal_achieved': False, 'query_labels_used': False}
    (ROOT/'research/simplex-source-budget-diagnostic-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__': main()
