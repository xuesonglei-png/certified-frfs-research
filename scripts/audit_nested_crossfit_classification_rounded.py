"""Nested crossfit isolation, exact original teacher guarantees and ranking."""
import numpy as np
import audit_nested_conditional_feasible as previous
from certified_frfs.crossfit_classification_rounded import crossfit_classification_path
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships

previous.original.original.FOLDER = previous.original.original.ROOT/'results/nested-crossfit-classification-rounded'


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    assert stage['support_rule'] == 'crossfit_classification_rounded_feasibility_repair' and stage['class_balanced']
    path = crossfit_classification_path(raw, y)
    assert path == stage['conditional_path']
    assert path['balanced_error_tie_decimals'] == 12
    # The restoration auditor operates on the full-reference geometry, whereas
    # crossfit preprocessing must start from raw rows. Inject the freshly
    # reconstructed raw-reference path only for its policy restoration checks.
    previous.conditional_backward_path = lambda *a, **k: path
    delegated = {**stage, 'support_rule': 'teacher_only_conditional_feasibility_repair'}
    previous.audit_stage(raw, y, query, truth, delegated, scores, counts, seed=seed)
    classes, own = np.unique(y, return_inverse=True)
    target = np.eye(len(classes))[own]-1/len(classes)
    folds = []
    splits = inner_splits(y, seed=20261011)
    assert path['crossfit_splits'] == [{'reference_indices': a.tolist(), 'validation_indices': b.tolist()} for a, b in splits]
    for a, b in splits:
        assert set(a).isdisjoint(b) and set(a)|set(b) == set(range(len(y)))
        P, Q = preprocess_fit(raw[a], raw[b])
        w = 1/(len(classes)*np.bincount(own[a], minlength=len(classes))[own[a]])
        folds.append((triangular_memberships(P), triangular_memberships(Q), target[a], target[b], w, own[b]))
    cache = {}
    def loss(fs):
        key = tuple(fs)
        if key not in cache:
            errors, values = [], []
            for P, Q, T, U, w, labels in folds:
                D = np.column_stack([np.ones(len(P)), P[:, fs].reshape(len(P), -1)])
                E = np.column_stack([np.ones(len(Q)), Q[:, fs].reshape(len(Q), -1)])
                penalty = np.eye(D.shape[1])*.001; penalty[0, 0] = 0
                beta = np.linalg.solve(D.T@(w[:, None]*D)+penalty, D.T@(w[:, None]*T))
                scores = E@beta; pred = scores.argmax(axis=1)
                errors.append(float(np.mean([np.mean(pred[labels == c] != c) for c in np.unique(labels)])))
                residual = np.mean((scores-U)**2, axis=1)
                values.append(float(np.mean([residual[labels == c].mean() for c in np.unique(labels)])))
            cache[key] = (round(float(np.mean(errors)), 12), float(np.mean(values)))
        return cache[key]
    for ranking in path['rankings']:
        active = ranking['features']; expected = {a: loss([b for b in active if b != a]) for a in active}
        assert set(expected) == {r['removed'] for r in ranking['choices']}
        direct = [expected[r['removed']] for r in ranking['choices']]
        np.testing.assert_allclose(direct, [(r['balanced_error'], r['MSE']) for r in ranking['choices']], rtol=1e-8, atol=1e-10)
        for left, right in zip(direct, direct[1:]):
            assert left[0] <= right[0]+1e-12
            if left[0] == right[0]:
                assert left[1] <= right[1]+1e-10
        counts['crossfit_ranking_costs_checked'] = counts.get('crossfit_ranking_costs_checked', 0)+len(expected)
    for step in path['trace']:
        np.testing.assert_allclose(loss(step['features']), [step['crossfit_balanced_error'], step['reconstruction_loss']], rtol=1e-8, atol=1e-10)
    counts['crossfit_subfolds_checked'] = counts.get('crossfit_subfolds_checked', 0)+len(folds)


previous.original.original.audit_stage = audit_stage

if __name__ == '__main__':
    previous.original.original.main()
