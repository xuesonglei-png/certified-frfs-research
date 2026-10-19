"""Replay original guarantees and independently compute balanced label costs."""
import numpy as np
import audit_nested_conditional_feasible as previous
from certified_frfs.conditional_label_backward import conditional_label_path
from certified_frfs.triangular_classifier import triangular_memberships
from certified_frfs.validated_pool import preprocess_fit

previous.original.original.FOLDER = previous.original.original.ROOT/'results/nested-conditional-label'
previous.conditional_backward_path = lambda X, y, coef, checks: conditional_label_path(X, y)


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    assert stage['support_rule'] == 'conditional_label_feasibility_repair'
    # Same feasibility-restoration invariants; do not mutate stored provenance.
    delegated = {**stage, 'support_rule': 'teacher_only_conditional_feasibility_repair'}
    previous.audit_stage(raw, y, query, truth, delegated, scores, counts, seed=seed)
    A, _ = preprocess_fit(raw, raw)
    Z = triangular_memberships(A)
    classes, own, sizes = np.unique(y, return_inverse=True, return_counts=True)
    target = np.eye(len(classes))[own]-1/len(classes)
    weights = 1/(len(classes)*sizes[own])
    cache = {}
    def loss(fs):
        key = tuple(fs)
        if key not in cache:
            design = np.column_stack([np.ones(len(A)), Z[:, fs].reshape(len(A), -1)])
            P = np.eye(design.shape[1])*.001; P[0, 0] = 0
            beta = np.linalg.solve(design.T@(weights[:, None]*design)+P, design.T@(weights[:, None]*target))
            cache[key] = float(weights@np.mean((design@beta-target)**2, axis=1))
        return cache[key]
    path = stage['conditional_path']
    for ranking in path['rankings']:
        active = ranking['features']
        direct = {a: loss([b for b in active if b != a]) for a in active}
        assert set(direct) == {r['removed'] for r in ranking['choices']}
        ordered = [direct[r['removed']] for r in ranking['choices']]
        np.testing.assert_allclose([r['loss'] for r in ranking['choices']], ordered, rtol=1e-8, atol=1e-10)
        assert np.all(np.diff(ordered) >= -1e-10)
        counts['label_ranking_costs_checked'] = counts.get('label_ranking_costs_checked', 0)+len(direct)
    for step in path['trace']:
        np.testing.assert_allclose(step['reconstruction_loss'], loss(step['features']), rtol=1e-8, atol=1e-10)


previous.original.original.audit_stage = audit_stage

if __name__ == '__main__':
    previous.original.original.main()
