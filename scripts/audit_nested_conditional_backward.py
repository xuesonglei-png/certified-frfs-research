"""Replay conditional ranking, certificates, policies and balanced deployment."""
import numpy as np
import audit_nested_balanced_joint as original
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.conditional_certified_backward import conditional_backward_path
from certified_frfs.triangular_classifier import triangular_memberships

original.grouped.repair.original.FOLDER = original.grouped.repair.original.ROOT/'results/nested-conditional-backward'
base_audit = original.grouped.repair.original.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    assert stage['support_rule'] == 'conditional_teacher_score_certified_backward'
    if not stage['joint_fit']['feasible']:
        return
    A, _ = preprocess_fit(raw, raw)
    path = conditional_backward_path(A, y, stage['teacher_coef'], checks=stage['d'])
    assert stage['conditional_path'] == path
    # Independently form each design; do not reuse the implementation's Gram cache.
    Z = triangular_memberships(A)
    W = np.asarray(stage['teacher_coef'])
    target = np.einsum('ndk,qdk->nq', Z, W-W.mean(axis=0, keepdims=True))
    weights = np.array([1/(len(W)*sum(y == label)) for label in y])
    losses = {}
    def direct_loss(fs):
        key = tuple(fs)
        if key not in losses:
            design = np.column_stack([np.ones(len(A)), Z[:, fs].reshape(len(A), -1)])
            penalty = np.eye(design.shape[1])*.001
            penalty[0, 0] = 0
            beta = np.linalg.solve(design.T@(weights[:, None]*design)+penalty,
                                   design.T@(weights[:, None]*target))
            losses[key] = float(weights@np.mean((design@beta-target)**2, axis=1))
        return losses[key]
    for ranking in path['rankings']:
        active = ranking['features']
        choices = sorted((direct_loss([a for a in active if a != removed]), removed) for removed in active)
        direct = {removed: value for value, removed in choices}
        assert set(direct) == {r['removed'] for r in ranking['choices']}
        ordered = [direct[r['removed']] for r in ranking['choices']]
        np.testing.assert_allclose([r['loss'] for r in ranking['choices']], ordered, rtol=1e-8, atol=1e-10)
        # Raw design and cached Gram differ at roundoff for redundant attributes.
        # Require direct costs to be ascending, permitting only numerical ties.
        assert np.all(np.diff(ordered) >= -1e-10)
        counts['conditional_ranking_costs_checked'] = counts.get('conditional_ranking_costs_checked', 0)+len(choices)
    for step in path['trace']:
        np.testing.assert_allclose(step['reconstruction_loss'], direct_loss(step['features']), rtol=1e-8, atol=1e-10)
    for fraction in (.5, .65, .8):
        chosen = [r for r in path['trace'] if len(r['features']) >= int(np.ceil(fraction*stage['d']))][-1]
        record = stage['policies'][str(fraction)]
        candidate = record['failure']['requested_candidate'] if (
            record['fallback'] and 'requested_candidate' in record['failure']) else record['candidate']
        assert candidate['features'] == chosen['features']
        np.testing.assert_allclose(candidate['weights'], chosen['weights'], atol=1e-12)
    counts['conditional_paths_checked'] = counts.get('conditional_paths_checked', 0)+1
    counts['conditional_attempts_checked'] = counts.get('conditional_attempts_checked', 0)+len(path['attempts'])


original.grouped.repair.original.audit_stage = audit_stage

if __name__ == '__main__':
    original.grouped.repair.original.main()
