"""Independent certificate/selection replay plus balanced objective replay."""
import numpy as np
from scipy.special import logsumexp
import audit_nested_joint_group as grouped
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_memberships

grouped.repair.original.FOLDER = grouped.repair.original.ROOT/'results/nested-balanced-joint'
base_audit = grouped.repair.original.audit_stage


def objective(A, y, stage, fit, *, group=False):
    W, b = np.array(fit['coef']), np.array(fit['intercept'])
    classes = np.array(stage['teacher_classes'])
    own = np.argmax(y[:, None] == classes[None, :], axis=1)
    weights = len(y)/(len(classes)*np.bincount(own, minlength=len(classes))[own])
    scores = np.einsum('ndk,qdk->nq', triangular_memberships(A), W)+b
    losses = logsumexp(scores, axis=1)-scores[np.arange(len(y)), own]
    centered = W-W.mean(axis=0, keepdims=True)
    penalty_W = W[1] if len(classes) == 2 else centered
    value = weights@losses+np.sum(penalty_W**2)/(2*stage['teacher_C'])
    if group:
        value += len(y)*.01*np.linalg.norm(centered, axis=(0, 2)).sum()
    np.testing.assert_allclose(value, fit['objective'], rtol=1e-8, atol=1e-7)


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    assert stage['class_balanced']
    A, _ = preprocess_fit(raw, raw)
    seen = set()
    for record in stage['policies'].values():
        fit = record['fit']
        identity = str(fit['coef'])+str(fit['intercept'])
        if identity in seen or 'objective' not in fit:
            continue
        seen.add(identity)
        assert fit['class_balanced']
        objective(A, y, stage, fit)
        counts['balanced_objectives_checked'] = counts.get('balanced_objectives_checked', 0)+1
    fit = stage['joint_fit']
    if 'objective' in fit:
        assert fit['class_balanced']
        objective(A, y, stage, fit, group=True)
        counts['group_objectives_checked'] = counts.get('group_objectives_checked', 0)+1


grouped.repair.original.audit_stage = audit_stage

if __name__ == '__main__':
    grouped.repair.original.main()
