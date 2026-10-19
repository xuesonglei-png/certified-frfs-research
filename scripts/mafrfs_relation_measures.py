"""Published equations 4/5/7/10, conditional on explicit fuzzy relations.

Source: Xu et al., IEEE TFS 34(1), 138-151 (2026), printed p141.
This is a measure-level implementation, NOT an author-faithful FD+/FCE+
algorithm reproduction. R_F, R_B, R_FL and S/N conventions remain external.
No crisp-label dependency or existing maf_framework proxy is substituted.
"""
import numpy as np


def validate_relation(R):
    R = np.asarray(R, float)
    if (R.ndim != 2 or R.shape[0] == 0 or R.shape[0] != R.shape[1]
            or not np.isfinite(R).all() or R.min() < 0 or R.max() > 1
            or not np.allclose(R, R.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(R), 1, atol=1e-12, rtol=0)):
        raise ValueError('finite symmetric reflexive fuzzy relation in [0,1] required')
    return R


def fuzzy_labels_from_full_relation(R_full, y):
    """Eq4: FIXED full-feature relation, including each reflexive self term."""
    R = validate_relation(R_full); y = np.asarray(y)
    if y.shape != (len(R),): raise ValueError('one label per reference object required')
    classes = np.unique(y)
    memberships = np.stack([R[:, y == c].sum(axis=1) for c in classes], axis=1)/R.sum(axis=1)[:, None]
    return classes, memberships


def fuzzy_lower_approximations(R_subset, fuzzy_labels, *, t_conorm, negation):
    """Eq5. S/N must be supplied; their author-specific conventions are unknown."""
    R = validate_relation(R_subset); labels = np.asarray(fuzzy_labels, float)
    if (labels.ndim != 2 or labels.shape[0] != len(R) or labels.shape[1] == 0
            or not np.isfinite(labels).all() or labels.min() < 0 or labels.max() > 1
            or not np.allclose(labels.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError('normalized full-relation fuzzy label memberships required')
    negated = np.asarray(negation(R), float)
    if negated.shape != R.shape or not np.isfinite(negated).all() or negated.min() < 0 or negated.max() > 1:
        raise ValueError('negation returned invalid relation values')
    combined = np.asarray(t_conorm(negated[:, :, None], labels[None, :, :]), float)
    if (combined.shape != (len(R), len(R), labels.shape[1])
            or not np.isfinite(combined).all() or combined.min() < 0 or combined.max() > 1):
        raise ValueError('t-conorm returned invalid membership values')
    return combined.min(axis=1)


def fuzzy_dependency(R_subset, fuzzy_labels, *, t_conorm, negation):
    """Eq7 uses max over classes (vee), NOT sum or crisp nearest-enemy loss."""
    lower = fuzzy_lower_approximations(R_subset, fuzzy_labels, t_conorm=t_conorm, negation=negation)
    return float(lower.max(axis=1).mean())


def fuzzy_conditional_entropy(R_subset, R_fuzzy_labels, *, log_base):
    """Eq10 with declared minimum intersection; R_FL must be supplied explicitly."""
    R, L = validate_relation(R_subset), validate_relation(R_fuzzy_labels)
    if R.shape != L.shape: raise ValueError('relations must share the same reference universe')
    if not np.isfinite(log_base) or log_base <= 1: raise ValueError('explicit logarithm base greater than 1 required')
    ratio = np.minimum(R, L).sum(axis=1)/R.sum(axis=1)
    return float(-np.log(ratio).mean()/np.log(log_base))
