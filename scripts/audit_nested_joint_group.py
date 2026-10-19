"""Independent stages/selection plus grouped norm and scalar margin replay."""
import numpy as np
import audit_nested_triangular_repair as repair
from certified_frfs.validated_pool import preprocess_fit
from audit_triangular_classifier_diagnostic import manual_margins

repair.original.FOLDER = repair.original.ROOT/'results/nested-joint-group'
base_audit = repair.original.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    joint = stage['joint_fit']
    if not joint['feasible']:
        counts['joint_failed_stages'] = counts.get('joint_failed_stages', 0)+1
        for key in ('0.5', '0.65', '0.8'):
            assert stage['policies'][key]['fallback']
            assert len(stage['policies'][key]['candidate']['features']) == stage['d']
        return
    A, _ = preprocess_fit(raw, raw)
    W = np.array(joint['coef'])
    classes = np.array(stage['teacher_classes'])
    original, _ = manual_margins(A, y, np.array(stage['teacher_coef']),
                                  np.array(stage['teacher_intercept']), classes)
    actual, _ = manual_margins(A, y, W, np.array(joint['intercept']), classes)
    mask = original > 1e-10
    assert np.all(actual[mask] >= .5*original[mask]-1e-8)
    assert np.all(actual[mask] > 1e-10)
    np.testing.assert_allclose(actual, joint['deployed_margins'], atol=1e-10)
    norms = np.linalg.norm(W-W.mean(axis=0, keepdims=True), axis=(0, 2))
    np.testing.assert_allclose(norms, joint['group_norms'], atol=1e-12)
    assert sorted(range(stage['d']), key=lambda a:(-norms[a], a)) == joint['ranking']
    counts['joint_models_checked'] = counts.get('joint_models_checked', 0)+1


repair.original.audit_stage = audit_stage

if __name__ == '__main__':
    repair.original.main()
