"""Independent balanced deployment and complete backward-path replay."""
import numpy as np
import audit_nested_balanced_joint as original
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_certified_backward import certified_backward_path

original.grouped.repair.original.FOLDER = original.grouped.repair.original.ROOT/'results/nested-joint-backward'
base_audit = original.grouped.repair.original.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit(raw, y, query, truth, stage, scores, counts, seed=seed)
    assert stage['support_rule'] == 'certificate_first_joint_backward'
    if not stage['joint_fit']['feasible']:
        return
    A, _ = preprocess_fit(raw, raw)
    path = certified_backward_path(A, y, stage['joint_fit']['ranking'])
    assert stage['backward_path'] == path
    for fraction in (.5, .65, .8):
        eligible = [r for r in path['trace'] if len(r['features']) >= int(np.ceil(fraction*stage['d']))]
        chosen = eligible[-1]
        record = stage['policies'][str(fraction)]
        candidate = record['failure']['requested_candidate'] if (
            record['fallback'] and 'requested_candidate' in record['failure']) else record['candidate']
        assert candidate['features'] == chosen['features']
        np.testing.assert_allclose(candidate['weights'], chosen['weights'], atol=1e-12)
    counts['backward_paths_checked'] = counts.get('backward_paths_checked', 0)+1
    counts['backward_attempts_checked'] = counts.get('backward_attempts_checked', 0)+len(path['attempts'])


original.grouped.repair.original.audit_stage = audit_stage

if __name__ == '__main__':
    original.grouped.repair.original.main()
