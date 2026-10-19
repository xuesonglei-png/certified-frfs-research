"""Replay independent margins, FRS certificates, predictions and selection."""
import audit_nested_triangular as original
import numpy as np
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit

original.FOLDER = original.ROOT / 'results/nested-triangular-repair'
base_audit_stage = original.audit_stage


def audit_stage(raw, y, query, truth, stage, scores, counts, *, seed=None):
    base_audit_stage(raw, y, query, truth, stage, scores, counts, seed=seed)
    A, _ = preprocess_fit(raw, raw)
    d = A.shape[1]
    seen = set()
    for record in stage['policies'].values():
        candidate = record['candidate']
        identity = tuple(candidate['features'])
        if identity in seen:
            continue
        seen.add(identity)
        trace = candidate['repair_trace']
        assert trace[-1]['features'] == candidate['features']
        assert trace[-1]['certificate']['feasible']
        assert all(not r['certificate']['feasible'] for r in trace[:-1])
        for left, right in zip(trace, trace[1:]):
            assert set(left['features']).issubset(right['features'])
            assert len(right['features']) == len(left['features']) + 1
        fs = candidate['features']
        cert = weight_certificate(A, y, fs, np.ones(d)/np.sqrt(d),
                                  np.array(candidate['weights'])[fs])
        assert cert == candidate['certificate']
        counts['repaired_candidates_checked'] = counts.get('repaired_candidates_checked', 0) + 1


original.audit_stage = audit_stage

if __name__ == '__main__':
    original.main()
