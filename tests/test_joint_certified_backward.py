import numpy as np
from certified_frfs.joint_certified_backward import certified_backward_path
from certified_frfs.feasible_transport import weight_certificate


def test_every_accepted_deletion_has_original_certificate():
    rng = np.random.default_rng(81)
    X = rng.random((16, 4))
    y = np.repeat([0, 1], 8)
    path = certified_backward_path(X, y, [0, 1, 2, 3])
    previous = set(range(4))
    for record in path['trace']:
        fs = record['features']
        assert set(fs).issubset(previous)
        assert len(fs) >= 2
        assert weight_certificate(X, y, fs, np.ones(4)/2, np.array(record['weights'])[fs])['feasible']
        previous = set(fs)
    assert all(r['accepted'] == r['certificate']['feasible'] for r in path['attempts'])
