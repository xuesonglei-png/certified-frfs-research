import numpy as np
from certified_frfs.centered_forest_fuzzy_logistic_batched import fit_forest_logistic as strict_fit
from certified_frfs.forest_teacher_guard_ablation import fit_forest_logistic as ablation_fit


def test_disabling_guards_is_explicit_and_cannot_claim_full_teacher_certificate():
    X = np.array([[u, .5, .5] for u in (.1, .9)]*8); y = (X[:, 0] > .5).astype(int)
    W = np.zeros((2, 3, 3)); W[1, 0] = [-100, 0, 100]
    args = (X, y, W, [0., 0.], [0, 1], [0, 1, 2], [(0, 1), (1, 2)])
    strict = strict_fit(*args, pair_penalty=3.)
    same = ablation_fit(*args, pair_penalty=3.)
    free = ablation_fit(*args, pair_penalty=3., enforce_teacher_guards=False)
    assert strict['feasible'] and same['feasible'] and free['feasible']
    np.testing.assert_allclose(strict['objective'], same['objective'], rtol=1e-6)
    assert same['teacher_guards_enforced'] and same['full_teacher_guards_verified']
    assert not free['teacher_guards_enforced'] and not free['full_teacher_guards_verified']
    assert free['teacher_protected_count'] == len(X)
    assert free['objective'] < .1*strict['objective']
