from itertools import product
import numpy as np
from signed_classwise_dependency import signed_object_witness_bounds, signed_classwise_dependency
from certified_frfs.core import positive_region


def test_signed_class_bound_is_sound_on_all_corners_and_dominates_previous():
    A = np.array([[0, .03], [1, 0], [.45, .4]]); y = np.array([0, 0, 1]); full = np.ones(2)/np.sqrt(2)
    result = signed_classwise_dependency(A, y, [1], full, np.array([np.sqrt(2)]))
    assert result['class_certificate']['feasible_all_classes']
    assert np.all(np.asarray(result['class_certificate']['upper_bounds']) <= result['star_class_certificate']['upper_bounds'])
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    for bits in product([0, 1], repeat=A.size):
        Z = np.where(np.asarray(bits).reshape(A.shape), hi, lo)
        diff = positive_region(Z, y, weights=full)-positive_region(Z, y, [1], weights=np.array([0, np.sqrt(2)]))
        assert np.all([diff[y == c].mean() <= result['class_certificate']['upper_bounds'][c]+1e-10 for c in (0, 1)])


def test_signed_object_bounds_allow_changing_nearest_enemies_on_joint_tables():
    A = np.array([[.50, .5], [.10, .2], [.46, .3], [.54, .7]])
    y = np.array([0, 0, 1, 1]); wf = np.array([.4, .4]); wr = np.array([.8])
    report = signed_object_witness_bounds(A, y, [0], wf, wr, eps=.04)
    assert len(report['objects'][0]['possible_enemies']) == 2 and report['objects'][0]['uniform_unsaturation']
    lo, hi = np.maximum(0, A-.04), np.minimum(1, A+.04); chosen = set()
    for bits in product([0, 1], repeat=A.size):
        Z = np.where(np.asarray(bits).reshape(A.shape), hi, lo)
        diff = positive_region(Z, y, weights=wf)-positive_region(Z, y, [0], weights=np.array([.8, 0]))
        assert np.all(diff <= np.asarray(report['signed_object_upper_bounds'])+1e-10)
        enemies = np.flatnonzero(y != y[0]); chosen.add(int(enemies[np.argmin(abs(Z[enemies, 0]-Z[0, 0]))]))
    assert chosen == {2, 3}


def test_saturation_branch_cannot_keep_a_negative_signed_bound():
    A = np.array([[0., 0.], [.05, .1], [.9, .9], [1., 1.]])
    y = np.array([0, 0, 1, 1]); wf = np.array([1., 1.]); wr = np.array([3.])
    report = signed_object_witness_bounds(A, y, [0], wf, wr)
    assert all(not r['uniform_unsaturation'] for r in report['objects'])
    assert all(r['upper_bound'] >= 0 for r in report['objects'])
    diff = positive_region(A, y, weights=wf)-positive_region(A, y, [0], weights=np.array([3., 0.]))
    assert np.all(diff <= report['signed_object_upper_bounds'])


def test_random_shared_tables_class_priors_and_zero_identity():
    rng = np.random.default_rng(793); A = rng.uniform(0, 1, (15, 3)); y = np.repeat([0, 1, 2], 5)
    wf = np.ones(3)/np.sqrt(3); wr = np.array([.85, np.sqrt(3)-.85])
    report = signed_classwise_dependency(A, y, [0, 1], wf, wr)
    for _ in range(80):
        Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        diff = positive_region(Z, y, weights=wf)-positive_region(Z, y, [0, 1], weights=np.r_[wr, 0.])
        cls = np.asarray([diff[y == c].mean() for c in (0, 1, 2)])
        assert np.all(cls <= np.asarray(report['class_certificate']['upper_bounds'])+1e-10)
        prior = rng.dirichlet(np.ones(3)); assert prior@cls <= report['class_certificate']['max_upper_bound']+1e-10
    identity = signed_classwise_dependency(A, y, [0, 1, 2], wf, wf)
    np.testing.assert_array_equal(identity['class_certificate']['upper_bounds'], [0., 0., 0.])


def test_independent_scalar_knot_witness_auditor_matches_all_object_bounds():
    from independent_signed_classwise import SignedWitnessValidator
    from audit_forest_compact_transfer import equivalent
    rng = np.random.default_rng(341); A = rng.uniform(0, 1, (15, 3)); y = np.repeat([0, 1, 2], 5)
    wf = np.ones(3)/np.sqrt(3); wr = np.array([.9, np.sqrt(3)-.9])
    report = signed_object_witness_bounds(A, y, [0, 1], wf, wr); validator = SignedWitnessValidator(A, y, wf)
    for i, recorded in enumerate(report['objects']):
        equivalent(recorded, validator.object(i, [0, 1], wr))
