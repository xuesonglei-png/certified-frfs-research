"""Validate shared-anchor LP lower bounds against independent scalar grids."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_anchor_enemy_triple import triple_lp


def direct_values(lo, hi, xs):
    near = np.maximum(0, np.maximum(lo[2:, 0][None, :]-xs[:, None], xs[:, None]-hi[2:, 0][None, :]))
    far = np.maximum(abs(xs-lo[1, 0]), abs(xs-hi[1, 0]))
    return near.max(axis=1)-far


def test_enemies_cannot_all_be_bad_at_different_anchor_positions():
    # Each individual enemy has a negative minimum versus this fixed friend;
    # their COMMON-anchor maximum has a strictly positive minimum.
    lo = np.array([[0.], [.5], [0.], [.5], [1.]])
    hi = np.array([[1.], [.5], [0.], [.5], [1.]])
    out = triple_lp(lo, hi, 0, [2, 3, 4], 1, np.ones(1))
    # Chord friend upper=.5 yields exact lower0 here, so cannot admit on the
    # true gap .5. This checks that primal/grid gaps are not substituted.
    assert out['lower_bound'] <= 0 and np.min(direct_values(lo, hi, np.linspace(0, 1, 1001))) == pytest.approx(.5)


def test_separated_friend_exact_chord_certifies_joint_enemy_gap():
    lo = np.array([[.4], [.5], [0.], [.45], [1.]])
    hi = np.array([[.49], [.5], [0.], [.45], [1.]])
    out = triple_lp(lo, hi, 0, [2, 3, 4], 1, np.ones(1))
    exact = np.min(direct_values(lo, hi, np.linspace(.4, .49, 1001)))
    assert out['lower_bound'] > .3 and out['lower_bound'] <= exact+1e-10
    assert exact-out['lower_bound'] < 1e-6


def test_random_scalar_duals_never_exceed_independent_grid_minimum():
    rng = np.random.default_rng(20261051)
    for _ in range(24):
        lo = rng.uniform(0, .8, (5, 1)); hi = np.minimum(1., lo+rng.uniform(0, .2, (5, 1)))
        out = triple_lp(lo, hi, 0, [2, 3, 4], 1, np.ones(1))
        exact_upper = np.min(direct_values(lo, hi, np.linspace(lo[0, 0], hi[0, 0], 1001)))
        assert out['lower_bound'] <= exact_upper+1e-10
        assert np.max(out['multipliers'], initial=0) <= 0


def test_reject_invalid_box_or_duplicate_object():
    lo = np.zeros((5, 1)); hi = np.ones((5, 1))
    with pytest.raises(ValueError): triple_lp(lo, hi, 0, [2, 2, 4], 1, np.ones(1))
    with pytest.raises(ValueError): triple_lp(lo, hi, 0, [2, 3, 4], 1, np.array([-1.]))
