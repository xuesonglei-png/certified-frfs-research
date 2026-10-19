"""The independent verifier must respect actual stored-distance tie behavior."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from independent_canonical_majority import friend_enemy_bounds
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms


def test_reduced_iris_near_tie_replays_shared_box_margin_for_actual_friends():
    A = np.array([
        [.4761904761904763, .5423728813559322, .5833333333333334],
        [.4285714285714286, .6271186440677966, .5416666666666667],
        [.4761904761904763, .576271186440678, .5416666666666667],
        [.4285714285714286, .5932203389830508, .5833333333333334],
        [.4761904761904763, .6101694915254237, .5416666666666667],
        [0., 0., 0.], [.9, .9, .9], [1., 1., 1.], [.1, .1, .1], [.2, .2, .2],
    ])
    y = np.repeat([1, 0], 5)
    weights = np.array([.5017064158981609, .7630237007827618, .7352698833190778])
    # Correct third friend of object3 is object4, not object2.
    friends, enemies, i = [1, 0, 4], [5, 6, 7, 8, 9], 3
    lo, hi = np.maximum(0, A-.01), np.minimum(1, A+.01)
    expected = []
    for enemy in enemies:
        contrasts = []
        for friend in friends:
            value = 0.
            for a, w in enumerate(weights):
                alternatives = []
                for t in (lo[i, a], hi[i, a]):
                    near = max(0., lo[enemy, a]-t, t-hi[enemy, a])
                    far = max(abs(t-lo[friend, a]), abs(t-hi[friend, a]))
                    alternatives.append(near-far)
                value += w*min(alternatives)
            contrasts.append(value)
        expected.append(min(contrasts))
    np.testing.assert_array_equal(friend_enemy_bounds(A, y, weights, i), expected)


def test_known_transfusion_reduced_geometry_has_identical_protected_counts():
    # Actual failure fold: 332 inner reference objects; reproduce its split
    # from the frozen dataset seed rather than depending on result JSON files.
    from sklearn.model_selection import StratifiedKFold
    X, y = load_dataset('transfusion')
    outer = next(StratifiedKFold(3, shuffle=True, random_state=101).split(X, y))[0]
    ca = list(StratifiedKFold(3, shuffle=True, random_state=20261033).split(X[outer], y[outer]))[2][0]
    A, _ = preprocess_fit(X[outer][ca], X[outer][ca]); A = A[:, [0, 1, 2]]; labels = y[outer][ca]
    weights = np.array([.5051692880459513, .7474153559770246, .7474153559770246])
    bounds = MajorityTerms(A, labels, weights=weights).bounds(range(3), allowed_enemies=2)
    independent = []
    for i in range(len(A)):
        values = friend_enemy_bounds(A, labels, weights, i)
        raw = np.partition(values, 2)[2]
        independent.append(raw-1e-10*(1+abs(raw)))
    np.testing.assert_array_equal(np.array(independent) > 1e-10, bounds > 1e-10)
