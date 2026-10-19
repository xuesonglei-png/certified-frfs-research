from itertools import product

import numpy as np
from certified_frfs.conditional_completion import ConditionalCompletion1NN, pair_completion_lower


def test_pair_infimum_matches_independent_exhaustive_breakpoint_evaluation():
    R = np.array([[.1, .7, .4], [.8, .2, .9]])
    q = [.3]
    candidates = product(*[sorted(set([0., 1., *R[:, a]])) for a in [1, 2]])
    actual = min((np.abs(np.array([q[0], *p]) - R[0]).sum()
                  - np.abs(np.array([q[0], *p]) - R[1]).sum()) / 3 for p in candidates)
    assert np.isclose(pair_completion_lower(R, 0, 1, [0], q), actual)


def test_shared_anchor_certifies_where_independent_bounds_cannot():
    # Missing coordinate shifts both reference distances by EXACTLY the same amount.
    R = np.array([[0., .5], [1., .5]])
    shared = ConditionalCompletion1NN(R, [0, 1], [0, 1])
    independent = ConditionalCompletion1NN(R, [0, 1], [0, 1], bound='independent')
    result = shared.predict_one(lambda a: [0.4, .9][a])
    assert result['acquired'] == [0] and result['prediction'] == 0
    assert independent.predict_one(lambda a: [0.4, .9][a])['count'] == 2
    for missing in np.linspace(0, 1, 51):
        assert shared.predict_full([.4, missing]) == 0


def test_partial_nominal_class_must_not_stop_if_completion_changes_class():
    R = np.array([[0., 0.], [.2, 1.]])
    model = ConditionalCompletion1NN(R, [0, 1], [0, 1])
    assert model.predict_one(lambda a: [0., 1.][a])['count'] == 2
    assert model.predict_full([0., 0.]) != model.predict_full([0., 1.])


def test_exhaustive_multiclass_acquisition_and_hidden_completion_soundness():
    R = np.array([[0., .2, .8], [.1, .4, .7], [.9, .3, .6], [.8, .9, .1], [.4, .8, .9]])
    y = np.array([0, 0, 1, 1, 2])
    for adaptive, bound in [(False, 'shared'), (True, 'shared'), (False, 'independent')]:
        model = ConditionalCompletion1NN(R, y, [1, 0, 2], bound=bound, adaptive=adaptive)
        for q in product([0., .15, .5, .85, 1.], repeat=3):
            accessed = []
            def get(a):
                assert a not in accessed
                accessed.append(a)
                return q[a]
            result = model.predict_one(get)
            assert accessed == result['acquired']
            assert result['prediction'] == model.predict_full(q)
            if result['early_certified']:
                missing = sorted(set(range(3)) - set(accessed))
                for values in product(*[sorted(set([0., 1., *R[:, a]])) for a in missing]):
                    completion = np.asarray(q).copy()
                    completion[missing] = values
                    assert model.predict_full(completion) == result['prediction']


def test_ties_force_full_read_and_one_class_needs_no_query_reads():
    model = ConditionalCompletion1NN([[.2], [.2]], [1, 0], [0])
    result = model.predict_one(lambda a: .2)
    assert result['count'] == 1 and result['prediction'] == 1
    one = ConditionalCompletion1NN([[.2, .8], [.4, .6]], [7, 7], [0, 1])
    assert one.predict_one(lambda a: (_ for _ in ()).throw(AssertionError()))['count'] == 0
