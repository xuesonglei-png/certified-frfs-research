"""Rare-class protection differs from population-weighted reference loss."""
from itertools import product
import numpy as np
import pytest
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import positive_region


def test_small_global_bound_can_hide_large_rare_class_loss():
    cert = {'dependency_certificate': {'upper_bound': .090000001,
            'redistributed_object_bounds': [0., 0., 0., .36]}}
    result = classwise_dependency_certificate([0, 0, 0, 1], cert)
    assert not result['feasible_all_classes']
    assert result['max_upper_bound'] == pytest.approx(.36, abs=1e-9)


def test_shared_box_corner_losses_obey_every_class_bound():
    X = np.array([[.1, .2], [.3, .25], [.7, .8], [.9, .6]])
    y = np.array([0, 0, 1, 1])
    wf, wr = np.ones(2)/np.sqrt(2), np.array([np.sqrt(2)])
    cert = weight_certificate(X, y, [0], wf, wr)
    result = classwise_dependency_certificate(y, cert)
    wide = np.array([wr[0], 0.])
    for signs in product((-1., 1.), repeat=X.size):
        Z = np.clip(X+.01*np.array(signs).reshape(X.shape), 0, 1)
        difference = positive_region(Z, y, weights=wf)-positive_region(Z, y, [0], wide)
        for c, upper in zip(result['classes'], result['upper_bounds']):
            assert np.mean(difference[y == c]) <= upper+1e-10


def test_full_identity_zero_bound_and_invalid_object_shape():
    cert = {'dependency_certificate': {'upper_bound': 0., 'raw_redistributed_bound': 0.}}
    result = classwise_dependency_certificate([0, 1, 1], cert)
    assert result['upper_bounds'] == [0., 0.] and result['feasible_all_classes']
    with pytest.raises(ValueError):
        classwise_dependency_certificate([0, 1], {'dependency_certificate': {'redistributed_object_bounds': [.1], 'upper_bound': .1}})
