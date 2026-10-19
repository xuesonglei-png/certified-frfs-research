import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from classwise_shared_attack_v1 import dependency, loss_gradient, attack


def test_gradient_includes_shared_enemy_rows():
    A = np.array([[.1, .2], [.16, .25], [.7, .64], [.85, .93]])
    y = np.array([0, 0, 1, 1]); full = np.array([.45, .37]); red = np.array([.1, .6])
    value, grad = loss_gradient(A, y, full, red, 0)
    assert np.linalg.norm(grad[y == 1]) > 0
    delta = 1e-6
    for i in range(len(A)):
        for a in range(A.shape[1]):
            plus, minus = A.copy(), A.copy()
            plus[i,a] += delta; minus[i,a] -= delta
            derivative = (loss_gradient(plus,y,full,red,0)[0]-loss_gradient(minus,y,full,red,0)[0])/(2*delta)
            assert derivative == pytest.approx(grad[i,a], abs=1e-9)
    np.testing.assert_allclose(grad.sum(axis=0), 0., atol=1e-12)


def test_class_attack_finds_hidden_violation_and_respects_shared_box():
    A = np.array([[.1, .2], [.16, .25], [.7, .64], [.85, .93]])
    y = np.array([0, 0, 1, 1]); full = np.array([.9, .2]); red = np.array([.1, .5])
    report = attack(A,y,full,red,0)
    Z = np.array(report['perturbed_table'])
    assert np.max(abs(Z-A)) <= .01+1e-12 and Z.min() >= 0 and Z.max() <= 1
    def scalar(w):
        return np.mean([min(1., min(sum(w[a]*abs(Z[i,a]-Z[j,a]) for a in range(2))
                                   for j in range(4) if y[j] != y[i])) for i in range(2)])
    assert scalar(full)-scalar(red) == pytest.approx(report['observed_signed_class_loss_lower'], abs=1e-12)
    assert report['class_violation_witness']
    assert report['observed_signed_class_loss_lower'] >= report['nominal_signed_class_loss']
    for run in report['runs']:
        assert all(b > a+1e-12 for a,b in zip(run['loss_trace'], run['loss_trace'][1:]))


def test_identity_and_absent_class():
    A = np.array([[.1], [.5]]); y = np.array([0, 1]); w = np.array([1.])
    assert attack(A,y,w,w,0)['observed_signed_class_loss_lower'] == 0
    with pytest.raises(ValueError, match='absent'):
        dependency(A,y,w,2)
