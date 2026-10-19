import numpy as np
import pytest
from robustaware_projection_v1 import hard5_ba, perturbed_hard5_ba, fit_robustaware_projection
from audit_hardaware_guarded_soft5_admission import independent_hard_BA


def test_actual_coordinate_votes_and_noise_replay():
    # Includes exact duplicate points and equal votes to exercise stable ties.
    A = np.array([[0,0],[0,0],[0,.1],[.1,0],[.8,.9],[.9,.8],[1,1],[1,1]], float)
    y = np.array([0,0,0,0,1,1,1,1]); w = np.array([.2,.8])
    assert hard5_ba(A,y,w) == independent_hard_BA(A,y,w)
    scores=[]
    for draw in range(3):
        rng=np.random.default_rng(20261011+draw)
        scores.append(independent_hard_BA(np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),y,w))
    assert perturbed_hard5_ba(A,y,w,draws=3) == np.mean(scores)


def test_invalid_mass_and_radius_are_rejected():
    A=np.arange(16).reshape(8,2)/15; y=np.array([0]*4+[1]*4)
    for kwargs, weights in (({},[.5,.5]),({'eps':.02},[1/np.sqrt(2)]*2),({'draws':0},[1/np.sqrt(2)]*2)):
        with pytest.raises(ValueError): fit_robustaware_projection(A,y,[0,1],weights,**kwargs)
