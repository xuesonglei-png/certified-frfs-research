from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_graph_dependency_v3 import class_upper, memberships
from shared_graph_dependency_v2 import class_upper as signed_upper


def test_positive_target_keeps_improvements_from_cancelling_deterioration():
    A=np.array([[.1,.6],[.6,.1],[.2,.8],[.8,.2]])
    y=np.array([0,0,1,1]); full=np.array([1.,.2]); reduced=np.array([.2,1.])
    delta=(memberships(A,y,full)-memberships(A,y,reduced))[:2]
    assert delta.min()<0 and delta.max()>0
    signed=signed_upper(A,y,full,reduced,0,eps=0.)
    positive=class_upper(A,y,full,reduced,0,eps=0.)
    assert positive['solver_upper_bound']==pytest.approx(np.maximum(0.,delta).mean(),abs=1e-8)
    assert positive['solver_upper_bound']>signed['solver_upper_bound']+1e-3


def test_positive_bound_covers_exact_sign_reference_and_real_perturbations():
    rng=np.random.default_rng(839)
    anomalies = 0
    for _ in range(3):
        A=rng.uniform(.2,.8,(5,2));y=np.array([0,0,1,1,1])
        full,reduced=rng.uniform(.1,1.2,(2,2))
        relaxed=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.)
        exact=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.,exact_abs=True)
        # This is the preserved regression that motivated v4: SciPy/presolve
        # can report a dual below an attained exact model point.
        anomalies += int(exact['observed_target_loss_lower'] > relaxed['solver_upper_bound'] + 1e-7)
    assert anomalies >= 1


def test_all_negative_loss_has_zero_positive_upper_at_zero_radius():
    A=np.array([[.1],[.2],[.7]]);y=np.array([0,0,1])
    report=class_upper(A,y,[.2],[.8],0,eps=0.)
    assert abs(report['solver_upper_bound'])<1e-10
    assert report['nominal_target_loss']==0.
