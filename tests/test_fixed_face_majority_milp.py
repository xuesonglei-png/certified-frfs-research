import copy
from collections import Counter
import numpy as np
import pytest
from fit_fixed_face_majority_milp import fit_majority_face_milp,strict_majority_report
from audit_fixed_face_majority_milp import check_fit,rank_report


def fixture():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    return A,np.repeat([0,1],6),np.ones(2)/np.sqrt(2)


def test_dynamic_friend_primal_face_and_original_certificate_are_replayed():
    A,y,anchor=fixture();seed=np.array([0.,np.sqrt(2)])
    for zeros,start in (([],anchor),([0],seed)):
        point,fit=fit_majority_face_milp(A,y,[0,1],anchor,full_reference=anchor,seed=start,fixed_zero_indices=zeros,time_limit=5.)
        counts=Counter();check_fit(A,y,fit,counts)
        assert counts['independent_mip_primal_replays']==1 and np.all(point[zeros]==0.)
        assert fit['final_report']['balanced_qualified_rate']==1.
        broken=copy.deepcopy(fit);broken['mip_incumbent'][len(anchor)]=0.25
        with pytest.raises(AssertionError):check_fit(A,y,broken,Counter())


def test_direct_rank_report_matches_scalar_independent_distances():
    A,y,anchor=fixture()
    for point in (anchor,np.array([0.,np.sqrt(2)]),np.array([np.sqrt(2),0.])):
        expected=rank_report(A,y,[0,1],point,.02)
        actual=strict_majority_report(A,y,[0,1],point)
        assert expected['rows']==actual['rows']
        np.testing.assert_allclose(expected['balanced_qualified_rate'],actual['balanced_qualified_rate'])
