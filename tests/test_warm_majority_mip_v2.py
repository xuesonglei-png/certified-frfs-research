import copy
from collections import Counter
import numpy as np
import pytest
from fit_warm_majority_face_milp import fit_warm_majority_face_milp
from audit_warm_majority_mip_v2 import check_fit


def test_known_feasible_start_and_deleted_coordinate_independently_replayed():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    y=np.repeat([0,1],6);anchor=np.ones(2)/np.sqrt(2);seed=np.array([0.,np.sqrt(2)])
    for zeros,start in (([],anchor),([0],seed)):
        point,fit=fit_warm_majority_face_milp(A,y,[0,1],anchor,full_reference=anchor,seed=start,fixed_zero_indices=zeros,time_limit=5.)
        counts=Counter();check_fit(A,y,fit,counts)
        assert counts['independent_known_feasible_seed_assignments']==1 and np.all(point[zeros]==0.)
        assert fit['mip_incumbent'] is not None
        broken=copy.deepcopy(fit);broken['warm_start']['partial_seed_values'][0]=.25
        with pytest.raises(AssertionError):check_fit(A,y,broken,Counter())
