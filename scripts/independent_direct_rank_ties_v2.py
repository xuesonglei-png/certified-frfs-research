"""Validate near-tied witness identities without relaxing majority/certification."""
import copy
import numpy as np
from audit_fixed_face_majority_milp import rank_report
from audit_forest_compact_transfer import equivalent


def validate_report(A,y,fs,point,margin,stored):
    fresh=rank_report(A,y,fs,point,margin);adjusted=copy.deepcopy(fresh)
    group=np.asarray(fresh['groups']);mass=A.shape[1]**.5
    # Conservative forward-error scale for normalized coordinate accumulation.
    tolerance=16*np.finfo(float).eps*max(1,len(fs))*2.
    for actual,supplied,out in zip(fresh['rows'],stored['rows'],adjusted['rows'],strict=True):
        i=actual['query'];assert supplied['query']==i
        distance=lambda j:sum(abs(A[j,a]*point[k]/mass-A[i,a]*point[k]/mass) for k,a in enumerate(fs))
        for key,same in (('three_friends',True),('two_allowed_enemies',False)):
            ids=supplied[key];expected=actual[key]
            assert len(ids)==len(expected)==len(set(ids))
            assert all(int(j)==j and 0<=j<len(A) and group[j]!=group[i] and ((y[j]==y[i])==same) for j in ids)
            if ids:
                values=[distance(j) for j in ids];threshold=max(distance(j) for j in expected)
                assert max(values)<=threshold+tolerance, 'Witness outside nearest-rank tie envelope'
                assert all(a<=b+tolerance for a,b in zip(values,values[1:])), 'Witness order not distance-consistent'
            out[key]=ids
        # Leave qualification, scalar gaps and aggregate objective UNCHANGED.
        assert supplied['qualified']==actual['qualified']
    equivalent(adjusted,stored)
    return fresh
