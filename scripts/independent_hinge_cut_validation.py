"""Validate an actual convex lower face, including nonunique subgradients."""
from itertools import combinations
import numpy as np
from independent_majority_hinge import coefficient,independent_oracle
from audit_forest_compact_transfer import equivalent


def validate_cut(A,point,info,cut):
    i=cut['query'];row=info['witnesses'][i];ids=cut['enemy_indices'];friends=cut['worst_friend_indices']
    assert row['constant_loss'] is None and len(ids)==len(set(ids))==len(friends)==3
    assert all(int(j)==j and 0<=j<len(row['enemies']) for j in ids)
    assert all(int(j)==j and 0<=j<3 for j in friends)
    active=int(round(3*cut['intercept']));assert 0<=active<=3
    equivalent(active/3,cut['intercept'])
    C=[coefficient(A,i,row['enemies'][e],row['friends'][f],info['eps']) for e,f in zip(ids,friends)]
    valid=[]
    for subset in combinations(range(3),active):
        gradient=-sum((C[j] for j in subset),start=np.zeros(len(point)))/(3*info['margin'])
        if np.allclose(gradient,cut['gradient'],rtol=1e-9,atol=1e-10):valid.append(subset)
    assert valid,'Stored cut is not a valid selected-enemy hinge lower face'
    value,canonical=independent_oracle(A,point,{**info,'witnesses':[row]})
    # Any sum of <=3 selected enemy affine hinges is globally below top3.
    # Tangency, rather than identical canonical gradient, is the correct
    # requirement when tied friends create multiple valid supporting faces.
    equivalent(float(np.array(cut['gradient'])@point+cut['intercept']),float(value[0]))
    nonunique=not np.allclose(canonical[0]['gradient'],cut['gradient'],rtol=1e-9,atol=1e-10)
    return {'valid_lower_face':True,'independent_tangency_passed':True,'canonical_gradient_differs':bool(nonunique),
            'valid_active_subsets':[list(s) for s in valid]}
