"""Training-only illustration, against explicitly derived soft leakage bounds.

These are NOT all previous FRS bounds and NOT held-out classification results.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from soft_topk_metric_objective import soft_topk_membership,topk_leakage_bound
from robust_majority_hinge import RobustMajorityHinge


def compare(A,y,w,*,temperature=.05,margin=.02):
    A,y,w=np.asarray(A,float),np.asarray(y),np.asarray(w,float)
    hinge=RobustMajorityHinge(A,y,w,margin=margin);loss,_=hinge.oracle(w)
    lo,hi=np.maximum(0,A-.01),np.minimum(1,A+.01);rows=[]
    for i,row in enumerate(hinge.rows):
        refs=np.array(row['references']);same=y[refs]==y[i]
        if len(refs)<5:
            rows.append({'query':i,'hinge_error_bound':min(1,float(loss[i])),
                         'soft_gap_error_bound':1.,'soft_exact_leakage_error_bound':1.,'independent_scalar_soft_replayed':True});continue
        lower=np.maximum(0,np.maximum(lo[refs]-hi[i],lo[i]-hi[refs]))
        upper=np.maximum(abs(lo[i]-hi[refs]),abs(hi[i]-lo[refs]))
        D=np.where(same[:,None],upper,lower)@w
        Q=soft_topk_membership(D[None,:],np.ones((1,len(refs)),bool),temperature=temperature)[0]
        # Separate scalar root replay, no production threshold or derivatives.
        if len(refs)==5:scalar=np.ones(5)
        else:
            radius=brentq(lambda t:float(expit((t-D)/temperature).sum()-5),float(D.min()-2),float(D.max()+2),xtol=1e-13)
            scalar=expit((radius-D)/temperature)
        np.testing.assert_allclose(Q,scalar,atol=1e-9,rtol=0)
        order=np.argsort(D,kind='stable');outside=order[5:]
        exact=float(Q[outside].sum());gap=float(D[order[5]]-D[order[4]]) if len(refs)>5 else 0.
        gap_leakage=topk_leakage_bound(gap,len(refs),temperature=temperature)
        assert exact<=gap_leakage+1e-8
        own=float(Q[same].sum()/5)
        # Wrong 5NN plurality => at most two true-class neighbors. Therefore
        # error <= (5/3)*(1-hard_own) <= (5/3)*(1-soft_own+leakage/5).
        rows.append({'query':i,'hinge_error_bound':min(1,float(loss[i])),
            'soft_gap_error_bound':min(1.,max(0.,(5/3)*(1-own+gap_leakage/5))),
            'soft_exact_leakage_error_bound':min(1.,max(0.,(5/3)*(1-own+exact/5))),
            'pessimistic_soft_own':own,'pessimistic_hard5_6_gap':gap,
            'gap_leakage_mass_bound':gap_leakage,'exact_pessimistic_leakage_mass':exact,
            'independent_scalar_soft_replayed':True})
    means={k:float(hinge.alpha@np.array([r[k] for r in rows])) for k in
           ('hinge_error_bound','soft_gap_error_bound','soft_exact_leakage_error_bound')}
    return {'rows':rows,'class_balanced_training_bounds':means,
        'scope':'same_finite_crossgroup_training_table_semantic_witness_hinge_vs_derived_soft_leakage_bounds_not_other_FRS_bounds_or_test_risk'}
