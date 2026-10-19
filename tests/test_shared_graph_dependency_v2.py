from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_graph_dependency_v1 import class_upper as old_upper
from shared_graph_dependency_v2 import class_upper, memberships
from signed_classwise_dependency import signed_classwise_dependency


def enemy_cuts(A,y,wf,wr,eps):
    fs = np.flatnonzero(wr)
    report = signed_classwise_dependency(A,y,fs,wf,wr[fs],eps=eps)
    result = {}
    lo,hi = np.maximum(0.,A-eps),np.minimum(1.,A+eps)
    for item in report['signed_object_report']['objects']:
        i = item['object']; candidates = item['possible_enemies']
        lb = np.maximum(0.,np.maximum(lo[i]-hi[candidates],lo[candidates]-hi[i]))@wr
        result[i] = {j: min(float(1.-l), float(u)) for j,l,u in
            zip(candidates,lb,item['candidate_signed_upper_bounds'],strict=True)}
    return report['signed_object_report']['signed_object_upper_bounds'],result


def test_rectangle_roof_removes_artificial_distance_at_same_corner():
    from shared_graph_dependency_v2 import build
    A = np.array([[.5],[.5]]); y = np.array([0,1])
    m,z,meta = build(A,y,[1.],[.5],0,eps=.5)
    matrix,rhs,cost = m.arrays()
    # q is the first variable after z, t and r; find its lower-epigraph row.
    from scipy.optimize import milp,Bounds,LinearConstraint
    low,high = np.asarray(m.lower),np.asarray(m.upper)
    low[z.ravel()]=0.;high[z.ravel()]=0.
    fit = milp(cost,integrality=m.integer,bounds=Bounds(low,high),
               constraints=LinearConstraint(matrix,-np.inf,rhs))
    assert fit.status==0 and abs(fit.fun)<1e-10


def test_conditional_cuts_are_safe_and_bounds_not_weaker_at_optimality():
    rng = np.random.default_rng(337)
    for _ in range(4):
        A = rng.uniform(.2,.8,(5,2)); y = np.array([0,0,1,1,1])
        wf,wr = rng.uniform(.1,1.,(2,2)); eps = .07
        objects,cuts = enemy_cuts(A,y,wf,wr,eps)
        previous = old_upper(A,y,wf,wr,0,eps=eps,seconds=10.,object_upper=objects)
        new = class_upper(A,y,wf,wr,0,eps=eps,seconds=10.,object_upper=objects,enemy_upper=cuts)
        assert new['solver_status']==0 and previous['solver_status']==0
        assert new['solver_upper_bound'] <= previous['solver_upper_bound']+1e-7
        exact = class_upper(A,y,wf,wr,0,eps=eps,seconds=10.,exact_abs=True)
        assert new['solver_upper_bound']+1e-7 >= exact['solver_upper_bound']
        for _ in range(30):
            Z = rng.uniform(A-eps,A+eps)
            actual = (memberships(Z,y,wf)-memberships(Z,y,wr))[:2].mean()
            assert actual <= new['solver_upper_bound']+1e-7
