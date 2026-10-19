from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_graph_dependency_v4 import class_upper, memberships


def test_old_presolve_discrepancy_regression_matches_attained_counterexample():
    rng=np.random.default_rng(839);A=rng.uniform(.2,.8,(5,2));y=np.array([0,0,1,1,1])
    full,reduced=rng.uniform(.1,1.2,(2,2))
    relaxed=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.)
    exact=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.,exact_abs=True)
    assert relaxed['solver_upper_bound']+1e-7>=exact['observed_target_loss_lower']
    assert relaxed['solver_upper_bound']>=.0228753589-1e-7
    assert relaxed['backend_record']['settings']['presolve']=='off'
    assert relaxed['backend_record']['matrix_roundtrip_max_difference']<=1e-12


def test_both_targets_against_actual_tables_and_exact_sign_upper():
    rng=np.random.default_rng(8239)
    for positive in (False,True):
        for _ in range(3):
            A=rng.uniform(.2,.8,(5,2));y=np.array([0,0,1,1,1])
            full,reduced=rng.uniform(.1,1.2,(2,2))
            relaxed=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.,positive=positive)
            exact=class_upper(A,y,full,reduced,0,eps=.08,seconds=10.,positive=positive,exact_abs=True)
            assert relaxed['solver_upper_bound']+1e-7>=exact['solver_upper_bound']
            for _ in range(25):
                Z=rng.uniform(A-.08,A+.08)
                delta=(memberships(Z,y,full)-memberships(Z,y,reduced))[:2]
                actual=(np.maximum(0.,delta) if positive else delta).mean()
                assert actual<=relaxed['solver_upper_bound']+1e-7
