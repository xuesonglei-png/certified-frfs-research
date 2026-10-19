import numpy as np
import pytest
from ensemble_vote_objective_v1 import SEEDS,build_ensemble,ensemble_objective
from independent_ensemble_vote_v1 import temperature,objective
from crossgroup_soft5_terms import crossgroup_terms
from density_calibrated_soft5 import calibrate,objective as own_mass
from soft_plurality_objective_v1 import plurality_objective
from test_dynamic_certificate_metric_v1 import table


def test_single_partition_equals_frozen_original_for_both_losses():
    A,y=table();w=np.array([.44,.56]);terms,tau,records=build_ensemble(A,y,w,seeds=SEEDS[:1])
    old,group=crossgroup_terms(A,y,eps=0.)
    for a,b in zip(terms[0],old,strict=True):np.testing.assert_array_equal(a,b)
    assert records[0]['partition']['query_indices']==group['query_indices']
    assert tau[0]==calibrate(old,w)[0]
    for family in ('own_mass','plurality'):
        actual,gradient=ensemble_objective(w,terms,y,tau,family=family)
        expected,grad=own_mass(w,old,tau[0]) if family=='own_mass' else plurality_objective(w,old,y,tau[0])
        np.testing.assert_allclose(actual,expected,atol=1e-13)
        np.testing.assert_allclose(gradient,grad,atol=1e-11)


def test_three_partitions_scalar_calibration_gradients_and_finite_difference():
    A,y=table();w=np.array([.44,.56]);terms,taus,records=build_ensemble(A,y,w)
    assert len({tuple(r['partition']['group_assignments']) for r in records})==3
    for seed,tau in zip(SEEDS,taus,strict=True):np.testing.assert_allclose(temperature(A,y,w,seed),tau,rtol=1e-10,atol=1e-13)
    for family in ('own_mass','plurality'):
        value,gradient=ensemble_objective(w,terms,y,taus,family=family)
        scalar,grad=objective(A,y,w,taus,SEEDS,family=family)
        np.testing.assert_allclose(value,scalar,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(gradient,grad,atol=2e-7,rtol=2e-6)
        direction=np.array([1.,-1.]);h=1e-7
        numeric=(ensemble_objective(w+h*direction,terms,y,taus,family=family)[0]-ensemble_objective(w-h*direction,terms,y,taus,family=family)[0])/(2*h)
        np.testing.assert_allclose(numeric,gradient@direction,atol=3e-6,rtol=3e-5)


def test_reg_not_tripled_and_input_validation():
    A,y=table();w=np.array([.44,.56]);terms,taus,_=build_ensemble(A,y,w)
    for family in ('own_mass','plurality'):
        v,g=ensemble_objective(w,terms,y,taus,family=family)
        bare,bg=ensemble_objective(w,terms,y,taus,family=family,reg=0.)
        log=np.log(len(w)*(w+1e-12))
        np.testing.assert_allclose(v-bare,.01*np.sum(w*log),atol=1e-13)
        np.testing.assert_allclose(g-bg,.01*(log+w/(w+1e-12)),atol=1e-11)
    with pytest.raises(ValueError):build_ensemble(A,y,w,seeds=())
    with pytest.raises(ValueError):build_ensemble(A,y,w,seeds=(1,1))
