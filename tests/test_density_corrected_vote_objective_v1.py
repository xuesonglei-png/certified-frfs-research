from itertools import combinations
from fractions import Fraction
import numpy as np
import pytest
from density_corrected_vote_objective_v1 import balanced_weighted_membership,build_density_corrected,objective
from independent_density_corrected_vote_v1 import scalar_membership,calibrated_temperature,objective as independent_objective
from density_calibrated_soft5 import balanced_membership
from ensemble_vote_objective_v1 import SEEDS
from test_dynamic_certificate_metric_v1 import table


def test_unit_coverage_is_original_and_weighted_threshold_independent():
    rng=np.random.default_rng(719);D=rng.uniform(0,1,(8,20));mask=np.ones(D.shape,bool)
    for tau in (.0005,.002,.05):
        Q,S=balanced_weighted_membership(D,mask,mask.astype(float),tau,return_slope=True)
        old,oldS=balanced_membership(D,mask,tau,return_slope=True)
        np.testing.assert_allclose(Q,old,atol=1e-10);np.testing.assert_allclose(S,oldS,atol=1e-10)
        R=rng.uniform(1,2,(8,20));Q,S=balanced_weighted_membership(D,mask,R,tau,return_slope=True)
        np.testing.assert_allclose((Q*R).sum(axis=1),5,atol=1e-9)
        for i in range(len(D)):
            scalar,slopes=scalar_membership(D[i],R[i],tau)
            np.testing.assert_allclose(Q[i],scalar,atol=1e-9,rtol=1e-8)
            np.testing.assert_allclose(S[i],slopes,atol=1e-9,rtol=1e-8)


def test_covered_gradients_and_common_temperature_for_both_losses():
    A,y=table();w=np.array([.44,.56]);parts,R,taus,records=build_density_corrected(A,y,w)
    for seed,tau in zip(SEEDS,taus,strict=True):np.testing.assert_allclose(tau,calibrated_temperature(A,y,w,seed),atol=1e-13,rtol=1e-10)
    for family in ('own_mass','plurality'):
        for correction in (False,True):
            value,gradient=objective(w,parts,R,y,taus,family=family,correction=correction)
            independent,igrad=independent_objective(A,y,w,taus,SEEDS,family=family,correction=correction)
            np.testing.assert_allclose(value,independent,atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(gradient,igrad,atol=2e-7,rtol=2e-6)
            direction=np.array([1.,-1.]);h=1e-7
            numeric=(objective(w+h*direction,parts,R,y,taus,family=family,correction=correction)[0]-objective(w-h*direction,parts,R,y,taus,family=family,correction=correction)[0])/(2*h)
            np.testing.assert_allclose(numeric,gradient@direction,atol=3e-6,rtol=3e-5)


def test_missing_reference_class_explicit_and_no_unweighted_loss_claim():
    A,y=table();A=np.vstack((A,[[.2,.8]]));y=np.r_[y,19]
    parts,R,taus,records=build_density_corrected(A,y,np.array([.5,.5]))
    assert any(r['missing_reference_class_queries'] for r in records)
    for part,coverage,r in zip(parts,R,records,strict=True):
        assert not set(r['query_indices'])&set(r['missing_reference_class_queries'])
        np.testing.assert_allclose(coverage.sum(axis=1),len(y)-1,atol=1e-12)
    with pytest.raises(ValueError):balanced_weighted_membership(np.zeros((1,4)),np.ones((1,4),bool),np.ones((1,4)),.01)


def test_fixed_radius_inverse_coverage_expectation_and_finite_population_variance():
    # EXACT enumeration of a uniformly sampled class population at a GIVEN
    # radius: proves this small witness, never unbiasedness of the fitted root.
    population=[Fraction(1,10),Fraction(2,10),Fraction(5,10),Fraction(9,10)]
    N,m=len(population),2
    estimates=[Fraction(N,m)*sum((population[i] for i in ids),Fraction(0))/5 for ids in combinations(range(N),m)]
    expected=sum(population,Fraction(0))/5
    assert sum(estimates,Fraction(0))/len(estimates)==expected
    variance=sum(((v-expected)**2 for v in estimates),Fraction(0))/len(estimates)
    mean=sum(population,Fraction(0))/N
    sample_variance=sum(((v-mean)**2 for v in population),Fraction(0))/(N-1)
    assert variance==Fraction(N*N,m)*(1-Fraction(m,N))*sample_variance/25
