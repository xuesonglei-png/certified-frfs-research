import copy
from collections import Counter
import numpy as np
import pytest
from plurality_certificate_metric_v1 import fit_plurality_certificate
from ensemble_vote_certificate_metric_v1 import fit_ensemble_vote_certificate
from audit_ensemble_vote_certificate_metric_v1 import check_ensemble_vote_certificate_fit
from ensemble_vote_objective_v1 import SEEDS
from test_dynamic_certificate_metric_v1 import table


def test_single_partition_reproduces_original_optimizer_and_three_are_audited():
    A,y=table();full=np.ones(2)/np.sqrt(2)
    for family in ('own_mass','plurality'):
        old,baseline=fit_plurality_certificate(A,y,[0,1],full,full,full,
            objective_family=family,max_steps=3,lp_time_limit=3.)
        for seeds in (SEEDS[:1],SEEDS):
            w,fit=fit_ensemble_vote_certificate(A,y,[0,1],full,full,full,
                objective_family=family,partition_seeds=seeds,max_steps=3,lp_time_limit=3.)
            assert fit['available']
            check_ensemble_vote_certificate_fit(A,y,fit,Counter())
            if len(seeds)==1:
                np.testing.assert_allclose(w,old,atol=1e-9,rtol=0)
                np.testing.assert_allclose(fit['final_objective'],baseline['final_objective'],atol=1e-10)
            tamper=copy.deepcopy(fit);tamper['partition_temperatures'][-1]*=2
            with pytest.raises(AssertionError):check_ensemble_vote_certificate_fit(A,y,tamper,Counter())


def test_ensemble_keeps_exact_zeros_and_rejects_reference_mutation():
    A,y=table();full=np.ones(2)/np.sqrt(2);seed=np.array([np.sqrt(2),0.])
    w,fit=fit_ensemble_vote_certificate(A,y,[0,1],seed,full,seed,
        fixed_zeros=[1],max_steps=3,lp_time_limit=3.)
    assert fit['available'] and w[1]==0.
    check_ensemble_vote_certificate_fit(A,y,fit,Counter())
    tamper=copy.deepcopy(fit);tamper['full_reference'][0]+=.1
    with pytest.raises(AssertionError):check_ensemble_vote_certificate_fit(A,y,tamper,Counter())
