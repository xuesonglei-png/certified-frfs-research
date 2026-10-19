import copy
from collections import Counter
import numpy as np
import pytest
from ensemble_vote_certificate_metric_v1 import fit_ensemble_vote_certificate
from density_corrected_vote_certificate_metric_v1 import fit_density_corrected_vote_certificate
from audit_density_corrected_vote_certificate_metric_v1 import check_density_corrected_vote_certificate_fit
from test_dynamic_certificate_metric_v1 import table


def test_density_fit_and_matched_uniform_fit_independently_reconstructed():
    A,y=table();full=np.ones(2)/np.sqrt(2)
    for family in ('own_mass','plurality'):
        old,baseline=fit_ensemble_vote_certificate(A,y,[0,1],full,full,full,
            objective_family=family,max_steps=3,lp_time_limit=3.)
        for correction in (False,True):
            w,fit=fit_density_corrected_vote_certificate(A,y,[0,1],full,full,full,
                objective_family=family,correction=correction,max_steps=3,lp_time_limit=3.)
            assert fit['available'] and fit['final_objective']<=fit['initial_objective']
            counts=Counter();check_density_corrected_vote_certificate_fit(A,y,fit,counts)
            assert counts['independent_direction_primal_dual_replays']>0
            if not correction:np.testing.assert_allclose(w,old,atol=1e-8,rtol=0)
            bad=copy.deepcopy(fit);bad['partition_records'][0]['class_population'][0][0]+=1
            with pytest.raises(AssertionError):check_density_corrected_vote_certificate_fit(A,y,bad,Counter())


def test_density_objective_preserves_zero_face_and_original_full_reference():
    A,y=table();full=np.ones(2)/np.sqrt(2);seed=np.array([np.sqrt(2),0.])
    w,fit=fit_density_corrected_vote_certificate(A,y,[0,1],seed,full,seed,
        fixed_zeros=[1],max_steps=3,lp_time_limit=3.)
    assert fit['available'] and w[1]==0.
    check_density_corrected_vote_certificate_fit(A,y,fit,Counter())
    bad=copy.deepcopy(fit);bad['full_reference'][0]+=.1
    with pytest.raises(AssertionError):check_density_corrected_vote_certificate_fit(A,y,bad,Counter())
