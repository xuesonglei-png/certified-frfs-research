import copy
from collections import Counter
import numpy as np
import pytest
from plurality_certificate_metric_v1 import fit_plurality_certificate
from audit_plurality_certificate_metric_v1 import check_plurality_certificate_fit
from test_dynamic_certificate_metric_v1 import table


def test_equal_calibration_vote_objectives_and_full_reference_certificate():
    A,y=table();full=np.ones(2)/np.sqrt(2);temperatures=[]
    for family in ('own_mass','plurality'):
        for refresh in (False,True):
            w,fit=fit_plurality_certificate(A,y,[0,1],full,full,full,
                objective_family=family,refresh_frs=refresh,max_steps=3,lp_time_limit=3.)
            assert fit['available'];check_plurality_certificate_fit(A,y,fit,Counter())
            assert fit['final_objective']<=fit['initial_objective']
            temperatures.append(fit['temperature'])
            tamper=copy.deepcopy(fit);tamper['temperature']*=2
            with pytest.raises(AssertionError):check_plurality_certificate_fit(A,y,tamper,Counter())
    assert len(set(temperatures))==1


def test_same_temperature_zero_face_and_nominal_objective_only():
    A,y=table();full=np.ones(2)/np.sqrt(2);seed=np.array([np.sqrt(2),0.])
    for family in ('own_mass','plurality'):
        w,fit=fit_plurality_certificate(A,y,[0,1],seed,full,seed,
            objective_family=family,fixed_zeros=[1],max_steps=3,lp_time_limit=3.)
        assert fit['available'] and w[1]==0.
        check_plurality_certificate_fit(A,y,fit,Counter())
    with pytest.raises(ValueError):fit_plurality_certificate(A,y,[0,1],full,full,full,optimization_eps=.01)
