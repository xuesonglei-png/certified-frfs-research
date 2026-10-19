import copy
from collections import Counter
import numpy as np
import pytest
from joint_certificate_refresh_metric_v1 import fit_dynamic_certificate
from audit_joint_certificate_refresh_metric_v1 import check_joint_refresh_fit
from test_dynamic_certificate_metric_v1 import table


def test_original_full_reference_and_independent_rebuilt_cell_audit():
    A,y=table();anchor=np.ones(2)/np.sqrt(2)
    for refresh_frs in (False,True):
        point,fit=fit_dynamic_certificate(A,y,[0,1],anchor,anchor,anchor,
            refresh=True,refresh_frs=refresh_frs,max_steps=4,lp_time_limit=3.)
        assert fit['available'];counts=Counter();check_joint_refresh_fit(A,y,fit,counts)
        assert fit['final_objective']<=fit['initial_objective']
        for r in fit['class_cell_history']:
            np.testing.assert_array_equal(r['class_cell']['full_weights'],anchor)
            assert r['class_cell']['eta']==.1
        if refresh_frs:
            assert len(fit['class_cell_history'])>1
            corrupted=copy.deepcopy(fit);corrupted['class_cell_history'][1]['class_cell']['full_weights'][0]+=.01
            with pytest.raises(AssertionError):check_joint_refresh_fit(A,y,corrupted,Counter())


def test_rebuilt_cells_cannot_revive_a_deleted_coordinate():
    A,y=table();full=np.ones(2)/np.sqrt(2);seed=np.array([np.sqrt(2),0.])
    point,fit=fit_dynamic_certificate(A,y,[0,1],seed,full,seed,fixed_zeros=[1],
        refresh=True,refresh_frs=True,max_steps=3,lp_time_limit=3.)
    assert fit['available'] and point[1]==0.
    check_joint_refresh_fit(A,y,fit,Counter())
