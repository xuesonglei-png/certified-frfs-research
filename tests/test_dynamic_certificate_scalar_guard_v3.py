from collections import Counter
import numpy as np
from dynamic_certificate_metric_v3 import fit_dynamic_certificate
from audit_dynamic_certificate_metric_v1 import check_dynamic_fit
from independent_frs_pair_transition_cell import scalar_cell_evaluation
from test_dynamic_certificate_metric_v1 import table


def test_guarded_real_optimization_replays_without_tolerance_change():
    A,y=table();anchor=np.ones(2)/np.sqrt(2)
    for refresh in (False,True):
        for eps in (0.,.01):
            point,fit=fit_dynamic_certificate(A,y,[0,1],anchor,anchor,anchor,
                refresh=refresh,optimization_eps=eps,max_steps=3,lp_time_limit=3.)
            assert fit['available'] and scalar_cell_evaluation(fit['class_cell'],point)['feasible']
            check_dynamic_fit(A,y,fit,Counter())
            assert fit['eta']==.1 and fit['certificate_eps']==.01 and fit['retention']==.9
            for step in fit['steps']:
                if step['lp_success']:
                    assert scalar_cell_evaluation(fit['class_cell'],step['vertex'])['feasible']
