from collections import Counter
import copy
import numpy as np
import pytest
from fit_latent_frs_hinge import fit_latent_frs_target
from audit_latent_frs_fit import check_latent_fit


def test_latent_majorization_keeps_FRS_bounds_descent_and_dual_replay():
    rng=np.random.default_rng(781);A=rng.uniform(0,1,(24,3));y=np.repeat([0,1],12);anchor=np.ones(3)/np.sqrt(3)
    target,fit=fit_latent_frs_target(A,y,[0,1,2],anchor,full_reference=anchor,reference_scope='original_full')
    assert fit['optimizer_success'],fit['status']
    assert fit['class_cell_final']['feasible'] and fit['training_objective_final']<=fit['training_objective_initial']+1e-10
    counts=Counter();record={'features':[0,1,2],'anchor':anchor.tolist(),'target':target.tolist(),'fit':fit}
    check_latent_fit(A,y,record,counts)
    assert counts['independent_master_primal_dual_replays']>=1
    for step in fit['steps']:
        assert len(step['triple_choices'])==len(A)
    altered=copy.deepcopy(record);altered['fit']['steps'][0]['triple_choices'][0]=100
    with pytest.raises((AssertionError,IndexError)):
        check_latent_fit(A,y,altered,Counter())


def test_switching_triples_filters_cached_planes_and_full_LP_audit_passes():
    # Synthetic seed chosen for branch coverage, not a benchmark result.
    rng=np.random.default_rng(4);A=rng.uniform(0,1,(30,4));y=np.repeat([0,1],15);anchor=np.ones(4)/2
    target,fit=fit_latent_frs_target(A,y,[0,1,2,3],anchor,full_reference=anchor,reference_scope='original_full')
    assert fit['optimizer_success'],fit['status']
    assert any(a['triple_choices']!=b['triple_choices'] for a,b in zip(fit['steps'],fit['steps'][1:]))
    assert len({(c['query'],c['triple_choice']) for c in fit['cuts']})>len(A)
    counts=Counter()
    check_latent_fit(A,y,{'features':[0,1,2,3],'anchor':anchor.tolist(),'target':target.tolist(),'fit':fit},counts)
    assert counts['independent_master_primal_dual_replays']>=2
