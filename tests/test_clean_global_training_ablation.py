import copy
from collections import Counter
import numpy as np
import pytest
from fit_sparse_global_transition_frs import fit_sparse_global_transition_target
from clean_global_frs_training_ablation import fit_block,STARTS
from audit_clean_global_training_block import check_block


def test_same_three_starts_budget_certificate_and_both_losses_independently_checked():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    y=np.repeat([0,1],6);anchor=np.ones(2)/np.sqrt(2)
    starts={'anchor':anchor.tolist(),'common_capped':anchor.tolist(),'common_zero':[0.,np.sqrt(2)]};fits={}
    for start in STARTS:
        point,fit=fit_sparse_global_transition_target(A,y,[0,1],anchor,full_reference=anchor,
            reference_scope='original_full',initialization=np.asarray(starts[start]),initialization_scope=start)
        fits[start]={'target':point.tolist(),'fit':fit}
    selected=min(STARTS,key=lambda s:(fits[s]['fit']['training_objective_final'],STARTS.index(s)))
    old={'available':True,'fits':fits,'selected_start':selected,'target':fits[selected]['target'],'optimizer_success':True}
    source={'features':[0,1],'anchor':anchor.tolist(),'starts':starts,'arms':{'global_signed_sparse':old},
        'global_probe':fits['anchor']['fit']['class_cell']}
    block=fit_block(A,y,source);counts=Counter();check_block(A,y,source,block,counts)
    assert counts['matched_nominal_robust_training_folds']==1
    assert len(block['nominal']['fits'])==3
    broken=copy.deepcopy(block);broken['nominal']['fits']['common_zero']['fit']['certificate_eps']=0.
    with pytest.raises(AssertionError):
        check_block(A,y,source,broken,Counter())


def test_unavailable_original_region_stays_explicit_without_fake_fit():
    source={'features':[0,1],'anchor':[.5,.5],'starts':{s:[.5,.5] for s in STARTS},
        'arms':{'global_signed_sparse':{'available':False,'fits':{},'selected_start':None,'target':[.5,.5],'optimizer_success':None}}}
    A=np.zeros((12,2));y=np.repeat([0,1],6);block=fit_block(A,y,source)
    assert block['nominal']['fits']=={} and block['evaluation'] is None
    counts=Counter();check_block(A,y,source,block,counts)
    assert counts['unavailable_original_global_regions']==1
