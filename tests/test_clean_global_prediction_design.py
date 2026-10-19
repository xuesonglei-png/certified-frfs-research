import copy
from collections import Counter
import numpy as np
import pytest
from clean_global_prediction_design import metrics_from_training
from audit_clean_global_prediction_design import reconstruct_metrics
from geometric_vote_evaluation import final_evaluation
from independent_geometric_evaluation import replay_final
from audit_forest_compact_transfer import equivalent


def fixture():
    anchor=[1/np.sqrt(2)]*2
    training={'complete':True,'fold':0,'d':2,'reference_indices':list(range(20)),
        'training_indices':list(range(12)),'block':{'features':[0,1],'anchor':anchor,
            'control':{'target':anchor},'nominal':{'target':[0.,np.sqrt(2)]}}}
    options={};selection={}
    for m,width in {'fgs_native120':120,'fgs_weighted_native120':1680,'sfrip_native6':6,'sfrip_weighted_native6':84}.items():
        options[m]=[{'metric':{'features':[0,1],'weights':anchor}} for i in range(width)]
        selection[m]={'selected_index':width-1,'options':[{'config_index':i,'target_index':0} for i in range(width)]}
    native={'complete':True,'reference_indices':training['reference_indices'],'selection':selection,
        'inner':[{'reference_indices':training['training_indices'],'validation_indices':list(range(12,20)),
            'bases':{'proposed':[{'features':[0,1],'weights':anchor}]},'options':options}]}
    return training,native


def test_exact_zero_support_native_full_budget_and_train_validation_alignment():
    training,native=fixture();metrics,choices=metrics_from_training(training,native)
    separate,check=reconstruct_metrics(training,native);equivalent(metrics,separate);equivalent(choices,check)
    assert metrics['nominal_training']=={'features':[1],'weights':[np.sqrt(2)]}
    assert choices['fgs_weighted_native120']['candidate_count']==1680
    broken=copy.deepcopy(native);broken['inner'][0]['validation_indices']=[0]
    with pytest.raises(AssertionError):
        metrics_from_training(training,broken)
    missing=copy.deepcopy(training);missing['block']=None
    left,_=metrics_from_training(missing,native);right,_=reconstruct_metrics(missing,native);equivalent(left,right)
    assert left['nominal_training']==left['anchor']


def test_both_training_endpoints_have_same_independent_query_and16noise_evaluation():
    training,native=fixture();metrics,_=metrics_from_training(training,native)
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)]);y=np.repeat([0,1],6)
    B=np.array([[.12,0.],[.13,1.]]);truth=np.array([0,1]);counts=Counter()
    for method in ('robust_training','nominal_training'):
        packed=final_evaluation(A,y,B,truth,metrics[method])
        replay_final(A,y,B,truth,{'metric':metrics[method],**packed},counts)
    assert counts['independent_noise_predictions']==32
