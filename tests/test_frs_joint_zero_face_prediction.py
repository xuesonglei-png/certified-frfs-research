import copy
from collections import Counter
import numpy as np
import pytest
from run_frs_joint_zero_face_prediction import metrics_from_training,REGIMES
from audit_frs_joint_zero_face_prediction import reconstruct_metrics
from geometric_vote_evaluation import final_evaluation
from independent_geometric_evaluation import replay_final
from audit_forest_compact_transfer import equivalent


def fixture():
    joint={'complete':True,'fold':0,'reference_indices':list(range(20)), 'training_indices':list(range(12)),
        'available':True,'d':3,'original_support':[0,1,2],'original_anchor':[1/np.sqrt(3)]*3,
        'eliminations':{m:{'target':[0.,0.,np.sqrt(3)]} for m in REGIMES}}
    options={};selection={}
    for m,width in {'fgs_native120':120,'fgs_weighted_native120':1680,'sfrip_native6':6,'sfrip_weighted_native6':84}.items():
        options[m]=[{'metric':{'features':[2],'weights':[1.]}} for i in range(width)]
        selection[m]={'selected_index':width-1,'options':[{'config_index':i,'target_index':0} for i in range(width)]}
    native={'complete':True,'reference_indices':joint['reference_indices'],'selection':selection,
        'inner':[{'reference_indices':joint['training_indices'],'validation_indices':list(range(12,20)),
            'bases':{'proposed':[{'features':joint['original_support'],'weights':joint['original_anchor']}]},'options':options}]}
    return joint,native


def test_export_preserves_exact_positive_support_full_native_budget_and_alignment():
    joint,native=fixture();metrics,choices=metrics_from_training(joint,native)
    independent,check=reconstruct_metrics(joint,native);equivalent(metrics,independent);equivalent(choices,check)
    assert metrics['joint_pair_transition']=={'features':[2],'weights':[np.sqrt(3)]}
    assert choices['fgs_weighted_native120']['candidate_count']==1680
    broken=copy.deepcopy(native);broken['inner'][0]['validation_indices']=[0]
    with pytest.raises(AssertionError):
        metrics_from_training(joint,broken)


def test_exported_joint_support_replays_clean_query_and_all16_noise_predictions():
    A=np.array([[x,x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)])
    y=np.repeat([0,1],6);B=np.array([[.15,.12,0.],[.12,.15,1.]]);truth=np.array([0,1])
    metric={'features':[2],'weights':[np.sqrt(3)]}
    packed=final_evaluation(A,y,B,truth,metric);counts=Counter()
    replay_final(A,y,B,truth,{'metric':metric,**packed},counts)
    assert counts['independent_noise_predictions']==16
    full={'features':[0,1,2],'weights':[0.,0.,np.sqrt(3)]}
    padded=final_evaluation(A,y,B,truth,full)
    equivalent(padded['evaluation'],packed['evaluation'])
    for left,right in zip(padded['perturbation_evaluations'],packed['perturbation_evaluations'],strict=True):
        equivalent(left,right)
