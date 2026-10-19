import copy
from collections import Counter
import numpy as np
import pytest
from certified_path_training_guard import guard,choose,ARMS
from audit_certified_path_training_guard import check_guard


def test_harmful_training_deletion_rejected_and_good_one_audited():
    A=np.array([[x,c] for c in (0.,1.) for x in (0.,0.,.1,.1,.2,.2)]);y=np.repeat([0,1],6)
    mass=np.sqrt(2);joint={'original_support':[0,1],'original_anchor':[mass/2,mass/2],'available':True,
        'eliminations':{'pair_transition':{'steps':[
            {'accepted':True,'target':[mass,0.]},{'accepted':False,'target':None},{'accepted':True,'target':[0.,mass]}]}}}
    saved=guard(A,y,joint);counts=Counter();check_guard(A,y,joint,saved,counts)
    for arm in ARMS:
        assert saved['choices'][arm]['eligible_indices']==[0,2] and saved['choices'][arm]['selected_index']==2
        assert saved['metrics'][arm]['features']==[1]
    broken=copy.deepcopy(saved);broken['choices'][ARMS[0]]['selected_index']=1
    with pytest.raises(AssertionError):check_guard(A,y,joint,broken,Counter())
    assert counts['independent_training_path_guard_selections']==2


def test_minority_harm_and_certification_harm_cannot_hide_in_balanced_mean():
    candidates=[{'weights':[.5,.5]},{'weights':[0.,1.]},{'weights':[1.,0.]}]
    base={'classes':[0,1],'class_counts':[10,10],'class_correct_counts':[6,6],
        'class_certified_correct_counts':[4,4],'balanced_accuracy':.6,'certified_balanced_accuracy':.4}
    minority={**base,'class_correct_counts':[9,5],'balanced_accuracy':.7}
    uncertified={**base,'class_correct_counts':[7,6],'class_certified_correct_counts':[4,3],
        'balanced_accuracy':.65,'certified_balanced_accuracy':.35}
    assert choose(candidates,[base,minority,uncertified],ARMS[0])['selected_index']==2
    assert choose(candidates,[base,minority,uncertified],ARMS[1])['selected_index']==0
