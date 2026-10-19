import numpy as np
from certified_frfs.nested_triangular_stage import fit_stage
from certified_frfs.triangular_classifier import triangular_box_margins

def test_complete_fuzzy_stage_outputs_actual_deployed_protection():
    X=np.array([[0.,0.],[.1,.1],[.2,.2],[.8,.8],[.9,.9],[1.,1.]])
    y=np.array([0,0,0,1,1,1]);stage=fit_stage(X,y,C=1)
    original=triangular_box_margins(X,y,np.array(stage['teacher_coef']),np.array(stage['teacher_intercept']),stage['teacher_classes'])
    for policy in stage['policies'].values():
        assert policy['candidate']['certificate']['feasible']
        fit=policy['fit'];m=triangular_box_margins(X,y,np.array(fit['coef']),np.array(fit['intercept']),stage['teacher_classes'])
        assert np.all(m[original>1e-10]>=.5*original[original>1e-10]-1e-8)
