"""Matched nominal versus robust loss; original robust constraints unchanged."""
import time
import numpy as np
from fit_clean_global_transition_frs import fit_clean_global_transition_target
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from frs_global_saturation_transition import evaluate_class_cell

ARMS=('robust_training','nominal_training')
STARTS=('anchor','common_capped','common_zero')


def control_record(source):
    old=source['arms']['global_signed_sparse']
    return {'available':old['available'],'selected_start':old['selected_start'],
        'target':old['target'],'optimizer_success':old['optimizer_success'],
        'source_arm':'global_signed_sparse','source_fits_reused_without_refitting':True,
        'training_start_objectives':{s:r['fit']['training_objective_final'] for s,r in old['fits'].items()},
        'optimizer_failures':sum(r['fit']['subproblem_failures'] for r in old['fits'].values())}


def evaluate(A,y,source,block):
    if not block['control']['available']:
        return None
    fs,anchor=source['features'],np.asarray(source['anchor']);mass=np.sqrt(A.shape[1])
    losses={eps:CappedThirdEnemyHinge(A[:,fs],y,anchor/mass,eps=eps) for eps in (0.,.01)}
    points={ARMS[0]:np.asarray(block['control']['target']),ARMS[1]:np.asarray(block['nominal']['target'])}
    risks={eps:{arm:float(loss.alpha@loss.losses(point/mass)) for arm,point in points.items()} for eps,loss in losses.items()}
    classes={arm:evaluate_class_cell(source['global_probe'],point)['class_upper_bounds'] for arm,point in points.items()}
    assert all(evaluate_class_cell(source['global_probe'],p)['feasible'] for p in points.values())
    return {'nominal_training_bounds':risks[0.],'robust_training_bounds':risks[.01],
        'nominal_bound_decrease_pp':100*(risks[0.][ARMS[0]]-risks[0.][ARMS[1]]),
        'robust_bound_decrease_pp':100*(risks[.01][ARMS[0]]-risks[.01][ARMS[1]]),
        'exact_zero_weights':{a:int(sum(p==0.)) for a,p in points.items()},
        'certified_class_bounds':classes,
        'optimizer_failures':{ARMS[0]:block['control']['optimizer_failures'],
            ARMS[1]:sum(r['fit']['subproblem_failures'] for r in block['nominal']['fits'].values())},
        'classification_scored':False}


def fit_block(A,y,source):
    assert tuple(source['starts'])==STARTS
    control=control_record(source)
    block={'features':source['features'],'anchor':source['anchor'],'starts':source['starts'],
        'control':control,'nominal':None,'evaluation':None}
    if not control['available']:
        block['nominal']={'available':False,'fits':{},'selected_start':None,
            'target':source['anchor'],'optimizer_success':None,'scope':'original_region_unavailable_no_search'}
        return block
    fs,anchor=source['features'],np.asarray(source['anchor']);full=np.ones(A.shape[1])/np.sqrt(A.shape[1])
    fits={}
    for start in STARTS:
        timer=time.perf_counter()
        point,fit=fit_clean_global_transition_target(A,y,fs,anchor,full_reference=full,
            reference_scope='original_full',initialization=np.asarray(source['starts'][start]),
            initialization_scope=start,optimization_eps=0.,sparsity_lambda=.05,sparsity_delta=.01)
        assert fit['class_cell']==source['global_probe']
        fits[start]={'features':fs,'anchor':anchor.tolist(),'target':point.tolist(),'fit':fit,
            'fit_seconds':time.perf_counter()-timer}
    selected=min(STARTS,key=lambda s:(fits[s]['fit']['training_objective_final'],STARTS.index(s)))
    block['nominal']={'available':True,'fits':fits,'selected_start':selected,'target':fits[selected]['target'],
        'optimizer_success':all(r['fit']['optimizer_success'] for r in fits.values())}
    block['evaluation']=evaluate(A,y,source,block)
    return block
