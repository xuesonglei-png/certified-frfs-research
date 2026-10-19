"""Independent actual-coordinate convex target, polytope, and deployment."""
import numpy as np
from majority_hinge_canonical_stage import GRID
from audit_majority_hinge_fit_cached import check_fit
from audit_predictive_target_knn_pilot import check_stage as old_check_stage
from audit_forest_compact_transfer import equivalent

WEIGHTED={'full','proposed','fgs_matched','sfrip_matched','unguarded_pool','raw_repair_pool'}


def check_stage(A,y,block,counts):
    old_check_stage(A,y,{'learners':[],'certificate_checks':block['certificate_checks']},counts);seen=set()
    for r in block['learners']:
        key=(tuple(r['features']),tuple(r['anchor']),r['family']);assert key not in seen;seen.add(key)
        assert r['family'] in ('majority_hinge_unconstrained','majority_hinge_constrained')
        assert r['fit']['constrained']==(r['family']=='majority_hinge_constrained');check_fit(A,y,r,counts)


def check_metric(metric,base,block,method,target_index,counts):
    assert metric['features']==base['features'] and metric['base_origin']==base['origin']
    assert metric['target_index']==target_index;anchor=np.array(base['weights'])
    if method not in WEIGHTED:
        assert target_index==0 and metric['learner_id'] is metric['guard_id'] is metric['deployed_guard_id'] is None
        assert not metric['guard_fallback'] and metric['interpolation']==0 and metric['target_family'] is None
        equivalent(anchor.tolist(),metric['weights']);return
    family,t=GRID[target_index];learner=block['learners'][metric['learner_id']]
    assert learner['features']==base['features'];equivalent(learner['anchor'],base['weights'])
    assert metric['interpolation']==t and metric['target_family']==learner['family']==family
    requested=(1-t)*anchor+t*np.array(learner['target']);equivalent(requested.tolist(),metric['requested_weights'])
    if method=='proposed':
        check=block['certificate_checks'][metric['guard_id']];equivalent(check['weights'],requested.tolist())
        assert check['features']==base['features'] and metric['guard_fallback']==(not check['accepted'])
        deployed=block['certificate_checks'][metric['deployed_guard_id']];assert deployed['accepted'] and deployed['features']==base['features']
        actual=anchor if metric['guard_fallback'] else requested;equivalent(actual.tolist(),deployed['weights'])
        equivalent(actual.tolist(),metric['weights']);counts['proposed_fallback_options']+=metric['guard_fallback']
    else:
        assert metric['guard_id'] is metric['deployed_guard_id'] is None and not metric['guard_fallback']
        equivalent(requested.tolist(),metric['weights'])
