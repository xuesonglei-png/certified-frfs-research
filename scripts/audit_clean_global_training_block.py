"""Independent matching, original box constraints, both losses and LP replays."""
import numpy as np
from audit_clean_global_transition_frs import check_clean_global_transition_fit,ScalarRankValidator
from independent_global_frs_transition import scalar_cell_evaluation
from audit_forest_compact_transfer import equivalent

STARTS=('anchor','common_capped','common_zero')
ARMS=('robust_training','nominal_training')


def check_block(A,y,source,saved,counts):
    old=source['arms']['global_signed_sparse'];control=saved['control'];nominal=saved['nominal']
    assert saved['features']==source['features'];equivalent(saved['anchor'],source['anchor'])
    equivalent(saved['starts'],source['starts']);assert tuple(saved['starts'])==STARTS
    expected={'available':old['available'],'selected_start':old['selected_start'],'target':old['target'],
        'optimizer_success':old['optimizer_success'],'source_arm':'global_signed_sparse',
        'source_fits_reused_without_refitting':True,
        'training_start_objectives':{s:r['fit']['training_objective_final'] for s,r in old['fits'].items()},
        'optimizer_failures':sum(r['fit']['subproblem_failures'] for r in old['fits'].values())}
    equivalent(control,expected);assert nominal['available']==old['available']
    if not old['available']:
        assert nominal['fits']=={} and nominal['selected_start'] is None and nominal['optimizer_success'] is None
        equivalent(nominal['target'],source['anchor']);assert saved['evaluation'] is None
        counts['unavailable_original_global_regions']+=1;return
    assert tuple(nominal['fits'])==STARTS and tuple(old['fits'])==STARTS
    for start,record in nominal['fits'].items():
        fit=record['fit'];previous=old['fits'][start]['fit']
        assert fit['optimization_eps']==0. and fit['training_witnesses']['eps']==0.
        assert previous['training_witnesses']['eps']==.01
        assert fit['certificate_eps']==fit['class_cell']['eps']==.01
        assert fit['class_cell']['eta']==.1 and fit['polytope']['retention']==.9
        for key in ('margin','reg','sparsity_lambda','sparsity_delta','max_major','max_iter','gap_tolerance','reference_scope','initialization_scope'):
            equivalent(fit[key],previous[key])
        equivalent(fit['requested_initial_weights'],source['starts'][start])
        equivalent(fit['class_cell'],source['global_probe'])
        equivalent(record['anchor'],source['anchor']);assert record['features']==source['features']
        check_clean_global_transition_fit(A,y,record,counts)
    selected=min(STARTS,key=lambda s:(nominal['fits'][s]['fit']['training_objective_final'],STARTS.index(s)))
    assert nominal['selected_start']==selected;equivalent(nominal['target'],nominal['fits'][selected]['target'])
    assert nominal['optimizer_success']==all(r['fit']['optimizer_success'] for r in nominal['fits'].values())
    full=np.ones(A.shape[1])/np.sqrt(A.shape[1]);fs=source['features'];mass=np.sqrt(A.shape[1])
    points={ARMS[0]:np.asarray(control['target']),ARMS[1]:np.asarray(nominal['target'])}
    risks={};alphas={}
    for eps,fit in ((.01,old['fits'][STARTS[0]]['fit']),(0.,nominal['fits'][STARTS[0]]['fit'])):
        validator=ScalarRankValidator(A[:,fs],fit['training_witnesses']);alpha=np.asarray(fit['training_witnesses']['alpha'])
        risks[eps]={arm:float(alpha@np.minimum(1.,validator.losses(p/mass))) for arm,p in points.items()};alphas[eps]=alpha
    np.testing.assert_array_equal(alphas[0.],alphas[.01])
    classes={}
    for arm,point in points.items():
        check=scalar_cell_evaluation(source['global_probe'],point)
        assert check['feasible'] and point.min()>=0 and abs(point.sum()-mass)<1e-8
        classes[arm]=check['class_upper_bounds']
    evaluation={'nominal_training_bounds':risks[0.],'robust_training_bounds':risks[.01],
        'nominal_bound_decrease_pp':100*(risks[0.][ARMS[0]]-risks[0.][ARMS[1]]),
        'robust_bound_decrease_pp':100*(risks[.01][ARMS[0]]-risks[.01][ARMS[1]]),
        'exact_zero_weights':{a:len(p)-int(np.count_nonzero(p)) for a,p in points.items()},
        'certified_class_bounds':classes,
        'optimizer_failures':{ARMS[0]:control['optimizer_failures'],ARMS[1]:sum(r['fit']['subproblem_failures'] for r in nominal['fits'].values())},
        'classification_scored':False}
    equivalent(evaluation,saved['evaluation']);counts['matched_nominal_robust_training_folds']+=1
