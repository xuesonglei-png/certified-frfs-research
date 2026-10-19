"""Full train-only classifier-capacity tuning with refolded support selection.

Each internal training fold regenerates its certified and recent-selector
supports and its three-state interaction trees. Internal validation labels
enter only classifier scores, never those selectors or preprocessing.
Recent selector configurations are fixed for this development pilot.
"""
import numpy as np
from .nested_teacher_regularized import inner_splits
from .validated_pool import preprocess_fit
from .geometry_only_forest_stage import GeometryStageContext
from .recent_budget import fgs_budget_path,sfrip_budget_path
from .functional_spline_forest import fit_functional_spline_forest
from .spline_forest import spline_scores
from .forest_penalty_selection import balanced_accuracy

METHODS = ('certified','full','fgs_native','sfrip_native','fgs_matched','sfrip_matched')
CS = (.001,.01,.1,1.,10.)
PAIRS = (None,10.,1.)
CONFIGS = [{'knots':[0.,.5,1.] if k == 3 else [0.,.25,.5,.75,1.],
    'k':k,'C':C,'pair_penalty':gamma,'k_index':ki,'C_index':ci,'pair_index':gi}
    for ki,k in enumerate((3,5)) for ci,C in enumerate(CS) for gi,gamma in enumerate(PAIRS)]
SEED = 20261017


def support_block(raw,y):
    context = GeometryStageContext(raw,y)
    stage = context.fit_stage(C=1.,pair_penalty=1.)
    candidate = stage['policies']['0.8']['candidate']; target = len(candidate['features'])
    fgs = fgs_budget_path(context.A,context.y,beta=.8,xi=1.)
    sfrip = sfrip_budget_path(context.A,context.y,beta=3.,confidence=.8)
    supports = {'certified':candidate['features'],'full':list(range(context.d)),
        'fgs_native':sorted(fgs['native']['features']),'sfrip_native':sorted(sfrip['native']['features']),
        'fgs_matched':sorted(fgs['ranking'][:target]),'sfrip_matched':sorted(sfrip['ranking'][:target])}
    assert all(len(supports[m]) == target for m in ('fgs_matched','sfrip_matched'))
    return context,{'supports':supports,'candidate':candidate,'target_count':target,'fgs_path':fgs,'sfrip_path':sfrip}


def choose(rows):
    if len(rows) != len(CONFIGS): raise ValueError('complete common classifier grid required')
    values = [float(np.mean(row['fold_clean'])) for row in rows]
    if any(not np.isfinite(v) for v in values): raise ValueError('finite training-validation scores required')
    selected = max(range(len(rows)),key=lambda i:(round(values[i],12),-CONFIGS[i]['k_index'],
        -CONFIGS[i]['C_index'],-CONFIGS[i]['pair_index']))
    return {'config_index':selected,'mean_clean':values[selected],'all_mean_clean':values,
        'tie_rule':'rounded_clean_lower_k_stronger_regularization_pair_index',
        'scope':'full_internal_pipeline_validation_choice_not_unbiased_test_risk_or_population_guarantee'}


def fit_candidate(A,y,fs,edges,config):
    gamma = config['pair_penalty']
    return fit_functional_spline_forest(A,y,fs,[] if gamma is None else edges,
        knots=config['knots'],C=config['C'],pair_penalty=1. if gamma is None else gamma)


def tune_functional_pipelines(raw,y):
    raw,y = np.asarray(raw,float),np.asarray(y)
    traces = {m:[{'config_index':ci,'fold_clean':[],'fold_model_ids':[]} for ci in range(len(CONFIGS))] for m in METHODS}
    folds = []
    for a,b in inner_splits(y,seed=SEED):
        context,supports = support_block(raw[a],y[a]); A,B = preprocess_fit(raw[a],raw[b]); models = []; cache = {}
        for method in METHODS:
            fs = supports['supports'][method]; edges = [] if not fs else context.tree(fs)['edges']
            for ci,config in enumerate(CONFIGS):
                key = (tuple(fs),ci)
                if key not in cache:
                    fit = fit_candidate(A,y[a],fs,edges,config)
                    score = balanced_accuracy(spline_scores(B,fit),fit['classes'],y[b]); cache[key] = len(models)
                    models.append({'fit':fit,'clean':score})
                mid = cache[key]; row = traces[method][ci]
                row['fold_clean'].append(models[mid]['clean']); row['fold_model_ids'].append(mid)
        folds.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'models':models,
            'support_block':supports,'trees':list(context.trees.values())})
    return {'folds':folds,'trace':traces,'choices':{m:choose(rows) for m,rows in traces.items()},'seed':SEED,
        'scope':'train_only_refolded_feature_selection_scaling_trees_common_functional_classifier_grid'}
