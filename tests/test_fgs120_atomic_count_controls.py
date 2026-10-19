from itertools import permutations
from collections import Counter
import numpy as np
from run_fgs120_atomic_count_controls import fit_inner,choose,outer_models,METHODS
from audit_fgs120_atomic_count_controls import independent_bases
from audit_simplex_soft5_knn_pilot import check_stage,check_metric
from audit_certified_metric_knn_pool import independent_evaluate
from audit_forest_compact_transfer import equivalent
from simplex_soft5_knn_stage import GRID


def test_all120_config_mapping_shared_targets_and_independent_predictions():
    rng=np.random.default_rng(91);A=rng.uniform(size=(24,5));B=rng.uniform(size=(9,5))
    y=np.arange(24)%3;truth=np.arange(9)%3
    paths=[{'path':{'ranking':list(p),'native':{'features':[p[0]]}}} for p in permutations(range(5))]
    block=fit_inner(A,y,B,truth,paths,2);anchors=independent_bases(A,paths,2);counts=Counter()
    check_stage(A,y,block['stage'],counts)
    for m in METHODS:
        step=len(GRID) if m==METHODS[0] else 1
        assert len(block['options'][m])==120*step
        for i,op in enumerate(block['options'][m]):
            assert op['config_index']==i//step and op['target_index']==i%step
            check_metric(op['metric'],anchors[m][i//step],block['stage'],
                'fgs_matched' if step>1 else 'fgs_uniform_matched',i%step,counts)
            assert len(op['metric']['features'])==2
    for record in block['records']:
        metric=record['metric']
        equivalent(independent_evaluate(A,y,B,truth,metric['features'],metric['weights']),record['evaluation'])
    selection={m:choose([block],m) for m in METHODS}
    models,stage=outer_models(A,y,paths,2,selection)
    for m in METHODS:
        assert len(models[m]['metric']['features'])==2
        assert models[m]['option_index']==selection[m]['selected_index']
