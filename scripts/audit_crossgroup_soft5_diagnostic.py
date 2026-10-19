"""Independent scalar objectives/polytope/guard/vote replay; inner only."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_crossgroup_soft5_diagnostic import ROOT,SOURCE,FOLDER,METHODS,digest
from crossgroup_soft5_stage import GRID
from audit_crossgroup_soft5_stage import check_stage,check_metric
from audit_certified_metric_knn_pool import independent_evaluate
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+'.json');s=json.loads(path.read_text());counts=Counter(datasets=1)
    assert s['complete'] and not s['new_outer_evaluation_performed'] and not s['query_labels_used_for_fit']
    for rel,sha in s['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    pa=verify(SOURCE);assert pa['artifact_hashes'][name]==digest(SOURCE/(name+'.json'))
    parent=json.loads((SOURCE/(name+'.json')).read_text());assert s['reference_indices']==parent['reference_indices']
    X,y=load_dataset(name);ri=np.array(s['reference_indices']);raw,labels=X[ri],y[ri]
    assert len(s['inner'])==len(parent['inner'])
    for fi,(b,old) in enumerate(zip(s['inner'],parent['inner'],strict=True)):
        assert b['reference_indices']==old['reference_indices'] and b['validation_indices']==old['validation_indices']
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);stage=b['stage'];check_stage(A,labels[ca],stage,counts)
        assert set(b['options'])==set(b['bases'])==set(METHODS)
        for m in METHODS:
            anchor=old['options'][m][parent['selection']['selected'][m]]['metric'];base=b['bases'][m]
            assert base['features']==anchor['features'] and base['weights']==anchor['weights']
            assert base['origin']=='frozen_audited_rescale_selected_metric'
            assert len(b['options'][m])==len(GRID)
            for i,row in enumerate(b['options'][m]):
                assert row['target_index']==i
                check_metric(row['metric'],base,stage,'proposed' if m=='proposed' else 'full',i,counts)
                metric=row['metric']
                fresh=independent_evaluate(A,labels[ca],B,labels[cb],metric['features'],metric['weights'])
                equivalent(fresh,row['evaluation']);counts['inner_predictions']+=1
        counts['inner_folds']+=1;print(name,'inner',fi+1,'crossgroup independently audited',flush=True)
    op=s['selection']['options'];sel=s['selection']['selected']
    assert set(op)==set(sel)==set(METHODS)
    for m in METHODS:
        assert len(op[m])==len(GRID)
        for i,row in enumerate(op[m]):
            assert row['target_index']==i
            for target,metric in [('mean_clean_BA','clean_BA'),('mean_balanced_logloss','balanced_logloss')]:
                equivalent(row[target],float(np.mean([b['options'][m][i]['evaluation']['metrics'][metric] for b in s['inner']])))
        choice=min(range(len(GRID)),key=lambda i:(-round(op[m][i]['mean_clean_BA'],12),round(op[m][i]['mean_balanced_logloss'],12),i))
        assert sel[m]==choice
    row={'dataset':name,'selected':sel,'anchor_BA':{m:op[m][0]['mean_clean_BA'] for m in METHODS},
        'selected_inner_BA':{m:op[m][sel[m]]['mean_clean_BA'] for m in METHODS},
        'fixed_grid_inner_BA':{m:[r['mean_clean_BA'] for r in op[m]] for m in METHODS}}
    return {'dataset':name,'sha256':digest(path),'counts':dict(counts),'row':row}


if __name__=='__main__':
    summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter()
    with ProcessPoolExecutor(max_workers=1) as executor:
        for future in as_completed([executor.submit(audit,r['dataset']) for r in summary['rows']]):
            result=future.result();reports.append(result);counts.update(result['counts'])
    rows=sorted([r['row'] for r in reports],key=lambda r:r['dataset']);equivalent(rows,summary['rows'])
    means={m:{'anchor':float(np.mean([r['anchor_BA'][m] for r in rows])),
        'selected_inner':float(np.mean([r['selected_inner_BA'][m] for r in rows])),
        'fixed_grid':np.mean([r['fixed_grid_inner_BA'][m] for r in rows],axis=0).tolist()} for m in METHODS}
    equivalent(means,summary['means']);assert summary['target_optimizer_failures']==counts['target_optimizer_failures']
    assert counts['datasets']==15 and counts['inner_folds']==44
    output={'passed':True,'counts':dict(counts),'artifact_hashes':{r['dataset']:r['sha256'] for r in reports},
        'summary_sha256':digest(FOLDER/'complete.json'),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'scope':'all15_44_inner_independent_crossgroup_scalar_loss_guards_votes_choices_not_outer_or_risk'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(output,flush=True)
