"""ALL15 shared post-refinement and separately rebuilt exact-count controls."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,FOLDER as SOURCE,PARENT as POOL,digest,final_evaluation
from run_multiscale_certified_metric_knn_pool import policy_metrics,evaluate
from simplex_soft5_knn_stage import GRID,PredictiveMetricStage
from postmetric_pruning_classifier_stage import REFINED,PruningStage
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired
from run_nested_joint_path import holm

FOLDER=ROOT/'results/postmetric-pruning-classifier'
SOURCE_ONLY={'full_source':'full','proposed_source':'proposed','fgs_native':'fgs_native','sfrip_native':'sfrip_native'}
EXACT={'fgs_exact':'fgs_matched','sfrip_exact':'sfrip_matched',
       'fgs_uniform_exact':'fgs_uniform_matched','sfrip_uniform_exact':'sfrip_uniform_matched'}
METHODS=tuple(REFINED)+tuple(SOURCE_ONLY)+tuple(EXACT)


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/postmetric_pruning_classifier_stage.py',ROOT/'scripts/guarded_postmetric_pruning.py',
        ROOT/'scripts/independent_postmetric_pruning.py',ROOT/'scripts/audit_postmetric_pruning_classifier.py',
        ROOT/'scripts/simplex_soft5_knn_stage.py',ROOT/'scripts/simplex_soft_topk_targets.py',
        ROOT/'scripts/soft_topk_metric_objective.py',ROOT/'scripts/soft_topk_predictive_metric.py',
        ROOT/'scripts/hardaware_guarded_soft5_descent.py',ROOT/'scripts/run_multiscale_certified_metric_knn_pool.py',
        ROOT/'scripts/run_certified_metric_knn_pool.py',ROOT/'scripts/run_simplex_soft5_knn_pilot.py',
        ROOT/'research/postmetric-pruning-classifier-protocol.md',SOURCE/(name+'.json'),SOURCE/'complete-audit.json',
        POOL/(name+'.json'),POOL/'complete-audit.json',ROOT/'results/postmetric-pruning-admission/complete-audit.json',
        ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def source_metric(parent,block,method,*,outer=False):
    if outer: return block['results'][method]['metric']
    i=parent['selection']['selected'][method]
    return block['options'][method][i]['metric']


def choose(rows):
    return min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_clean_BA'],12),
        round(rows[i]['mean_balanced_logloss'],12),rows[i]['mean_actual_count'],i))


def combine(inners,method):
    rows=[]
    for i,stored in enumerate(inners[0]['options'][method]):
        folds=[b['options'][method][i] for b in inners]
        rows.append({k:stored[k] for k in ('variant','selector_index','target_index') if k in stored})
        rows[-1].update({'mean_clean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in folds])),
            'mean_balanced_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in folds])),
            'mean_actual_count':float(np.mean([len(r['metric']['features']) for r in folds]))})
    return rows


def scored(A,y,B,truth,metric,**recipe):
    return {**recipe,'metric':metric,'evaluation':evaluate(A,y,B,truth,metric['features'],metric['weights'])}


def exact_options(stage,raw_block,count,method):
    key=EXACT[method]; bases=policy_metrics(stage.A,raw_block,count=count)[key]
    weighted=method in ('fgs_exact','sfrip_exact'); rows=[]
    for si,base in enumerate(bases):
        for ti in range(len(GRID) if weighted else 1):
            metric=stage.metric(base,ti,weighted=weighted,certified=False)
            rows.append({'selector_index':si,'target_index':ti,'metric':metric})
    return rows


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); pa=verify(SOURCE); ma=verify(POOL)
    assert pa['artifact_hashes'][name]==digest(SOURCE/(name+'.json')) and ma['artifact_hashes'][name]==digest(POOL/(name+'.json'))
    parent=json.loads((SOURCE/(name+'.json')).read_text()); raw_parent=json.loads((POOL/(name+'.json')).read_text())
    for rel,sha in parent['fingerprints'].items(): assert digest(ROOT/rel)==sha
    hashes=fingerprints(name); out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return
    X,y=load_dataset(name); ri,vi=np.array(parent['reference_indices']),np.array(parent['validation_indices'])
    assert parent['reference_indices']==raw_parent['reference_indices'] and parent['validation_indices']==raw_parent['validation_indices']
    raw,labels=X[ri],y[ri]; inners=[]; fitted=[]
    for source in parent['inner']:
        ca,cb=np.array(source['reference_indices']),np.array(source['validation_indices'])
        A,B=preprocess_fit(raw[ca],raw[cb]); stage=PruningStage(A,labels[ca]); options={}
        for m,old in REFINED.items():
            base=source_metric(parent,source,old)
            options[m]=[scored(A,labels[ca],B,labels[cb],stage.metric(base,v,require_original=m=='proposed'),variant=v) for v in (0,1)]
        for m,old in SOURCE_ONLY.items():
            base=source_metric(parent,source,old)
            metric={'features':base['features'],'weights':base['weights'],'variant':0,'pruning_id':None,'guard_reference':None}
            options[m]=[scored(A,labels[ca],B,labels[cb],metric)]
        inners.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'options':options,
                       'pruning_records':stage.records})
        fitted.append((A,B,labels[ca],labels[cb]))
        print(name,'inner',len(inners),'shared refinement complete',len(stage.records),'unique pruning fits',flush=True)
    combined={m:combine(inners,m) for m in tuple(REFINED)+tuple(SOURCE_ONLY)}
    selected={m:choose(rows) for m,rows in combined.items()}; prop_variant=combined['proposed'][selected['proposed']]['variant']
    for bi,(source,block,tables) in enumerate(zip(raw_parent['candidate_reference_blocks'][:-1],inners,fitted,strict=True)):
        A,B,train_y,valid_y=tables; count=len(block['options']['proposed'][prop_variant]['metric']['features'])
        block['exact_proposed_count']=count; stage=PredictiveMetricStage(A,train_y)
        for m in EXACT:
            block['options'][m]=[{**r,'evaluation':evaluate(A,train_y,B,valid_y,r['metric']['features'],r['metric']['weights'])}
                for r in exact_options(stage,source,count,m)]
        block['exact_stage']=stage.export()
        print(name,'inner',bi+1,'exact count',count,'target fits',len(stage.learners),flush=True)
    for m in EXACT: combined[m]=combine(inners,m); selected[m]=choose(combined[m])
    A,B=preprocess_fit(raw,X[vi]); stage=PruningStage(A,labels); results={}
    for m,old in REFINED.items():
        variant=combined[m][selected[m]]['variant']; base=source_metric(parent,parent['outer'],old,outer=True)
        metric=stage.metric(base,variant,require_original=m=='proposed')
        results[m]={'option_index':selected[m],'metric':metric,**final_evaluation(A,labels,B,y[vi],metric)}
    for m,old in SOURCE_ONLY.items():
        base=source_metric(parent,parent['outer'],old,outer=True)
        metric={'features':base['features'],'weights':base['weights'],'variant':0,'pruning_id':None,'guard_reference':None}
        results[m]={'option_index':0,'metric':metric,**final_evaluation(A,labels,B,y[vi],metric)}
    count=len(results['proposed']['metric']['features']); exact_stage=PredictiveMetricStage(A,labels)
    for m in EXACT:
        rows=exact_options(exact_stage,raw_parent['candidate_reference_blocks'][-1],count,m)
        chosen=rows[selected[m]]; metric=chosen['metric']
        results[m]={'option_index':selected[m],**chosen,**final_evaluation(A,labels,B,y[vi],metric)}
        assert len(metric['features'])==count
    assert hashes==fingerprints(name)
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),
        'fingerprints':hashes,'inner':inners,'selection':{'options':combined,'selected':selected},
        'outer':{'results':results,'pruning_records':stage.records,'exact_stage':exact_stage.export(),'exact_proposed_count':count},
        'complete':True,'query_labels_used_for_selection':False,'independent_audit_required':True,'goal_achieved':False,
        'scope':'all15_exposed_development_frozen_source_shared_binary_postpruning_distinct_exact_count_controls'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'COMPLETE','proposed BA',results['proposed']['evaluation']['metrics']['clean_BA'],'count',count,flush=True)


def summarize(names):
    rows=[]; new_failures=0; pruning_checks=0
    for name in sorted(names):
        saved=json.loads((FOLDER/(name+'.json')).read_text()); result=saved['outer']['results']; assert saved['complete']
        rows.append({'dataset':name,'d':saved['d'],'actual_counts':{m:len(result[m]['metric']['features']) for m in METHODS},
            'removed':1-len(result['proposed']['metric']['features'])/saved['d'],
            'metrics':{m:result[m]['evaluation']['metrics'] for m in METHODS}})
        for block in saved['inner']+[saved['outer']]:
            new_failures+=sum(not l['fit']['optimizer_success'] for l in block['exact_stage']['learners'])
            pruning_checks+=sum(len(r['fit']['checks']) for r in block['pruning_records'])
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    contrasts={m:paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m!='proposed'}
    primary=('fgs_refined','sfrip_refined'); removed=float(np.mean([r['removed'] for r in rows]))
    inherited=json.loads((SOURCE/'complete.json').read_text())['counts']['target_optimizer_failures']
    gate=all(contrasts[m]['mean_pp']>=.5 and contrasts[m]['wins']>=10 for m in primary) and removed>=.1 and contrasts['full_source']['mean_pp']>=0 and inherited+new_failures==0
    result={'rows':rows,'means':means,'contrasts_clean_BA':contrasts,'mean_removed':removed,
        'counts':{'datasets':len(rows),'inner_folds':sum(len(json.loads((FOLDER/(n+'.json')).read_text())['inner']) for n in names),
            'pruning_checks':pruning_checks,'new_exact_optimizer_failures':new_failures,'inherited_optimizer_failures':inherited},
        'primary_descriptive_holm':holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in primary}),
        'pilot_gate_passed':bool(gate),'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_exposed_development_shared_refinement_primary_and_separate_exact_count_controls_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True); names=sorted(verify(SOURCE)['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=1) as executor:
        for f in as_completed([executor.submit(run,name) for name in names]): f.result()
    summarize(names)
