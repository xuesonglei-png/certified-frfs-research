"""ALL15 common crossgroup1000/geometry refinement; frozen upstream supports."""
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest,PARENT as POOL
from continue_soft_topk_knn_pilot import verify
from converged_crossgroup_stage import GRID,PredictiveMetricStage
from geometric_vote_evaluation import evaluate,final_evaluation
from exact_count_anchor_v2 import exact_count_policies
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired
from run_nested_joint_path import holm

SOURCE=ROOT/'results/atomic-pruning-classifier-v2'
PATHS=ROOT/'results/expanded-fgs-comparison'
FOLDER=ROOT/'results/converged-geometric-benchmark'
REFINED=('proposed','unguarded_refined','raw_repair_refined','full_refined','fgs_refined','sfrip_refined')
NATIVE=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')
MATCHED=('fgs_weighted_count120','fgs_uniform_count120','sfrip_weighted_count6','sfrip_uniform_count6')
METHODS=REFINED+NATIVE+MATCHED+('full_uniform',)
WEIGHTED=set(REFINED)|{'fgs_weighted_native120','sfrip_weighted_native6','fgs_weighted_count120','sfrip_weighted_count6'}
PRIMARY=('fgs_weighted_native120','fgs_weighted_count120','sfrip_weighted_native6','sfrip_weighted_count6')


def fingerprints(name):
    freeze=json.loads((ROOT/'research/converged-geometric-benchmark-source-freeze.json').read_text())
    actual={rel:digest(ROOT/rel) for rel in freeze['fingerprints']}
    assert actual==freeze['fingerprints'],'registered code changed'
    for folder in (SOURCE,PATHS,POOL):
        path=folder/(name+'.json');actual[path.relative_to(ROOT).as_posix()]=digest(path)
    return actual


def anchors(A,atomic,fgs,raw,selected,*,outer=False):
    d=A.shape[1];count=atomic['exact_proposed_count'];out={}
    for m in REFINED:
        op=atomic['results'][m] if outer else atomic['options'][m][selected[m]]
        out[m]=[{k:op['metric'][k] for k in ('features','weights')}|{'origin':'frozen_atomic_v2_selected_metric'}]
    graph=[r['path'] for r in fgs['paths']];sfrip=raw['paths']['sfrip2025']
    for prefix,paths in (('fgs',graph),('sfrip',sfrip)):
        suffix='120' if prefix=='fgs' else '6'
        out[prefix+'_native'+suffix]=[];out[prefix+'_weighted_native'+suffix]=[]
        for path in paths:
            fs=sorted(path['native']['features'])
            out[prefix+'_native'+suffix].append({'features':fs,'weights':[1.]*len(fs),'origin':'native_uniform'})
            out[prefix+'_weighted_native'+suffix].append({'features':fs,
                'weights':[float(np.sqrt(d)/len(fs))]*len(fs) if fs else [],'origin':'native_fixed_mass_uniform'})
    policies=exact_count_policies(A,{'paths':{'fgs2023':graph,'sfrip2025':sfrip}},count)
    for m,key in [('fgs_weighted_count120','fgs_matched'),('fgs_uniform_count120','fgs_uniform_matched'),
                  ('sfrip_weighted_count6','sfrip_matched'),('sfrip_uniform_count6','sfrip_uniform_matched')]:out[m]=policies[key]
    out['full_uniform']=[{'features':list(range(d)),'weights':(np.ones(d)/np.sqrt(d)).tolist(),'origin':'uniform_full'}]
    assert set(out)==set(METHODS)
    return out


def model(stage,base,m,ti):
    weighted=m in WEIGHTED and bool(base['features'])
    return stage.metric(base,ti if weighted else 0,weighted=weighted,certified=m=='proposed')


def fit_inner(A,y,B,truth,bases):
    stage=PredictiveMetricStage(A,y);records=[];lookup={};options={};start=time.perf_counter()
    for m in METHODS:
        options[m]=[]
        for ci,base in enumerate(bases[m]):
            for ti in range(len(GRID) if m in WEIGHTED else 1):
                metric=model(stage,base,m,ti);key=(tuple(metric['features']),tuple(metric['weights']))
                if key not in lookup:
                    lookup[key]=len(records);records.append({'metric':metric,'evaluation':evaluate(A,y,B,truth,metric)})
                options[m].append({'config_index':ci,'target_index':ti,'metric':metric,'record_id':lookup[key]})
    return {'bases':bases,'options':options,'records':records,'stage':stage.export(),'fit_and_score_seconds':time.perf_counter()-start}


def choose(blocks,m):
    rows=[]
    for i,op in enumerate(blocks[0]['options'][m]):
        records=[b['records'][b['options'][m][i]['record_id']] for b in blocks]
        rows.append({'config_index':op['config_index'],'target_index':op['target_index'],
            'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in records])),
            'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in records])),
            'mean_count':float(np.mean([len(r['metric']['features']) for r in records]))})
    index=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),rows[i]['mean_count'],i))
    return {'options':rows,'selected_index':index}


def run(name):
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);hashes=fingerprints(name);out=FOLDER/(name+'.json')
    if out.exists():
        stored=json.loads(out.read_text());assert stored['fingerprints']==hashes
        if stored['complete']:return
    source=json.loads((SOURCE/(name+'.json')).read_text());paths=json.loads((PATHS/(name+'.json')).read_text())
    pool=json.loads((POOL/(name+'.json')).read_text())
    for folder,blob in ((SOURCE,source),(PATHS,paths),(POOL,pool)):
        audit=json.loads((folder/'complete-audit.json').read_text());assert audit['passed']
        assert audit['artifact_hashes'][name]==digest(folder/(name+'.json'))
        for rel,sha in blob['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
        assert blob['reference_indices']==source['reference_indices'] and blob['validation_indices']==source['validation_indices']
    X,y=load_dataset(name);ri,vi=np.array(source['reference_indices']),np.array(source['validation_indices'])
    assert not set(ri)&set(vi);raw,labels=X[ri],y[ri];inner=[];selected=source['selection']['selected']
    for fi,(b,p,r) in enumerate(zip(source['inner'],paths['inner'],pool['candidate_reference_blocks'][:-1],strict=True)):
        for other in (p,r):
            assert other['reference_indices']==b['reference_indices'] and other['validation_indices']==b['validation_indices']
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);A,B=preprocess_fit(raw[ca],raw[cb])
        part=FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            cached=json.loads(part.read_text());assert cached['fingerprints']==hashes;fresh=cached['block']
        else:
            bases=anchors(A,b,p,r,selected);fresh=fit_inner(A,labels[ca],B,labels[cb],bases)
            fresh={'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'count':b['exact_proposed_count'],**fresh}
            part.write_text(json.dumps({'fingerprints':hashes,'block':fresh},separators=(',',':')),encoding='utf-8')
        inner.append(fresh)
        print(name,'inner',fi+1,'complete',len(fresh['stage']['learners']),'learners',len(fresh['records']),'unique evaluations',flush=True)
    selection={m:choose(inner,m) for m in METHODS};A,B=preprocess_fit(raw,X[vi])
    bases=anchors(A,source['outer'],paths['outer'],pool['candidate_reference_blocks'][-1],selected,outer=True)
    stage=PredictiveMetricStage(A,labels);results={}
    for m in METHODS:
        recipe=selection[m]['options'][selection[m]['selected_index']]
        metric=model(stage,bases[m][recipe['config_index']],m,recipe['target_index'])
        results[m]={'option_index':selection[m]['selected_index'],**recipe,'metric':metric,
                    **final_evaluation(A,labels,B,y[vi],metric)}
    count=len(results['proposed']['metric']['features'])
    assert count==source['outer']['exact_proposed_count']
    assert all(len(results[m]['metric']['features'])==count for m in MATCHED)
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),
        'fingerprints':hashes,'inner':inner,'selection':selection,
        'outer':{'bases':bases,'results':results,'stage':stage.export(),'count':count},
        'complete':True,'query_labels_used_for_selection':False,'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_exposed_development_sequential_frozen_support_refinement_common1000_and_geometric_votes'}
    assert fingerprints(name)==hashes;out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'OUTER COMPLETE',results['proposed']['evaluation']['metrics']['clean_BA'],flush=True)


def summarize(names):
    rows=[];failures=0;fallbacks=0
    for name in sorted(names):
        saved=json.loads((FOLDER/(name+'.json')).read_text());r=saved['outer']['results'];count=saved['outer']['count']
        rows.append({'dataset':name,'d':saved['d'],'removed':1-count/saved['d'],
            'actual_counts':{m:len(r[m]['metric']['features']) for m in METHODS},
            'positive_weight_counts':{m:sum(w>0 for w in r[m]['metric']['weights']) for m in METHODS},
            'metrics':{m:r[m]['evaluation']['metrics'] for m in METHODS}})
        for b in saved['inner']+[saved['outer']]:
            failures+=sum(not l['fit']['optimizer_success'] for l in b['stage']['learners'])
            if 'options' in b:fallbacks+=sum(o['metric']['guard_fallback'] for o in b['options']['proposed'])
        fallbacks+=r['proposed']['metric']['guard_fallback']
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    contrasts={m:paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m!='proposed'}
    corrected=holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in PRIMARY});removed=float(np.mean([r['removed'] for r in rows]))
    gate=all(contrasts[m]['mean_pp']>=.5 and contrasts[m]['wins']>=10 and contrasts[m]['bootstrap95_pp'][0]>0
             and corrected[m]<.05 for m in PRIMARY) and removed>=.1 and contrasts['full_uniform']['mean_pp']>=0 and failures==0
    inherited=json.loads((SOURCE/'complete.json').read_text())['counts']
    summary={'rows':rows,'means':means,'contrasts_clean_BA':contrasts,'primary_descriptive_holm':corrected,
        'mean_removed':removed,'counts':{'datasets':len(rows),'inner_folds':sum(len(json.loads((FOLDER/(n+'.json')).read_text())['inner']) for n in names),
        'new_target_optimizer_failures':failures,'proposed_fallback_options':fallbacks,
        'upstream_optimizer_failures_preserved':inherited['inherited_optimizer_failures']+inherited['new_exact_optimizer_failures']},
        'pilot_gate_passed':bool(gate),'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_repeatedly_exposed_development_not_confirmatory_prediction_superiority',
        'limitations':['Sequential tuning of frozen upstream selections, not fresh fully nested joint support search.',
                      'FGS and SFRIP are documented numerical reproductions; weighted variants are adaptations.',
                      'No claim of state-of-the-art or universal test advantage; new data needed after method freeze.',
                      'Geometric tie convention and common optimization are not novel FRS theory.']}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=sorted(verify(SOURCE)['artifact_hashes']);verify(PATHS);verify(POOL)
    assert len(names)==15
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(run,n) for n in names]):f.result()
    summarize(names)
