"""Common14-choice weightedFGS120 and uniformFGS120 at audited atomic counts."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest,final_evaluation
from run_multiscale_certified_metric_knn_pool import evaluate
from simplex_soft5_knn_stage import GRID,PredictiveMetricStage
from exact_count_anchor_v2 import exact_count_policies
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired
from run_nested_joint_path import holm

SOURCE=ROOT/'results/atomic-pruning-classifier-v2'
PATHS=ROOT/'results/expanded-fgs-comparison'
FOLDER=ROOT/'results/fgs120-atomic-count-controls'
METHODS=('fgs_weighted120_atomic_count','fgs_uniform120_atomic_count')


def bases(A,paths,count):
    block={'paths':{'fgs2023':[row['path'] for row in paths],'sfrip2025':[]}}
    policy=exact_count_policies(A,block,count)
    return {METHODS[0]:policy['fgs_matched'],METHODS[1]:policy['fgs_uniform_matched']}


def fit_inner(A,y,B,truth,paths,count):
    anchors=bases(A,paths,count);stage=PredictiveMetricStage(A,y);records=[];lookup={};options={}
    for method in METHODS:
        weighted=method==METHODS[0];options[method]=[]
        for ci,base in enumerate(anchors[method]):
            for ti in range(len(GRID) if weighted else 1):
                metric=stage.metric(base,ti,certified=False,weighted=weighted)
                key=(tuple(metric['features']),tuple(metric['weights']))
                if key not in lookup:
                    lookup[key]=len(records);records.append({'metric':metric,
                        'evaluation':evaluate(A,y,B,truth,metric['features'],metric['weights'])})
                # Metrics contain recipe identity even if coordinates duplicate.
                options[method].append({'config_index':ci,'target_index':ti,'metric':metric,'record_id':lookup[key]})
    return {'count':count,'options':options,'records':records,'stage':stage.export()}


def choose(blocks,method):
    combined=[]
    for i,op in enumerate(blocks[0]['options'][method]):
        values=[b['records'][b['options'][method][i]['record_id']]['evaluation']['metrics'] for b in blocks]
        combined.append({'config_index':op['config_index'],'target_index':op['target_index'],
            'mean_BA':float(np.mean([v['clean_BA'] for v in values])),
            'mean_logloss':float(np.mean([v['balanced_logloss'] for v in values]))})
    index=min(range(len(combined)),key=lambda i:(-round(combined[i]['mean_BA'],12),round(combined[i]['mean_logloss'],12),i))
    return {'options':combined,'selected_index':index}


def outer_models(A,y,paths,count,selection):
    anchors=bases(A,paths,count);stage=PredictiveMetricStage(A,y);results={}
    for method in METHODS:
        op=selection[method]['options'][selection[method]['selected_index']]
        metric=stage.metric(anchors[method][op['config_index']],op['target_index'],
            weighted=method==METHODS[0],certified=False)
        results[method]={**op,'option_index':selection[method]['selected_index'],'metric':metric}
    return results,stage.export()


def fingerprints(name):
    frozen=json.loads((ROOT/'research/fgs120-atomic-count-controls-source-freeze.json').read_text())
    actual={rel:digest(ROOT/rel) for rel in frozen['fingerprints']};assert actual==frozen['fingerprints']
    files=[SOURCE/(name+'.json'),SOURCE/'complete-audit.json',PATHS/(name+'.json'),PATHS/'complete-audit.json']
    actual.update({p.relative_to(ROOT).as_posix():digest(p) for p in files});return actual


def run(name):
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);hashes=fingerprints(name);out=FOLDER/(name+'.json')
    if out.exists():
        old=json.loads(out.read_text());assert old['fingerprints']==hashes
        if old['complete']:return
    for folder in (SOURCE,PATHS):assert verify(folder)['artifact_hashes'][name]==digest(folder/(name+'.json'))
    source=json.loads((SOURCE/(name+'.json')).read_text());paths=json.loads((PATHS/(name+'.json')).read_text())
    assert source['reference_indices']==paths['reference_indices'] and source['validation_indices']==paths['validation_indices']
    X,y=load_dataset(name);ri,vi=np.array(source['reference_indices']),np.array(source['validation_indices']);raw,labels=X[ri],y[ri]
    inner=[]
    for fi,(block,pblock) in enumerate(zip(source['inner'],paths['inner'],strict=True)):
        assert block['reference_indices']==pblock['reference_indices'] and block['validation_indices']==pblock['validation_indices']
        ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices']);A,B=preprocess_fit(raw[ca],raw[cb])
        fresh=fit_inner(A,labels[ca],B,labels[cb],pblock['paths'],block['exact_proposed_count'])
        inner.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),**fresh})
        print(name,'inner',fi+1,'FGS120 weighted/uniform exact count',fresh['count'],'learners',len(fresh['stage']['learners']),flush=True)
    selection={m:choose(inner,m) for m in METHODS};A,B=preprocess_fit(raw,X[vi]);count=source['outer']['exact_proposed_count']
    results,stage=outer_models(A,labels,paths['outer']['paths'],count,selection)
    for result in results.values():result.update(final_evaluation(A,labels,B,y[vi],result['metric']))
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),
        'fingerprints':hashes,'inner':inner,'selection':selection,'outer':{'results':results,'stage':stage,'count':count},
        'complete':True,'query_labels_used_for_selection':False,'independent_audit_required':True,'goal_achieved':False,
        'scope':'all15_common_simplex14_FGS120_ranking_atomic_count_adaptations_not_author_native_algorithms_or_confirmation'}
    assert fingerprints(name)==hashes;out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'COMPLETE FGS120 count controls',flush=True)


def summarize(names):
    source=json.loads((SOURCE/'complete.json').read_text());old={r['dataset']:r for r in source['rows']};rows=[];failures=0
    for name in sorted(names):
        s=json.loads((FOLDER/(name+'.json')).read_text());r=s['outer']['results'];count=s['outer']['count']
        assert all(len(r[m]['metric']['features'])==count==old[name]['actual_counts']['proposed'] for m in METHODS)
        rows.append({'dataset':name,'count':count,'metrics':{m:r[m]['evaluation']['metrics'] for m in METHODS},
            'proposed_metrics':old[name]['metrics']['proposed']})
        failures+=sum(not l['fit']['optimizer_success'] for b in s['inner']+[s['outer']] for l in b['stage']['learners'])
    contrasts={m:paired([r['proposed_metrics']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS}
    output={'rows':rows,'means':{m:float(np.mean([r['metrics'][m]['clean_BA'] for r in rows])) for m in METHODS},
        'proposed_minus_controls':contrasts,'descriptive_holm':holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in METHODS}),
        'target_optimizer_failures':failures,'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_exposed_development_common_weighted14_and_uniform_FGS120_exact_atomic_counts'}
    (FOLDER/'complete.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print({k:v for k,v in output.items() if k!='rows'},flush=True)


if __name__=='__main__':
    names=sorted(verify(SOURCE)['artifact_hashes'])
    for name in names:run(name)
    summarize(names)
