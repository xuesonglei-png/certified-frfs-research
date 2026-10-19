"""ALL15 same fixed convex target; inner-only diagnostic, common arms."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_converged_geometric_benchmark import ROOT,SOURCE,REFINED,digest
from majority_hinge_stage import GRID,PredictiveMetricStage
from geometric_vote_evaluation import evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify

FOLDER=ROOT/'results/majority-hinge-inner-all15'
NAMES=tuple(sorted(verify(SOURCE)['artifact_hashes']))


def run(name):
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    freeze=json.loads((ROOT/'research/majority-hinge-inner-all15-source-freeze.json').read_text())
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    source_path=SOURCE/(name+'.json');assert verify(SOURCE)['artifact_hashes'][name]==digest(source_path)
    source=json.loads(source_path.read_text());out=FOLDER/(name+'.json');hashes={**freeze['fingerprints'],source_path.relative_to(ROOT).as_posix():digest(source_path)}
    if out.exists():
        saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return
    X,y=load_dataset(name);ri=np.array(source['reference_indices']);raw,labels=X[ri],y[ri];inner=[]
    for fi,b in enumerate(source['inner']):
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);A,B=preprocess_fit(raw[ca],raw[cb]);stage=PredictiveMetricStage(A,labels[ca])
        bases={m:{k:b['options'][m][source['selection']['selected'][m]]['metric'][k] for k in ('features','weights')}|
                 {'origin':'frozen_atomic_v2_selected_metric'} for m in REFINED}
        records=[];lookup={};options={}
        for m in REFINED:
            options[m]=[]
            for ti in range(len(GRID)):
                metric=stage.metric(bases[m],ti,certified=m=='proposed');key=(tuple(metric['features']),tuple(metric['weights']))
                if key not in lookup:
                    lookup[key]=len(records);records.append({'metric':metric,'evaluation':evaluate(A,labels[ca],B,labels[cb],metric)})
                options[m].append({'target_index':ti,'metric':metric,'record_id':lookup[key]})
        inner.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'bases':bases,'stage':stage.export(),'records':records,'options':options})
        print(name,'inner',fi+1,'LP targets',len(stage.learners),'failures',sum(not r['fit']['optimizer_success'] for r in stage.learners),flush=True)
    selection={}
    for m in REFINED:
        rows=[]
        for i in range(len(GRID)):
            scores=[b['records'][b['options'][m][i]['record_id']]['evaluation']['metrics'] for b in inner]
            rows.append({'target_index':i,'mean_BA':float(np.mean([r['clean_BA'] for r in scores])),
                'mean_logloss':float(np.mean([r['balanced_logloss'] for r in scores]))})
        choice=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),i))
        selection[m]={'options':rows,'selected':choice,'selected_inner_gain_pp':100*(rows[choice]['mean_BA']-rows[0]['mean_BA'])}
    saved={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'inner':inner,'selection':selection,
        'complete':True,'goal_achieved':False,'new_outer_evaluation_performed':False,'scope':'all15_exposed_inner_only_common_convex_majority_hinge_prototype'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8');print(name,'INNER COMPLETE',selection['proposed']['selected_inner_gain_pp'],flush=True)


def main():
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(run,n) for n in NAMES]):f.result()
    reports=[json.loads((FOLDER/(n+'.json')).read_text()) for n in NAMES]
    summary={'datasets':list(NAMES),'inner_folds':sum(len(r['inner']) for r in reports),
        'mean_inner_gain_pp':{m:float(np.mean([r['selection'][m]['selected_inner_gain_pp'] for r in reports])) for m in REFINED},
        'target_optimizer_failures':sum(not l['fit']['optimizer_success'] for r in reports for b in r['inner'] for l in b['stage']['learners']),
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in NAMES},'goal_achieved':False,
        'new_outer_evaluation_performed':False,'independent_audit_required':True,
        'scope':'inner_only_reselection_gains_not_prediction_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
