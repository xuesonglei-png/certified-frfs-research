"""ALL15/44 inner-only shared crossgroup fit; no new outer evaluation."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_simplex_soft5_knn_pilot import ROOT,digest
from run_multiscale_certified_metric_knn_pool import evaluate
from crossgroup_soft5_stage import GRID,PredictiveMetricStage
from continue_soft_topk_knn_pilot import verify

SOURCE=ROOT/'results/postmetric-pruning-classifier'
FOLDER=ROOT/'results/crossgroup-soft5-diagnostic'
METHODS=('proposed','unguarded_refined','raw_repair_refined','full_refined','fgs_refined','sfrip_refined')


def fingerprints(name):
    frozen=json.loads((ROOT/'research/crossgroup-soft5-diagnostic-source-freeze.json').read_text())
    files={rel:digest(ROOT/rel) for rel in frozen['fingerprints']}
    assert files==frozen['fingerprints']
    files.update({(SOURCE/(name+'.json')).relative_to(ROOT).as_posix():digest(SOURCE/(name+'.json'))})
    return files


def build(stage,bases):
    return {method:[stage.metric(base,i,certified=method=='proposed') for i in range(len(GRID))]
            for method,base in bases.items()}


def run(name):
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);hashes=fingerprints(name)
    out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return
    audit=verify(SOURCE);assert audit['artifact_hashes'][name]==digest(SOURCE/(name+'.json'))
    parent=json.loads((SOURCE/(name+'.json')).read_text())
    for rel,sha in parent['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    X,y=load_dataset(name);ri=np.array(parent['reference_indices']);raw,labels=X[ri],y[ri]
    inner=[]
    for old in parent['inner']:
        ca,cb=np.array(old['reference_indices']),np.array(old['validation_indices'])
        A,B=preprocess_fit(raw[ca],raw[cb]);stage=PredictiveMetricStage(A,labels[ca])
        bases={m:{'features':old['options'][m][parent['selection']['selected'][m]]['metric']['features'],
                  'weights':old['options'][m][parent['selection']['selected'][m]]['metric']['weights'],
                  'origin':'frozen_audited_rescale_selected_metric'} for m in METHODS}
        metrics=build(stage,bases)
        options={m:[{'target_index':i,'metric':metric,
            'evaluation':evaluate(A,labels[ca],B,labels[cb],metric['features'],metric['weights'])}
            for i,metric in enumerate(metrics[m])] for m in METHODS}
        inner.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),
            'bases':bases,'options':options,'stage':stage.export()})
        print(name,'inner',len(inner),'crossgroup targets',len(stage.learners),flush=True)
    combined={m:[{'target_index':i,
        'mean_clean_BA':float(np.mean([b['options'][m][i]['evaluation']['metrics']['clean_BA'] for b in inner])),
        'mean_balanced_logloss':float(np.mean([b['options'][m][i]['evaluation']['metrics']['balanced_logloss'] for b in inner]))}
        for i in range(len(GRID))] for m in METHODS}
    selected={m:min(range(len(GRID)),key=lambda i:(-round(combined[m][i]['mean_clean_BA'],12),
        round(combined[m][i]['mean_balanced_logloss'],12),i)) for m in METHODS}
    assert fingerprints(name)==hashes
    saved={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'inner':inner,
        'selection':{'options':combined,'selected':selected},'complete':True,'goal_achieved':False,
        'new_outer_evaluation_performed':False,'query_labels_used_for_fit':False,
        'scope':'all15_exposed_development_shared_crossgroup_inner_validation_diagnostic_not_prediction_superiority'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'COMPLETE inner only',combined['proposed'][selected['proposed']]['mean_clean_BA']-combined['proposed'][0]['mean_clean_BA'],flush=True)


def summarize(names):
    rows=[];failures=0
    for name in sorted(names):
        s=json.loads((FOLDER/(name+'.json')).read_text());op=s['selection']['options'];sel=s['selection']['selected']
        rows.append({'dataset':name,'selected':sel,
            'anchor_BA':{m:op[m][0]['mean_clean_BA'] for m in METHODS},
            'selected_inner_BA':{m:op[m][sel[m]]['mean_clean_BA'] for m in METHODS},
            'fixed_grid_inner_BA':{m:[r['mean_clean_BA'] for r in op[m]] for m in METHODS}})
        failures+=sum(not r['fit']['optimizer_success'] for b in s['inner'] for r in b['stage']['learners'])
    means={m:{'anchor':float(np.mean([r['anchor_BA'][m] for r in rows])),
        'selected_inner':float(np.mean([r['selected_inner_BA'][m] for r in rows])),
        'fixed_grid':np.mean([r['fixed_grid_inner_BA'][m] for r in rows],axis=0).tolist()} for m in METHODS}
    output={'rows':rows,'means':means,'target_optimizer_failures':failures,'goal_achieved':False,
        'independent_audit_required':True,'new_outer_evaluation_performed':False,
        'limitation':'grid contains anchor; selected inner nonworsening is tautological, not generalization evidence',
        'scope':'all15_inner_validation_diagnostic_no_new_outer_predictions'}
    (FOLDER/'complete.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print({k:v for k,v in output.items() if k!='rows'},flush=True)


if __name__=='__main__':
    names=sorted(verify(SOURCE)['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=1) as executor:
        for future in as_completed([executor.submit(run,name) for name in names]):future.result()
    summarize(names)
