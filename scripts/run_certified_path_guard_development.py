"""Select from audited certified paths, then actual ALL15 classification."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_fixed_face_frs_development_v2 import ROOT,PARENTS,prerequisites,digest,NATIVES
from run_frs_joint_zero_face_prediction import alignment
from run_frs_zero_face_repair_experiment import builtin
from certified_path_training_guard import guard,ARMS
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/certified-path-guard-development'
FREEZE=ROOT/'research/certified-path-guard-development-source-freeze.json'
METHODS=('anchor','full_uniform','joint_pair')+ARMS+NATIVES


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    return hashes


def row(saved):
    return {'dataset':saved['dataset'],'d':saved['d'],'available':saved['available'],
        'candidate_count':len(saved['guard']['candidates']),
        'selected_path_steps':{a:saved['guard']['candidates'][saved['guard']['choices'][a]['selected_index']]['source_step'] for a in ARMS},
        'metrics':{m:r['evaluation']['metrics'] for m,r in saved['results'].items()},
        'actual_feature_counts':{m:len(v['features']) for m,v in saved['metrics'].items()},
        'additional_guard_seconds':saved['additional_guard_seconds']}


def run(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True);path=FOLDER/(name+'.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['complete'] and old['fingerprints']==hashes;return
    sources={n:digest(p/(name+'.json')) for n,p in PARENTS.items()}
    assert all(sources[n]==parents[n]['artifact_hashes'][name] for n in PARENTS)
    joint=json.loads((PARENTS['frs-joint-zero-face']/(name+'.json')).read_text())
    native=json.loads((PARENTS['majority-hinge-canonical-benchmark']/(name+'.json')).read_text())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text());first=alignment(joint,native)
    ri=np.asarray(joint['reference_indices']);ca=np.asarray(joint['training_indices']);cb=np.asarray(first['validation_indices'])
    assert not set(ca)&set(cb)
    X,y=load_dataset(name);A,B=preprocess_fit(X[ri][ca],X[ri][cb]);labels=y[ri][ca];timer=time.perf_counter()
    guarded=guard(A,labels,joint);guard_seconds=time.perf_counter()-timer
    metrics={m:previous['metrics'][m] for m in ('anchor','full_uniform')+NATIVES}
    metrics['joint_pair']=previous['metrics']['joint_pair_transition'];metrics.update(guarded['metrics']);metrics={m:metrics[m] for m in METHODS}
    cached={m:m for m in ('anchor','full_uniform')+NATIVES};cached['joint_pair']='joint_pair_transition';results={};reuse={}
    for method in METHODS:
        if method in cached:
            src=cached[method];assert metrics[method]==previous['metrics'][src]
            results[method]=previous['results'][src];reuse[method]='audited_joint_prediction:'+src
        else:
            same=next((m for m in results if metrics[m]==metrics[method]),None)
            if same is not None:results[method]=results[same];reuse[method]=same
            else:results[method]=final_evaluation(A,labels,B,y[ri][cb],metrics[method]);reuse[method]=None
    saved={'dataset':name,'d':X.shape[1],'fold':0,'reference_indices':ri.tolist(),'training_indices':ca.tolist(),'validation_indices':cb.tolist(),
        'available':joint['available'],'source_hashes':sources,'fingerprints':hashes,'guard':guarded,'metrics':metrics,'results':results,
        'prediction_reuse':reuse,'additional_guard_seconds':guard_seconds,'source_joint_four_regime_seconds':joint['fit_seconds'],
        'complete':True,'classification_scored':True,'query_labels_used_for_new_endpoint_selection':False,
        'new_outer_evaluation_performed':False,'classification_superiority_established':False,'goal_achieved':False}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'actual training-guard path first-inner BA',{m:results[m]['evaluation']['metrics']['clean_BA'] for m in METHODS},
        'supports',{m:len(metrics[m]['features']) for m in ('anchor','joint_pair')+ARMS},flush=True)


def summarize(names):
    rows=[row(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS};contrasts={}
    for arm in ARMS:
        contrasts[arm]={}
        for control in ('joint_pair','anchor','fgs_weighted_native120'):
            d=[100*(r['metrics'][arm]['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
            contrasts[arm][control]={'mean_BA_delta_pp':float(np.mean(d)),'wins':sum(v>1e-10 for v in d),'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)}
    out={'rows':rows,'datasets':len(names),'means':means,'contrasts':contrasts,
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_superiority_established':False,'goal_achieved':False,
        'limits':['All15 first-inner data repeatedly exposed; no independent superiority/significance.',
            'Class counts preserved only in enclosing training fold crossgroups; no validation accuracy guarantee.',
            'Guard selects an earlier already certified path endpoint; no new metric learning or optimal reduct claim.',
            'Additional guard runtime excludes prior four-regime LP path generation; no end-to-end efficiency claim.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:run(args.dataset)
    else:
        names=sorted(prerequisites()['frs-joint-zero-face']['artifact_hashes'])
        for name in names:run(name)
        summarize(names)
