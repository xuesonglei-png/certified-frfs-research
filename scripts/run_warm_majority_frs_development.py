"""ALL15 matched direct-majority metric training and genuine predictions."""
import argparse
import copy
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_fixed_face_frs_development_v2 import ROOT,PARENTS,prerequisites,digest,compact_metric,NATIVES
from run_frs_joint_zero_face_prediction import alignment
from run_frs_zero_face_repair_experiment import builtin
from fit_warm_majority_face_milp import fit_warm_majority_face_milp
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/warm-majority-frs-development'
FREEZE=ROOT/'research/warm-majority-frs-development-source-freeze.json'
ARMS=('direct_control','direct_face')
METHODS=('anchor','full_uniform','joint_pair')+ARMS+NATIVES


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    return hashes


def row(saved):
    return {'dataset':saved['dataset'],'d':saved['d'],'available':saved['available'],
        'fixed_deletions':saved['fixed_zero_indices'],
        'metrics':{m:r['evaluation']['metrics'] for m,r in saved['results'].items()},
        'actual_feature_counts':{m:len(v['features']) for m,v in saved['metrics'].items()},
        'training':{a:None if not saved['available'] else {k:saved['fits'][a][k] for k in
            ('accepted_descent','status','mip_status','mip_gap_solver_reported','initial_objective','final_objective','fit_seconds')} for a in ARMS}}


def run(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True);path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['complete'] and saved['fingerprints']==hashes;return
    source_hashes={n:digest(p/(name+'.json')) for n,p in PARENTS.items()}
    assert all(source_hashes[n]==parents[n]['artifact_hashes'][name] for n in PARENTS)
    joint=json.loads((PARENTS['frs-joint-zero-face']/(name+'.json')).read_text())
    native=json.loads((PARENTS['majority-hinge-canonical-benchmark']/(name+'.json')).read_text())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text());first=alignment(joint,native)
    fs,anchor=joint['original_support'],np.asarray(joint['original_anchor']);available=joint['available']
    zeros=joint['eliminations']['pair_transition']['exact_zero_indices'] if available else []
    seed=np.asarray(joint['eliminations']['pair_transition']['target']) if available else anchor
    X,y=load_dataset(name);ri=np.asarray(joint['reference_indices']);ca=np.asarray(joint['training_indices']);cb=np.asarray(first['validation_indices'])
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);labels=y[ri][ca];fits={};reuse={}
    if available:
        for arm in ARMS:
            partial=FOLDER/(name+'.'+arm+'.partial.json')
            if partial.exists():
                old=json.loads(partial.read_text());assert old['fingerprints']==hashes and old['source_hashes']==source_hashes
                fits[arm]=old['fit'];reuse[arm]=old['reuse'];continue
            if arm=='direct_face' and not zeros:
                fit=copy.deepcopy(fits['direct_control']);reuse[arm]='identical_empty_face'
            else:
                facial=arm=='direct_face'
                _,fit=fit_warm_majority_face_milp(A,labels,fs,anchor,full_reference=np.asarray(joint['geometry']['cell']['full_weights']),
                    seed=seed if facial else anchor,fixed_zero_indices=zeros if facial else [],time_limit=120.)
                assert fit['class_cell']==joint['geometry']['cell'];reuse[arm]=None
            fits[arm]=fit
            partial.write_text(json.dumps({'fingerprints':hashes,'source_hashes':source_hashes,'fit':fit,'reuse':reuse[arm]},default=builtin,separators=(',',':')),encoding='utf-8')
            print(name,arm,{k:fit[k] for k in ('status','mip_status','initial_objective','final_objective','fit_seconds')},flush=True)
    metrics={m:previous['metrics'][m] for m in ('anchor','full_uniform')+NATIVES};metrics['joint_pair']=previous['metrics']['joint_pair_transition']
    for arm in ARMS:metrics[arm]=compact_metric(fs,fits[arm]['weights'] if available else anchor)
    metrics={m:metrics[m] for m in METHODS};cached_map={m:m for m in ('anchor','full_uniform')+NATIVES};cached_map['joint_pair']='joint_pair_transition'
    results={};prediction_reuse={}
    for method in METHODS:
        if method in cached_map:
            src=cached_map[method];assert metrics[method]==previous['metrics'][src]
            results[method]=previous['results'][src];prediction_reuse[method]='audited_joint_prediction:'+src
        else:
            same=next((m for m in results if metrics[m]==metrics[method]),None)
            if same is None:results[method]=final_evaluation(A,labels,B,y[ri][cb],metrics[method]);prediction_reuse[method]=None
            else:results[method]=results[same];prediction_reuse[method]=same
    saved={'dataset':name,'d':X.shape[1],'fold':0,'reference_indices':ri.tolist(),'training_indices':ca.tolist(),'validation_indices':cb.tolist(),
        'available':available,'fixed_zero_indices':zeros,'face_seed':seed.tolist(),'source_hashes':source_hashes,'fingerprints':hashes,
        'fits':fits,'fit_reuse':reuse,'metrics':metrics,'results':results,'prediction_reuse':prediction_reuse,
        'complete':True,'classification_scored':True,'query_labels_used_for_new_endpoint_selection':False,
        'new_outer_evaluation_performed':False,'classification_superiority_established':False,'goal_achieved':False}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'actual direct-majority first-inner BA',{m:results[m]['evaluation']['metrics']['clean_BA'] for m in METHODS},flush=True)


def summarize(names):
    rows=[row(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS};contrasts={}
    for control in ('direct_control','joint_pair','anchor','fgs_weighted_native120'):
        d=[100*(r['metrics']['direct_face']['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
        contrasts[control]={'mean_BA_delta_pp':float(np.mean(d)),'wins':sum(v>1e-10 for v in d),'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)}
    out={'rows':rows,'datasets':len(names),'means':means,'contrasts':contrasts,
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_superiority_established':False,'goal_achieved':False,
        'limits':['All15 first inner data exposed; no independent superiority or significance.',
            'Training majority objective differs from multiclass plurality and generalization error.',
            'Time-limit MILP incumbent and solver-reported bound are not independently proven global optimality.',
            'Matched full support and full native candidate families preserved.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:run(args.dataset)
    else:
        names=sorted(prerequisites()['frs-joint-zero-face']['artifact_hashes'])
        for name in names:run(name)
        summarize(names)
