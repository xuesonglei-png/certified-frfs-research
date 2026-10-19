"""Training, prediction and resumable per-arm artifacts on unchanged ALL15 folds."""
import argparse
import copy
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from run_frs_zero_face_repair_experiment import builtin
from fit_fixed_face_frs import fit_fixed_face_frs_target
from run_frs_joint_zero_face_prediction import alignment
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

PARENTS={n:ROOT/'results'/n for n in ('frs-joint-zero-face','majority-hinge-canonical-benchmark',
    'capped-frs-training','frs-joint-zero-face-prediction')}
FOLDER=ROOT/'results/fixed-face-frs-development'
FREEZE=ROOT/'research/fixed-face-frs-development-source-freeze.json'
ARMS=('full_support_robust','full_support_nominal','face_robust','face_nominal')
NATIVES=('fgs_native120','fgs_weighted_native120','sfrip_native6','sfrip_weighted_native6')
METHODS=('anchor','full_uniform','joint_pair')+ARMS+NATIVES


def prerequisites():
    out={n:verify(p) for n,p in PARENTS.items()}
    names=set(out['frs-joint-zero-face']['artifact_hashes'])
    assert len(names)==15 and all(set(v['artifact_hashes'])==names for v in out.values())
    return out


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():
        assert digest(ROOT/rel)==sha,rel
    return hashes


def adapt_robust_control(record):
    out=copy.deepcopy(record);fit=out['fit']
    assert fit['initialization_scope']=='anchor' and fit['initialization_accepted']
    fit.update(fixed_zero_indices=[],optimization_eps=.01,certificate_eps=.01)
    for step in fit['steps']:
        for trace in step['subproblem_trace']:
            trace['face_seed_blend']=trace.pop('anchor_blend')
    return out


def compact_metric(features,weights):
    assert len(features)==len(weights) and min(weights)>=0 and np.isfinite(weights).all()
    active=[i for i,w in enumerate(weights) if w!=0.]
    return {'features':[features[i] for i in active],'weights':[weights[i] for i in active]}


def row(saved):
    scores={m:r['evaluation']['metrics'] for m,r in saved['results'].items()}
    return {'dataset':saved['dataset'],'d':saved['d'],'available':saved['available'],
        'fixed_deletions':saved['fixed_zero_indices'], 'metrics':scores,
        'actual_feature_counts':{m:len(v['features']) for m,v in saved['metrics'].items()},
        'optimizer_failures':{a:saved['fits'][a]['fit']['subproblem_failures'] if saved['available'] else None for a in ARMS},
        'fit_seconds':{a:saved['fits'][a]['fit_seconds'] if saved['available'] else None for a in ARMS},
        'fit_reuse':saved['fit_reuse']}


def run(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True)
    path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['complete'] and saved['fingerprints']==hashes
        return
    source_hashes={n:digest(p/(name+'.json')) for n,p in PARENTS.items()}
    assert all(source_hashes[n]==parents[n]['artifact_hashes'][name] for n in PARENTS)
    joint=json.loads((PARENTS['frs-joint-zero-face']/(name+'.json')).read_text())
    native=json.loads((PARENTS['majority-hinge-canonical-benchmark']/(name+'.json')).read_text())
    old=json.loads((PARENTS['capped-frs-training']/(name+'.json')).read_text())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text())
    first=alignment(joint,native);block=old['inner_training'][0]
    assert old['reference_indices']==joint['reference_indices']
    assert block['reference_indices']==first['reference_indices']==joint['training_indices']
    fs,anchor=joint['original_support'],np.asarray(joint['original_anchor'])
    assert block['features']==fs and block['anchor']==anchor.tolist() and block['strict_cell_available']==joint['available']
    X,y=load_dataset(name);ri=np.asarray(joint['reference_indices']);ca=np.asarray(joint['training_indices']);cb=np.asarray(first['validation_indices'])
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);labels=y[ri][ca]
    zeros=joint['eliminations']['pair_transition']['exact_zero_indices'] if joint['available'] else []
    seed=np.asarray(joint['eliminations']['pair_transition']['target']) if joint['available'] else anchor.copy()
    fits={};reuse={}
    if joint['available']:
        for arm in ARMS:
            partial=FOLDER/(name+'.'+arm+'.partial.json')
            if partial.exists():
                cached=json.loads(partial.read_text())
                assert cached['fingerprints']==hashes and cached['source_hashes']==source_hashes
                fits[arm]=cached['record'];reuse[arm]=cached['reuse'];continue
            if arm=='full_support_robust':
                record=adapt_robust_control(block['fits']['anchor'])
                reuse[arm]='audited_capped_anchor_fit_only'
            elif arm.startswith('face_') and not zeros:
                control='full_support_'+arm.split('_')[1]
                record=copy.deepcopy(fits[control]);reuse[arm]=control+'_identical_empty_face'
            else:
                timer=time.perf_counter();eps=.01 if arm.endswith('robust') else 0.
                facial=arm.startswith('face_')
                point,fit=fit_fixed_face_frs_target(A,labels,fs,anchor,
                    full_reference=np.asarray(joint['geometry']['cell']['full_weights']),reference_scope='original_full',
                    initialization=seed if facial else anchor,initialization_scope='audited_joint_pair_face' if facial else 'anchor',
                    fixed_zero_indices=zeros if facial else [],optimization_eps=eps)
                assert fit['class_cell']==joint['geometry']['cell']
                record={'features':fs,'anchor':anchor.tolist(),'target':point.tolist(),'fit':fit,'fit_seconds':time.perf_counter()-timer}
                reuse[arm]=None
            fits[arm]=record
            partial.write_text(json.dumps({'fingerprints':hashes,'source_hashes':source_hashes,'record':record,'reuse':reuse[arm]},default=builtin,separators=(',',':')),encoding='utf-8')
            print(name,arm,'training endpoint',{'objective':record['fit']['training_objective_final'],
                'zeros':int(sum(v==0. for v in record['target'])),'failures':record['fit']['subproblem_failures'],'reuse':reuse[arm]},flush=True)
    metrics={m:previous['metrics'][m] for m in ('anchor','full_uniform')+NATIVES}
    metrics['joint_pair']=previous['metrics']['joint_pair_transition']
    for arm in ARMS:
        metrics[arm]=compact_metric(fs,fits[arm]['target'] if joint['available'] else anchor)
    metrics={m:metrics[m] for m in METHODS}
    cached_map={m:m for m in ('anchor','full_uniform')+NATIVES};cached_map['joint_pair']='joint_pair_transition'
    results={};prediction_reuse={};timer=time.perf_counter()
    for method in METHODS:
        if method in cached_map:
            src=cached_map[method];assert metrics[method]==previous['metrics'][src]
            results[method]=previous['results'][src];prediction_reuse[method]='audited_joint_prediction:'+src
        else:
            same=next((m for m in results if metrics[m]==metrics[method]),None)
            if same is not None:
                results[method]=results[same];prediction_reuse[method]=same
            else:
                results[method]=final_evaluation(A,labels,B,y[ri][cb],metrics[method]);prediction_reuse[method]=None
    saved={'dataset':name,'d':X.shape[1],'fold':0,'reference_indices':ri.tolist(),'training_indices':ca.tolist(),
        'validation_indices':cb.tolist(),'available':joint['available'],'fixed_zero_indices':zeros,'face_seed':seed.tolist(),
        'source_hashes':source_hashes,'fits':fits,'fit_reuse':reuse,'metrics':metrics,'results':results,
        'prediction_reuse':prediction_reuse,'prediction_seconds':time.perf_counter()-timer,'fingerprints':hashes,
        'complete':True,'classification_scored':True,'query_labels_used_for_new_endpoint_selection':False,
        'new_outer_evaluation_performed':False,'classification_superiority_established':False,'goal_achieved':False}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'actual first-inner BA',{m:round(results[m]['evaluation']['metrics']['clean_BA'],6) for m in METHODS},flush=True)


def summarize(names):
    rows=[]
    for name in names:
        saved=json.loads((FOLDER/(name+'.json')).read_text());rows.append(row(saved))
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    contrasts={}
    for facial in ('face_robust','face_nominal'):
        contrast={}
        for control in ('full_support_'+facial.split('_')[1],'joint_pair','anchor','fgs_weighted_native120'):
            d=np.array([100*(r['metrics'][facial]['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows])
            contrast[control]={'mean_BA_delta_pp':float(d.mean()),'wins':int(sum(d>1e-10)),
                'losses':int(sum(d< -1e-10)),'ties':int(sum(abs(d)<=1e-10))}
        contrasts[facial]=contrast
    out={'rows':rows,'datasets':len(names),'means':means,'contrasts':contrasts,
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'classification_scored':True,'classification_superiority_established':False,'new_outer_evaluation_performed':False,'goal_achieved':False,
        'limits':['All15 first-inner folds exposed: development only, no significance claim.',
            'Fixed face and original reference preserved; training descent is relative to the facial seed.',
            'Full-support controls use same original support and one anchor start.',
            'Strong native full candidate budgets retained; FD+/FCE+ proxy not represented as formal replication.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:run(args.dataset)
    else:
        names=sorted(prerequisites()['frs-joint-zero-face']['artifact_hashes'])
        for name in names:run(name)
        summarize(names)
