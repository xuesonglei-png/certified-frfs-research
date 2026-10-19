"""Prospectively frozen ALL15 development factorial; one arm per artifact."""
import argparse
import copy
import json
import time
import ijson
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from dynamic_certificate_metric_v3 import fit_dynamic_certificate
from run_fixed_face_frs_development_v2 import compact_metric,NATIVES
from geometric_vote_evaluation import final_evaluation
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

def builtin(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)

FOLDER=ROOT/'results/dynamic-certificate-development-v3'
FREEZE=ROOT/'research/dynamic-certificate-development-v3-source-freeze.json'
PARENTS={n:ROOT/'results'/n for n in ('frs-joint-zero-face','frs-joint-zero-face-prediction','majority-hinge-canonical-benchmark')}
ARMS=tuple(f'{cell}_{support}_{objective}' for objective in ('nominal','robust')
    for support in ('control','face') for cell in ('fixed','dynamic'))
METHODS=('anchor','full_uniform','joint_pair')+NATIVES+ARMS


def frozen():
    record=json.loads(FREEZE.read_text())
    for rel,sha in record['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    return record


def read_field(path,prefix):
    with path.open('rb') as stream:return next(ijson.items(stream,prefix,use_float=True))


def inputs(name,record):
    hashes={n:digest(p/(name+'.json')) for n,p in PARENTS.items()}
    for n,sha in hashes.items():assert sha==record['parents'][n]['artifact_hashes'][name]
    jp=PARENTS['frs-joint-zero-face']/(name+'.json')
    fields={k:read_field(jp,k) for k in ('available','d','original_support','original_anchor','reference_indices','training_indices')}
    if fields['available']:
        fields['zeros']=read_field(jp,'eliminations.pair_transition.exact_zero_indices')
        fields['seed']=read_field(jp,'eliminations.pair_transition.target')
        fields['full_reference']=read_field(jp,'geometry.cell.full_weights')
    else:fields.update(zeros=[],seed=fields['original_anchor'],full_reference=(np.ones(fields['d'])/np.sqrt(fields['d'])).tolist())
    previous=json.loads((PARENTS['frs-joint-zero-face-prediction']/(name+'.json')).read_text())
    for k in ('reference_indices','training_indices'):assert fields[k]==previous[k]
    assert not set(previous['training_indices'])&set(previous['validation_indices'])
    X,y=load_dataset(name);ri,ca,cb=[np.array(previous[k]) for k in ('reference_indices','training_indices','validation_indices')]
    A,B=preprocess_fit(X[ri][ca],X[ri][cb]);assert X.shape[1]==fields['d']
    return fields,previous,A,y[ri][ca],B,y[ri][cb],hashes


def run(name):
    threadpool_limits(1);freeze=frozen();FOLDER.mkdir(exist_ok=True);path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['complete'] and saved['freeze_sha256']==digest(FREEZE);return
    source,previous,A,y,B,truth,hashes=inputs(name,freeze);fs=source['original_support']
    anchor=np.array(source['original_anchor']);seed=np.array(source['seed']);zeros=source['zeros']
    metrics={m:previous['metrics']['joint_pair_transition' if m=='joint_pair' else m] for m in METHODS if m not in ARMS}
    results={m:previous['results']['joint_pair_transition' if m=='joint_pair' else m] for m in metrics}
    training={};fit_hashes={};reuse={}
    for arm in ARMS:
        partial=FOLDER/(name+'.'+arm+'.fit.json')
        if partial.exists():
            saved=json.loads(partial.read_text());assert saved['source_hashes']==hashes and saved['freeze_sha256']==digest(FREEZE)
            fit=saved['fit'];reuse[arm]=saved['reuse']
        else:
            cell,support,objective=arm.split('_');control=f'{cell}_control_{objective}'
            if not source['available']:
                fit={'available':False,'weights':seed.tolist(),'status':'source_certified_geometry_unavailable','fit_seconds':0.,'steps':[]};reuse[arm]=None
            elif support=='face' and not zeros:
                fit=copy.deepcopy(json.loads((FOLDER/(name+'.'+control+'.fit.json')).read_text())['fit']);reuse[arm]=control
            else:
                _,fit=fit_dynamic_certificate(A,y,fs,anchor,source['full_reference'],
                    seed if support=='face' else anchor,fixed_zeros=zeros if support=='face' else (),
                    refresh=cell=='dynamic',optimization_eps=0. if objective=='nominal' else .01,
                    max_steps=12,lp_time_limit=20.);reuse[arm]=None
            saved={'fit':fit,'reuse':reuse[arm],'source_hashes':hashes,'freeze_sha256':digest(FREEZE)}
            partial.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
        metrics[arm]=compact_metric(fs,fit['weights']);same=next((m for m in results if metrics[m]==metrics[arm]),None)
        results[arm]=copy.deepcopy(results[same]) if same else final_evaluation(A,y,B,truth,metrics[arm])
        fit_hashes[arm]=digest(partial)
        training[arm]={k:fit.get(k) for k in ('available','status','initial_objective','final_objective','accepted_steps','fit_seconds')}
        training[arm]['nonempty_refreshes']=sum(s.get('witness_refresh_changed',False) for s in fit['steps'])
        print(name,arm,training[arm], 'clean_BA',results[arm]['evaluation']['metrics']['clean_BA'],flush=True)
        del fit,saved
    out={'dataset':name,'d':source['d'],'source_available':source['available'],'source_hashes':hashes,
        'freeze_sha256':digest(FREEZE),'fit_hashes':fit_hashes,'fit_reuse':reuse,'training':training,
        'metrics':metrics,'results':results,'reference_indices':previous['reference_indices'],
        'training_indices':previous['training_indices'],'validation_indices':previous['validation_indices'],
        'complete':True,'heldout_labels_used_for_endpoint_selection':False,'new_outer_evaluation_performed':False,
        'goal_achieved':False,'classification_superiority_established':False}
    path.write_text(json.dumps(out,default=builtin,separators=(',',':')),encoding='utf-8')


def row(saved):
    return {'dataset':saved['dataset'],'d':saved['d'],'metrics':{m:r['evaluation']['metrics'] for m,r in saved['results'].items()},
        'actual_feature_counts':{m:len(v['features']) for m,v in saved['metrics'].items()},'training':saved['training']}


def summarize():
    freeze=frozen();names=sorted(freeze['parents']['frs-joint-zero-face']['artifact_hashes'])
    rows=[row(json.loads((FOLDER/(name+'.json')).read_text())) for name in names]
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    contrasts={}
    for arm in ARMS:
        cell,support,objective=arm.split('_');controls=set(('anchor','joint_pair','fgs_weighted_native120',f'fixed_{support}_{objective}',f'{cell}_control_{objective}'))-{arm}
        contrasts[arm]={}
        for control in sorted(controls):
            d=[100*(r['metrics'][arm]['clean_BA']-r['metrics'][control]['clean_BA']) for r in rows]
            contrasts[arm][control]={'mean_BA_delta_pp':float(np.mean(d)),
                'wins':sum(v>1e-10 for v in d),'losses':sum(v< -1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)}
    out={'datasets':len(rows),'rows':rows,'means':means,'contrasts':contrasts,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},
        'mean_actual_feature_counts':{m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},
        'goal_achieved':False,'classification_superiority_established':False,'independent_audit_required':True,
        'limits':['Repeatedly exposed ALL15 first-inner development; no significance or independent generalization claim.',
            'No endpoint is selected using these validation labels; all eight predeclared arms reported.',
            'Original full protection, original FRS reference and thresholds are retained; unavailable arms explicitly recorded.',
            'Local cell refresh is not exhaustive union search or global optimality; strict-majority sufficient certificates are finite-table statements.',
            'Complete native FGS/SFRIP candidate families and weighted variants unchanged.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');args=parser.parse_args()
    if args.dataset:run(args.dataset)
    else:summarize()
