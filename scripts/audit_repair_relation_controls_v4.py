"""Replay full baseline grids, nested choices, metrics, and common attacks.

Baseline graph/radius path construction reuses production implementations;
this is explicitly not independent verification of author fidelity.
"""
import argparse,json,time
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_relation_support_selector_v1 import equivalent,independent_evaluate
from audit_selective_support_selector_v3 import check_score
from run_repair_relation_controls_v4 import FOLDER,VARIANTS,candidate_block,metrics,select,simple_score,paths
from repair_relation_selector_v4 import local_soft_relation
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest,freeze

def audit(name):
    threadpool_limits(1);freeze();path=FOLDER/(name+'.json');sha=digest(path)
    saved=json.loads(path.read_text());parent=json.loads((OUT/(name+'.json')).read_text())
    assert saved['complete'] and saved['parent_sha256']==digest(OUT/(name+'.json'))
    X,y=load_dataset(name);counts=Counter();start=time.perf_counter()
    for row,old in zip(saved['rows'],parent['rows'],strict=True):
        assert row['train_indices']==old['train_indices'] and row['test_indices']==old['test_indices']
        ri,vi=np.array(row['train_indices']),np.array(row['test_indices']);raw,labels=X[ri],y[ri]
        assert not set(ri)&set(vi)
        for b,up in zip(row['inner'],old['fit']['inner'],strict=True):
            ii,jj=np.array(up['fit_indices']),np.array(up['validation_indices'])
            assert b['fit_indices']==ii.tolist() and b['validation_indices']==jj.tolist()
            A,B=preprocess_fit(raw[ii],raw[jj]);choice=old['fit']['chosen']
            expected_count=len(up['families'][choice['family']]['candidates'][choice['recipe_index']]['model']['features'])
            rebuilt=candidate_block(A,labels[ii],B,labels[jj],expected_count)
            for key,value in rebuilt.items():equivalent(b[key],value)
            checked=set()
            for method,options in b['options'].items():
                expected=120 if method.startswith('fgs_') else 6;assert len(options)==expected
                for item in options:
                    c=item['metric'];key=(tuple(c['features']),None if c['weights'] is None else tuple(c['weights']))
                    if c['weights'] is None:assert item['score'] is None;continue
                    if key in checked:continue
                    checked.add(key)
                    score=independent_evaluate(A,labels[ii],B,labels[jj],c['features'],c['weights']) if c['features'] else simple_score(A,labels[ii],B,labels[jj],c)
                    equivalent(item['score'],score);counts['independent_inner_metrics']+=1
            counts['full_grid_replays']+=1
        for method,selection in row['selection'].items():equivalent(selection,select(row['inner'],method))
        A,B=preprocess_fit(raw,X[vi]);refs=paths(A,labels);equivalent(row['outer_paths'],refs)
        soft,fit=local_soft_relation(A,labels);equivalent(row['local_soft_fit'],fit)
        actual_count=len(old['fit']['selected']['features'])
        for method,pick in row['selection'].items():
            if pick['chosen'] is None:assert row['models'][method] is None and row['scores'][method] is None;continue
            algorithm,variant=method.split('_',1);_,path_record=refs[algorithm][pick['chosen']['config_index']]
            expected=metrics(A,path_record,soft,actual_count)[variant];equivalent(row['models'][method],expected)
        seen=set()
        for method,c in row['models'].items():
            if c is None or c['weights'] is None:assert row['scores'][method] is None;continue
            key=(tuple(c['features']),tuple(c['weights']))
            if key not in seen:
                if c['features']:check_score(A,labels,B,y[vi],c,row['scores'][method],counts)
                else:equivalent(row['scores'][method],simple_score(A,labels,B,y[vi],c))
                seen.add(key)
            for draw in range(8):
                rng=np.random.default_rng(20262000+10*row['fold']+draw)
                na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
                score=independent_evaluate(na,labels,nb,y[vi],c['features'],c['weights']) if c['features'] else simple_score(na,labels,nb,y[vi],c)
                equivalent(row['shared_perturbations'][method][draw],score);counts['independent_shared_attacks']+=1
            equivalent(row['perturbed_BA'][method],float(np.mean([s['clean_BA'] for s in row['shared_perturbations'][method]])))
        counts['outer_folds']+=1;print(name,row['fold'],'recent controls audited',dict(counts),flush=True)
    freeze();assert digest(FOLDER/(name+'.json'))==sha
    result={'passed':True,'dataset':name,'artifact_sha256':sha,'counts':dict(counts),'seconds':time.perf_counter()-start,
        'auditor_sha256':digest(__file__),'limits':['baseline paths/learner use production replay, author fidelity is not independently established',
            'guarded full soft control training certification not independently replayed here',
            'exposed development, not confirmation']}
    (FOLDER/(name+'-audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):audit(n)
