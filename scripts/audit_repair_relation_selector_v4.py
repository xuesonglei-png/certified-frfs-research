"""Separate guard, restoration, split, choice, prediction and noise replay."""
import argparse,json,time
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import _ranking
from repair_relation_selector_v4 import FAMILIES,FRACTIONS,relations
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest,freeze
from audit_selective_support_selector_v3 import check_candidate,check_score
from audit_relation_support_selector_v1 import equivalent,independent_evaluate

def replay_trace(A,y,reference,rank,requested,trace,model,counts):
    seen=set()
    for index,c in enumerate(trace):
        is_fallback=index==len(trace)-1 and model['uniform_fallback']
        count=A.shape[1] if is_fallback else requested+index
        fs=list(range(A.shape[1])) if is_fallback else sorted(rank[:count])
        ref=np.ones(A.shape[1])/np.sqrt(A.shape[1]) if is_fallback else reference
        key=(tuple(fs),tuple(c['weights']),tuple(ref))
        if key not in seen:check_candidate(A,y,ref,fs,c,counts);seen.add(key)
        if index<len(trace)-1:assert not c['joint_feasible']
    assert trace[-1]['joint_feasible']
    for k,v in trace[-1].items():equivalent(model[k],v)
    assert model['requested_count']==requested
    assert model['restored_count']==len(model['features'])-requested
    counts['restoration_recipes']+=1

def check_fit(X,y,fit,counts):
    blocks=fit['inner'];summary=[]
    splits=list(StratifiedKFold(fit['inner_splits'],shuffle=True,random_state=fit['seed']).split(X,y))
    assert len(blocks)==len(splits)
    for f,((ii,jj),b) in enumerate(zip(splits,blocks)):
        assert b['fit_indices']==ii.tolist() and b['validation_indices']==jj.tolist()
        assert not set(ii)&set(jj)
        A,B=preprocess_fit(X[ii],X[jj]);refs,learner=relations(A,y[ii])
        equivalent(b['learner'],learner);counts['learner_production_replays']+=1
        for family in FAMILIES:
            part=b['families'][family];equivalent(part['reference_weights'],refs[family].tolist())
            rank=_ranking(A,y[ii],refs[family],family,fit['seed']+f)
            assert part['ranking']==rank
            for i,frac in enumerate(FRACTIONS):
                c=part['candidates'][i];assert c['fraction']==frac
                requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(frac*A.shape[1])))
                replay_trace(A,y[ii],refs[family],rank,requested,c['trace'],c['model'],counts)
                check_score(A,y[ii],B,y[jj],c['model'],c['score'],counts)
        print('audited inner',f,dict(counts),flush=True)
    for family in FAMILIES:
        for i,frac in enumerate(FRACTIONS):
            cc=[b['families'][family]['candidates'][i] for b in blocks]
            summary.append({'family':family,'recipe_index':i,'fraction':frac,
                'mean_BA':float(np.mean([c['score']['clean_BA'] for c in cc])),
                'mean_certified_BA':float(np.mean([c['score']['query_certified_BA'] for c in cc])),
                'mean_count':float(np.mean([len(c['model']['features']) for c in cc])),
                'mean_logloss':float(np.mean([c['score']['balanced_logloss'] for c in cc]))})
    equivalent(fit['options'],summary)
    best=max(r['mean_BA'] for r in summary)
    chosen=max((r for r in summary if r['mean_BA']>=best-.005-1e-12),
        key=lambda r:(r['mean_certified_BA'],r['mean_BA'],-r['mean_count'],
                      -r['mean_logloss'],-FAMILIES.index(r['family']),-r['recipe_index']))
    equivalent(fit['chosen'],chosen)
    A,_=preprocess_fit(X,X);refs,learner=relations(A,y);family=chosen['family']
    equivalent(fit['learner'],learner);equivalent(fit['reference_weights'],refs[family].tolist())
    rank=_ranking(A,y,refs[family],family,fit['seed']+100);assert fit['ranking']==rank
    requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(chosen['fraction']*A.shape[1])))
    replay_trace(A,y,refs[family],rank,requested,fit['outer_trace'],fit['selected'],counts)
    counts['independent_selections']+=1

def audit(name):
    threadpool_limits(1);freeze();path=OUT/(name+'.json');sha=digest(path)
    saved=json.loads(path.read_text());assert saved['complete'];X,y=load_dataset(name);counts=Counter();start=time.perf_counter()
    for fold,((ii,jj),r) in enumerate(zip(StratifiedKFold(3,shuffle=True,random_state=20261011).split(X,y),saved['rows'])):
        assert r['fold']==fold and r['train_indices']==ii.tolist() and r['test_indices']==jj.tolist()
        check_fit(X[ii],y[ii],r['fit'],counts);A,B=preprocess_fit(X[ii],X[jj]);d=A.shape[1];c=r['fit']['selected']
        models={'proposed':c,'full':{'features':list(range(d)),'weights':np.ones(d)/np.sqrt(d)},
            'matched_uniform':{'features':c['features'],'weights':np.ones(len(c['features']))/np.sqrt(len(c['features']))}}
        for m,v in models.items():check_score(A,y[ii],B,y[jj],v,r[m],counts)
        for draw in range(8):
            rng=np.random.default_rng(20262000+10*fold+draw)
            na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            for m,v in models.items():
                equivalent(independent_evaluate(na,y[ii],nb,y[jj],v['features'],v['weights']),r['shared_perturbations'][m][draw])
                counts['independent_noise_predictions']+=1
        for m in models:equivalent(r['perturbed_BA'][m],float(np.mean([s['clean_BA'] for s in r['shared_perturbations'][m]])))
        counts['outer_folds']+=1
    assert digest(path)==sha;freeze()
    result={'passed':True,'dataset':name,'artifact_sha256':sha,'counts':dict(counts),
        'seconds':time.perf_counter()-start,'auditor_sha256':digest(__file__),
        'limits':['optimizer/ranking production replay, not independent global optimality',
                  'ordinary floating-point guarded certificates','exposed development, not confirmatory']}
    (OUT/(name+'-audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):audit(n)
