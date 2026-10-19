"""Scalar replay of selected v5 prediction, original guards and anchor choice."""
import argparse,json,time
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from repair_relation_selector_v4 import FAMILIES,FRACTIONS,relations
from certified_frfs.relation_support_selector_v1 import _ranking
from run_repair_relation_selector_v4 import OUT,NAMES,digest
from run_anchor_guarded_selector_v5 import FOLDER,freeze
from anchor_guarded_selector_v5 import RULES
from audit_repair_relation_selector_v4 import replay_trace
from audit_selective_support_selector_v3 import check_score
from audit_relation_support_selector_v1 import equivalent,independent_evaluate

def independent_choice(fit,rule):
    anchor=[b['families']['uniform_sqrt']['candidates'][3]['score']['clean_BA'] for b in fit['inner']];pool=[]
    for family in FAMILIES:
        for i,fraction in enumerate(FRACTIONS):
            cc=[b['families'][family]['candidates'][i] for b in fit['inner']]
            ba=[c['score']['clean_BA'] for c in cc];delta=[a-b for a,b in zip(ba,anchor)]
            mean=sum(delta)/len(delta)
            stderr=(sum((v-mean)**2 for v in delta)/(len(delta)-1)/len(delta))**.5 if len(delta)>1 else 0.
            pool.append({'family':family,'recipe_index':i,'fraction':fraction,'mean_BA':sum(ba)/len(ba),
                'mean_certified_BA':sum(c['score']['query_certified_BA'] for c in cc)/len(cc),
                'mean_count':sum(len(c['model']['features']) for c in cc)/len(cc),
                'mean_logloss':sum(c['score']['balanced_logloss'] for c in cc)/len(cc),
                'paired_delta':delta,'paired_standard_error':stderr,'anchor_gate_passed':mean-stderr>=-1e-12})
    eligible=[p for p in pool if p['anchor_gate_passed']] if rule=='paired_anchor' else pool
    return pool,max(eligible,key=lambda r:(round(r['mean_BA'],12),round(r['mean_certified_BA'],12),
        -r['mean_count'],-round(r['mean_logloss'],12),-FAMILIES.index(r['family']),-r['recipe_index']))

def audit(name):
    threadpool_limits(1);hashes=freeze();p=FOLDER/(name+'.json');sha=digest(p);saved=json.loads(p.read_text());assert saved['complete']
    source=json.loads((OUT/(name+'.json')).read_text());X,y=load_dataset(name);counts=Counter();start=time.perf_counter()
    for r,old in zip(saved['rows'],source['rows'],strict=True):
        assert r['train_indices']==old['train_indices'] and r['test_indices']==old['test_indices']
        ii,jj=np.array(r['train_indices']),np.array(r['test_indices']);A,B=preprocess_fit(X[ii],X[jj]);refs,learner=relations(A,y[ii])
        for rule in RULES:
            pool,pick=independent_choice(old['fit'],rule);fit=r['fits'][rule]
            equivalent(fit['inner_options'],pool);equivalent(fit['chosen'],pick)
            equivalent(fit['learner'],learner);equivalent(fit['reference_weights'],refs[pick['family']].tolist())
            rank=_ranking(A,y[ii],refs[pick['family']],pick['family'],20261011+r['fold']+100);assert rank==fit['ranking']
            requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(A.shape[1]*pick['fraction'])))
            replay_trace(A,y[ii],refs[pick['family']],rank,requested,fit['outer_trace'],fit['selected'],counts)
            check_score(A,y[ii],B,y[jj],fit['selected'],r['scores'][rule],counts)
            for draw in range(8):
                rng=np.random.default_rng(20262000+10*r['fold']+draw)
                na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
                c=fit['selected'];equivalent(independent_evaluate(na,y[ii],nb,y[jj],c['features'],c['weights']),r['shared_perturbations'][rule][draw])
                counts['independent_noise_predictions']+=1
            equivalent(r['perturbed_BA'][rule],float(np.mean([s['clean_BA'] for s in r['shared_perturbations'][rule]])))
        counts['outer_folds']+=1
        print(name,r['fold'],'v5 guard/vote/selection replayed',dict(counts),flush=True)
    assert sha==digest(p) and hashes==freeze()
    report={'passed':True,'dataset':name,'artifact_sha256':sha,'counts':dict(counts),'seconds':time.perf_counter()-start,
        'auditor_sha256':digest(__file__),'parent_inner_audit_required':True,
        'limits':['parent inner evidence must separately pass its audit','optimizer/ranking production replay',
            'ordinary floats, not formal directed rounding','exposed data, not confirmation']}
    (FOLDER/(name+'-audit.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):audit(n)
