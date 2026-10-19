"""Inner-only prospective mechanism test, with no outer prediction or scoring."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.selective_support_selector_v3 import query_scores
from repair_relation_selector_v4 import repaired,FRACTIONS
from graph_consensus_recipe_v6 import consensus_ranking
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest,freeze as freeze_parent
from audit_selective_support_selector_v3 import check_candidate,check_score
from audit_relation_support_selector_v1 import equivalent
from collections import Counter

FOLDER=ROOT/'results/graph-consensus-inner-v6'
FAMILIES=('uniform_sqrt','learned_majority','half_local_soft')

def freeze():
    hashes=freeze_parent();files=[ROOT/'scripts'/p for p in ('graph_consensus_recipe_v6.py','run_graph_consensus_inner_v6.py')]
    files += [ROOT/'research/graph-consensus-inner-v6-protocol.md']
    for n in NAMES:
        files += [OUT/(n+'.json'),ROOT/'results/repair-relation-controls-v4'/(n+'.json')]
    hashes.update({p.relative_to(ROOT).as_posix():digest(p) for p in files})
    FOLDER.mkdir(exist_ok=True);p=FOLDER/'source-freeze.json'
    if p.exists():assert json.loads(p.read_text())['hashes']==hashes
    else:p.write_text(json.dumps({'hashes':hashes,'created_unix':time.time(),
        'new_outer_scoring':False,'families':FAMILIES},indent=2))
    return hashes

def run(name):
    threadpool_limits(1);hashes=freeze();p=FOLDER/(name+'.json')
    saved=json.loads(p.read_text()) if p.exists() else None
    if saved:
        assert saved['source_hashes']==hashes
        if saved['complete']:return
    parent=json.loads((OUT/(name+'.json')).read_text())
    baseline=json.loads((ROOT/'results/repair-relation-controls-v4'/(name+'.json')).read_text())
    assert parent['complete'] and baseline['complete'];X,y=load_dataset(name);panels=saved['panels'] if saved else []
    start=time.perf_counter()
    for row,control in zip(parent['rows'],baseline['rows'],strict=True):
        if row['fold']<len(panels):continue
        assert row['train_indices']==control['train_indices']
        ri=np.array(row['train_indices']);raw,labels=X[ri],y[ri];blocks=[];counts=Counter()
        for b,cb in zip(row['fit']['inner'],control['inner'],strict=True):
            ii,jj=np.array(b['fit_indices']),np.array(b['validation_indices'])
            assert cb['fit_indices']==ii.tolist() and cb['validation_indices']==jj.tolist()
            A,B=preprocess_fit(raw[ii],raw[jj]);d=A.shape[1]
            rank,info=consensus_ranking(cb['raw_paths']['fgs'],d);families={}
            for family in FAMILIES:
                ref=np.array(b['families'][family]['reference_weights']);cache={};scored={};recipes=[];guard_checks=set()
                for fraction in FRACTIONS:
                    requested=max(int(np.ceil(d/2)),int(np.ceil(fraction*d)))
                    c,trace=repaired(A,labels[ii],ref,rank,requested,cache)
                    key=(tuple(c['features']),tuple(c['weights']))
                    if key not in scored:
                        scored[key]=query_scores(A,labels[ii],B,labels[jj],c)
                        check_score(A,labels[ii],B,labels[jj],c,scored[key],counts)
                    for candidate in trace:
                        fallback=len(candidate['features'])==d and c['uniform_fallback'] and candidate is trace[-1]
                        witness=np.ones(d)/np.sqrt(d) if fallback else ref
                        gkey=(tuple(candidate['features']),tuple(candidate['weights']),tuple(witness))
                        if gkey not in guard_checks:
                            check_candidate(A,labels[ii],witness,candidate['features'],candidate,counts);guard_checks.add(gkey)
                    recipes.append({'fraction':fraction,'model':c,'trace':trace,'score':scored[key]})
                families[family]={'reference_weights':ref.tolist(),'candidates':recipes}
            blocks.append({'fit_indices':ii.tolist(),'validation_indices':jj.tolist(),
                'consensus_ranking':rank,'consensus':info,'families':families})
            print(name,row['fold'],'inner',len(blocks),'graph candidates checked',dict(counts),flush=True)
        options=[]
        for family in FAMILIES:
            for i,fraction in enumerate(FRACTIONS):
                cc=[b['families'][family]['candidates'][i] for b in blocks]
                options.append({'family':family,'recipe_index':i,'fraction':fraction,
                    'mean_BA':float(np.mean([c['score']['clean_BA'] for c in cc])),
                    'mean_certified_BA':float(np.mean([c['score']['query_certified_BA'] for c in cc])),
                    'mean_count':float(np.mean([len(c['model']['features']) for c in cc]))})
        best=max(options,key=lambda r:(round(r['mean_BA'],12),round(r['mean_certified_BA'],12),-r['mean_count'],
            -FAMILIES.index(r['family']),-r['recipe_index']))
        old=max(row['fit']['options'],key=lambda r:r['mean_BA'])
        anchor=float(np.mean([b['families']['uniform_sqrt']['candidates'][3]['score']['clean_BA'] for b in row['fit']['inner']]))
        panels.append({'outer_training_fold':row['fold'],'inner':blocks,'options':options,'selected_training_only':best,
            'best_inner_delta_previous_pool_pp':100*(best['mean_BA']-old['mean_BA']),
            'best_inner_delta_full_anchor_pp':100*(best['mean_BA']-anchor),'independent_check_counts':dict(counts)})
        result={'dataset':name,'d':X.shape[1],'panels':panels,'source_hashes':hashes,'complete':len(panels)==3,
            'seconds':time.perf_counter()-start,'new_outer_scoring':False,'goal_achieved':False,
            'scope':'inner_mechanism_development_and_partial_scalar_replay_not_confirmatory'}
        assert freeze()==hashes
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));tmp.replace(p)
        print(name,row['fold'],'inner graph gain previous',panels[-1]['best_inner_delta_previous_pool_pp'],
            'full',panels[-1]['best_inner_delta_full_anchor_pp'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):run(n)
