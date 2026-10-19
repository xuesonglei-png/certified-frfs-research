"""Run both frozen selection rules; checkpoint and retain all six datasets."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.selective_support_selector_v3 import query_scores
from certified_frfs.relation_support_selector_v1 import weighted_knn
from repair_relation_selector_v4 import relations
from anchor_guarded_selector_v5 import RULES,choose,refit
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest,freeze as freeze_parent

FOLDER=ROOT/'results/anchor-guarded-selector-v5'

def freeze():
    parent=freeze_parent();files=[ROOT/'scripts'/p for p in ('anchor_guarded_selector_v5.py',
        'run_anchor_guarded_selector_v5.py','audit_anchor_guarded_selector_v5.py','summarize_anchor_guarded_selector_v5.py')]
    files += [ROOT/'research/anchor-guarded-selector-v5-protocol.md']
    files += [OUT/(n+'.json') for n in NAMES]
    hashes={**parent,**{p.relative_to(ROOT).as_posix():digest(p) for p in files}}
    FOLDER.mkdir(exist_ok=True);p=FOLDER/'source-freeze.json'
    if p.exists():assert json.loads(p.read_text())['hashes']==hashes
    else:p.write_text(json.dumps({'hashes':hashes,'primary':'paired_anchor','rules':RULES,
            'created_unix':time.time()},indent=2))
    return hashes

def run(name):
    threadpool_limits(1);hashes=freeze();source=json.loads((OUT/(name+'.json')).read_text());assert source['complete']
    p=FOLDER/(name+'.json');saved=json.loads(p.read_text()) if p.exists() else None
    if saved:
        assert saved['source_hashes']==hashes
        if saved['complete']:return
    rows=saved['rows'] if saved else [];X,y=load_dataset(name);start=time.perf_counter()
    for old in source['rows']:
        fold=old['fold']
        if fold<len(rows):continue
        ii,jj=np.array(old['train_indices']),np.array(old['test_indices']);A,B=preprocess_fit(X[ii],X[jj])
        fitted=relations(A,y[ii]);fits={};models={};cache={};scores={}
        for rule in RULES:
            pool,chosen=choose(old['fit'],rule);key=(chosen['family'],chosen['recipe_index'])
            if key not in cache:cache[key]=refit(X[ii],y[ii],chosen,20261011+fold,fitted=fitted)
            f={**cache[key],'chosen':chosen,'inner_options':pool};fits[rule]=f;models[rule]=f['selected']
            scores[rule]=query_scores(A,y[ii],B,y[jj],models[rule])
        noise={r:[] for r in RULES}
        for draw in range(8):
            rng=np.random.default_rng(20262000+10*fold+draw)
            na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            for rule,c in models.items():
                s=weighted_knn(na,y[ii],nb,y[jj],c['features'],c['weights'])
                assert not np.any(np.array(scores[rule]['query_certificate']['certified']) & (np.array(s['predictions'])!=scores[rule]['predictions']))
                noise[rule].append(s)
        rows.append({'fold':fold,'train_indices':ii.tolist(),'test_indices':jj.tolist(),'fits':fits,'scores':scores,
            'full':old['full'],'previous_v4':old['proposed'],'shared_perturbations':noise,
            'perturbed_BA':{r:float(np.mean([s['clean_BA'] for s in vals])) for r,vals in noise.items()}})
        assert freeze()==hashes
        result={'dataset':name,'d':A.shape[1],'rows':rows,'complete':len(rows)==3,'source_hashes':hashes,
            'seconds':time.perf_counter()-start,'primary':'paired_anchor','scope':'exposed_nested_development_not_confirmation'}
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));tmp.replace(p)
        print(name,fold,{r:{'BA':round(100*scores[r]['clean_BA'],3),
            'family':fits[r]['chosen']['family'],'count':len(models[r]['features'])} for r in RULES},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):run(n)
