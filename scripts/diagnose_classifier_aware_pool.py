"""Classifier-aware selection from archived certified pools, all exposed tasks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.classifier_aware_pool import _score
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.query_probability import QueryProbabilityObjective

FOLDER=ROOT/'results/classifier-aware-pool-v2'


def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
    controls=json.loads((ROOT/'results/nested-classifier-pipelines'/f'{name}.json').read_text())
    X,y=load_dataset(name);records=[]
    for index,s in enumerate(prior['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri]
        cs=[c for c in s['fit']['candidates'] if c['certificate']['feasible']];trace=[]
        folds=[(np.array(f['reference']),np.array(f['query'])) for f in QueryProbabilityObjective(raw,labels,seed=1701).splits]
        tables=[(ti,qi,*preprocess_fit(raw[ti],raw[qi])) for ti,qi in folds]
        for C in (.1,1.,10.):
            for j,c in enumerate(cs):
                scores=[_score(A,labels[ti],B,labels[qi],c['features'],C) for ti,qi,A,B in tables]
                trace.append({'candidate_index':j,'C':C,'mean_clean':float(np.mean(scores)),'folds':scores,'size':len(c['features'])})
        chosen=max(range(len(trace)),key=lambda j:(trace[j]['mean_clean'],-trace[j]['size'],-j));t=trace[chosen];selected=cs[t['candidate_index']]
        A,B=preprocess_fit(raw,X[vi]);fs=selected['features'];w=np.array(selected['weights']);d=X.shape[1]
        certificate=weight_certificate(A,labels,fs,np.ones(d)/np.sqrt(d),w[fs]);assert certificate['feasible']
        rng=np.random.default_rng(144+index);versions=[(A,B)]
        for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        values=[_score(R,labels,Q,y[vi],fs,t['C']) for R,Q in versions]
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'trace':trace,'chosen_index':chosen,
            'selected':selected,'certificate':certificate,'metrics':{'clean':values[0],'noise01':float(np.mean(values[1:]))},
            'controls':{m:v['metrics'] for m,v in controls['splits'][index]['results']['logistic'].items()}})
    paths=[Path(__file__),ROOT/'src/certified_frfs/classifier_aware_pool.py',ROOT/'research/classifier-aware-pool-protocol.md',
           ROOT/'results/long-capped-sparse'/f'{name}.json',ROOT/'results/nested-classifier-pipelines'/f'{name}.json']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':records,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')
    print(name,'complete',flush=True)


def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'removed':float(np.mean([1-len(r['selected']['features'])/p['d'] for r in s])),
            'control':{m:float(np.mean([r['controls'][m]['clean'] for r in s])) for m in s[0]['controls']}})
    comparisons={m:{'gain':float(np.mean([r['clean']-r['control'][m] for r in rows])),
                   'wins':sum(r['clean']>r['control'][m]+1e-12 for r in rows)} for m in rows[0]['control']}
    removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_removed':removed,'comparisons':comparisons,
        'gate_passed':removed>=.1 and comparisons['full']['gain']>=-.005 and all(comparisons[m]['gain']>=.005 and comparisons[m]['wins']>=10 for m in ('fgs2023','sfrip2025')),
        'scope':'adaptive_reference_pool_selection_all15_all44_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']];FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
