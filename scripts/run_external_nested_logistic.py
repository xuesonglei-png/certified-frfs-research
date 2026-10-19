"""Exact nested selector/classifier audit with outer and inner group safety."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_newdata_compact import DATA, SOURCES
from compare_trajectory_recent import GRIDS, SELECTORS
from compare_nested_classifier_pipelines import predict, CS
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension

FOLDER=ROOT/'results/external-nested-logistic'
METHODS=('proposed','full','trajectory',*GRIDS)


def dataset(name):
    with np.load(DATA/f'{name}.npz') as z:
        X,y,ids=z['X'],z['y'],z['source_indices']
    groups=None
    if name=='parkinsons':
        with (DATA/'parkinsons-parkinsons.data').open(encoding='utf-8-sig') as f:
            records=list(csv.DictReader(f))
        groups=np.array([r['name'].rsplit('_',1)[0] for r in records])[ids]
    if name=='breast_original':
        with (DATA/'breast_original-breast-cancer-wisconsin.data').open(encoding='utf-8-sig') as f:
            records=list(csv.reader(f))
        groups=np.array([r[0] for r in records])[ids]
    return X,y,groups


def folds(X,y,groups,*,outer):
    if groups is not None:
        splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=13101 if outer else 1013).split(X,y,groups))
        for ti,qi in splits:
            assert set(groups[ti]).isdisjoint(groups[qi])
            if len(np.unique(y[ti]))<2:
                raise ValueError('grouped reference has fewer than two classes')
        return splits
    if outer:
        return list(StratifiedKFold(3,shuffle=True,random_state=13101).split(X,y))
    return [(np.array(s['reference']),np.array(s['query'])) for s in QueryProbabilityObjective(X,y).splits]


def fit(X,y):
    trajectory=fit_feasible_query(X,y,steps=10,certified=False,weight_cap_ratio=2)
    certified=certify_capped_trajectory(X,y,trajectory)
    sparse=sparse_trajectory_extension(X,y,certified)
    return {'selected':sparse['compact_selected'],'trajectory_selected':certified['selected'],
            'steps':len(trajectory['trace'])-1,'training_splits':trajectory['training_splits']}


def score(A,y,B,truth,fs,C):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(truth,predict('logistic',C,A,y,B,fs,np.ones(len(fs)))))


def fingerprints(name):
    paths=[Path(__file__),ROOT/'scripts/compare_nested_classifier_pipelines.py',
           ROOT/'research/external-nested-logistic-protocol.md',DATA/f'{name}.npz',DATA/f'{name}-metadata.json']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    for p in DATA.glob(name+'-*'):
        if p not in paths:paths.append(p)
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1);hashes=fingerprints(name);out=FOLDER/f'{name}.json'
    if out.exists():assert json.loads(out.read_text())['fingerprints']==hashes;return
    X,y,groups=dataset(name);rows=[];d=X.shape[1]
    for fold,(tr,te) in enumerate(folds(X,y,groups,outer=True)):
        raw,labels=X[tr],y[tr];innergroups=None if groups is None else groups[tr]
        trace={m:[{'config':cfg,'C':C,'folds':[]} for cfg in (GRIDS[m] if m in GRIDS else [{}]) for C in CS] for m in METHODS}
        inner_records=[]
        for j,(ti,qi) in enumerate(folds(raw,labels,innergroups,outer=False)):
            A,B=preprocess_fit(raw[ti],raw[qi]);fitted=fit(raw[ti],labels[ti])
            reps={'proposed':fitted['selected']['features'],'trajectory':fitted['trajectory_selected']['features'],'full':list(range(d))}
            cache={}
            for m,grid in GRIDS.items():
                for cfg in grid:cache[(m,json.dumps(cfg,sort_keys=True))]=SELECTORS[m](A,labels[ti],**cfg)['features']
            for m,choices in trace.items():
                for c in choices:
                    fs=cache[(m,json.dumps(c['config'],sort_keys=True))] if m in GRIDS else reps[m]
                    c['folds'].append({'features':fs,'clean':score(A,labels[ti],B,labels[qi],fs,c['C'])})
            inner_records.append({'reference':ti.tolist(),'query':qi.tolist(),'fit':fitted})
            print(name,fold,'inner',j,'complete',flush=True)
        fitted=fit(raw,labels);A,B=preprocess_fit(raw,X[te])
        reps={'proposed':fitted['selected']['features'],'trajectory':fitted['trajectory_selected']['features'],'full':list(range(d))}
        rng=np.random.default_rng(144+fold);versions=[(A,B)]
        for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        results={}
        for m,choices in trace.items():
            for c in choices:
                c['mean_clean']=float(np.mean([r['clean'] for r in c['folds']]))
                c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
            chosen=max(range(len(choices)),key=lambda j:(choices[j]['mean_clean'],-choices[j]['mean_size'],-j));c=choices[chosen]
            fs=SELECTORS[m](A,labels,**c['config'])['features'] if m in GRIDS else reps[m]
            values=[score(R,labels,Q,y[te],fs,c['C']) for R,Q in versions]
            results[m]={'chosen_index':chosen,'features':fs,'metrics':{'clean':values[0],'noise01':float(np.mean(values[1:]))}}
        rows.append({'train_indices':tr.tolist(),'test_indices':te.tolist(),'inner_records':inner_records,
                     'fit':fitted,'trace':trace,'results':results})
        print(name,fold,'external complete',flush=True)
    assert fingerprints(name)==hashes
    out.write_text(json.dumps({'dataset':name,'d':d,'n':len(X),'groups':None if groups is None else groups.tolist(),
                              'splits':rows,'fingerprints':hashes},indent=2),encoding='utf-8')


def summarize():
    rows=[]
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text());assert p['fingerprints']==fingerprints(name)
        rows.append({'dataset':name,'means':{m:{k:float(np.mean([s['results'][m]['metrics'][k] for s in p['splits']])) for k in ('clean','noise01')} for m in METHODS},
                     'removed':float(np.mean([1-len(s['results']['proposed']['features'])/p['d'] for s in p['splits']])),
                     'full_fallbacks':sum(len(s['results']['proposed']['features'])==p['d'] for s in p['splits'])})
    comparisons={m:{'clean_gain':float(np.mean([r['means']['proposed']['clean']-r['means'][m]['clean'] for r in rows])),
                   'wins':sum(r['means']['proposed']['clean']>r['means'][m]['clean']+1e-12 for r in rows)} for m in METHODS[1:]}
    removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'comparisons':comparisons,'means':{m:{k:float(np.mean([r['means'][m][k] for r in rows])) for k in ('clean','noise01')} for m in METHODS},
            'mean_removed':removed,'gate_passed':removed>=.1 and comparisons['full']['clean_gain']>=-.005 and all(comparisons[m]['clean_gain']>=.005 and comparisons[m]['wins']>=6 for m in GRIDS),
            'scope':'eight_previously_exposed_tasks_exact_tenstep_nested_logistic_grouped_where_known_not_confirmation'}
    assert all(s['fit']['selected']['certificate']['feasible'] for name in SOURCES for s in json.loads((FOLDER/f'{name}.json').read_text())['splits'])
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in SOURCES]):f.result()
    summarize()
