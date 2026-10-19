from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from compare_trajectory_recent import GRIDS, SELECTORS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_query_descent import fit_feasible_query
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension

FOLDER=ROOT/'results/nested-classifier-pipelines'
KINDS=('rbf_svm','logistic'); CS=(.1,1.,10.)


def candidates(X,y):
    trajectory=fit_feasible_query(X,y,steps=10,certified=False,weight_cap_ratio=2)
    certified=certify_capped_trajectory(X,y,trajectory)
    sparse=sparse_trajectory_extension(X,y,certified)
    return representations(sparse['compact_selected'],certified['selected'],X.shape[1])


def representations(compact,dense,d):
    def learned(c):
        ids=c['features'];w=np.array(c['weights'])[ids]
        return ids,w/w.mean()
    return {'proposed_uniform':(compact['features'],np.ones(len(compact['features']))),
            'proposed_weighted':learned(compact),'dense_weighted':learned(dense),
            'full':(list(range(d)),np.ones(d))}


def predict(kind,C,A,y,B,ids,w):
    if not ids:
        return DummyClassifier(strategy='prior').fit(np.zeros((len(y),1)),y).predict(np.zeros((len(B),1)))
    model=SVC(C=C,gamma='scale') if kind=='rbf_svm' else LogisticRegression(C=C,max_iter=2000,random_state=0)
    return model.fit(A[:,ids]*w,y).predict(B[:,ids]*w)


def fingerprints(name):
    paths=[Path(__file__),ROOT/'research/nested-classifier-pipeline-protocol.md',
           ROOT/'results/long-capped-sparse'/f'{name}.json',ROOT/'data/processed'/f'{name}.npz']
    paths.extend(sorted((ROOT/'src/certified_frfs').glob('*.py')))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1);hashes=fingerprints(name);out=FOLDER/f'{name}.json'
    if out.exists():
        assert json.loads(out.read_text())['fingerprints']==hashes
        return
    X,y=load_dataset(name);prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());rows=[]
    for index,s in enumerate(prior['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri]
        splits=QueryProbabilityObjective(raw,labels).splits
        trace={kind:{m:[{'C':C,'config':config,'folds':[]} for config in (GRIDS[m] if m in GRIDS else [{}]) for C in CS]
                     for m in ('proposed_uniform','proposed_weighted','dense_weighted','full',*GRIDS)} for kind in KINDS}
        inner_records=[]
        for j,fold in enumerate(splits):
            ti,qi=np.array(fold['reference']),np.array(fold['query']);A,B=preprocess_fit(raw[ti],raw[qi])
            reps=candidates(raw[ti],labels[ti]);baseline_reps={}
            for m,grid in GRIDS.items():
                for cfg in grid:
                    ids=SELECTORS[m](A,labels[ti],**cfg)['features']
                    baseline_reps[(m,json.dumps(cfg,sort_keys=True))]=(ids,np.ones(len(ids)))
            inner_records.append({'reference':ti.tolist(),'query':qi.tolist(),
                'representations':{m:{'features':ids,'weights':w.tolist()} for m,(ids,w) in reps.items()}})
            for kind in KINDS:
                for m,choices in trace[kind].items():
                    for choice in choices:
                        ids,w=baseline_reps[(m,json.dumps(choice['config'],sort_keys=True))] if m in GRIDS else reps[m]
                        value=float(balanced_accuracy_score(labels[qi],predict(kind,choice['C'],A,labels[ti],B,ids,w)))
                        choice['folds'].append({'clean':value,'features':ids})
            print(name,index,'inner',j,'complete',flush=True)
        A,B=preprocess_fit(raw,X[vi]);reps=representations(s['fit']['compact_selected'],s['certified_trajectory']['selected'],X.shape[1])
        rng=np.random.default_rng(144+index);versions=[(A,B)]
        for _ in range(3):
            versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        results={}
        for kind in KINDS:
            results[kind]={}
            for m,choices in trace[kind].items():
                for c in choices:
                    c['mean_clean']=float(np.mean([f['clean'] for f in c['folds']]))
                    c['mean_size']=float(np.mean([len(f['features']) for f in c['folds']]))
                chosen=max(range(len(choices)),key=lambda j:(choices[j]['mean_clean'],-choices[j]['mean_size'],-j));c=choices[chosen]
                if m in GRIDS:
                    ids=SELECTORS[m](A,labels,**c['config'])['features'];w=np.ones(len(ids))
                else:ids,w=reps[m]
                values=[float(balanced_accuracy_score(y[vi],predict(kind,c['C'],R,labels,Q,ids,w))) for R,Q in versions]
                results[kind][m]={'chosen_index':chosen,'features':ids,'weights':w.tolist(),
                    'metrics':{'clean':values[0],'noise01':float(np.mean(values[1:]))}}
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner_records':inner_records,
                     'trace':trace,'results':results})
        print(name,index,'external scoring complete',flush=True)
    assert fingerprints(name)==hashes
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':hashes},indent=2),encoding='utf-8')


def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());assert p['fingerprints']==fingerprints(name)
        means={kind:{m:{k:float(np.mean([s['results'][kind][m]['metrics'][k] for s in p['splits']]))
                      for k in ('clean','noise01')} for m in p['splits'][0]['results'][kind]} for kind in KINDS}
        rows.append({'dataset':name,'means':means,'removed':float(np.mean([1-len(s['results']['rbf_svm']['proposed_uniform']['features'])/p['d'] for s in p['splits']]))})
    means={kind:{m:{k:float(np.mean([r['means'][kind][m][k] for r in rows])) for k in ('clean','noise01')}
                 for m in rows[0]['means'][kind]} for kind in KINDS}
    comparisons={kind:{variant:{m:{'clean_gain':float(np.mean([r['means'][kind][variant]['clean']-r['means'][kind][m]['clean'] for r in rows])),
        'wins':sum(r['means'][kind][variant]['clean']>r['means'][kind][m]['clean']+1e-12 for r in rows)} for m in ('full','dense_weighted',*GRIDS)}
        for variant in ('proposed_uniform','proposed_weighted')} for kind in KINDS}
    removed=float(np.mean([r['removed'] for r in rows]))
    gates={kind:{v:removed>=.1 and all(comparisons[kind][v][m]['clean_gain']>=.01 and comparisons[kind][v][m]['wins']>=10 for m in GRIDS)
                 for v in ('proposed_uniform','proposed_weighted')} for kind in KINDS}
    result={'rows':rows,'means':means,'comparisons':comparisons,'mean_removed':removed,'gates':gates,
            'scope':'all15_all44_classifier_specific_nested_pipeline_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,n) for n in names]):future.result()
    summarize(names)
