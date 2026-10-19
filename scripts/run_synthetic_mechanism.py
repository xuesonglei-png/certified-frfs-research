"""Prespecified mechanism diagnostic, not evidence replacing real datasets."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.crossfit_search import select_crossfit_queries
from certified_frfs.baselines import select_mi, select_relieff
from certified_frfs.query_certificate import certify_queries
threadpool_limits(1)


def generate(seed,regime):
    rng=np.random.default_rng(seed)
    n,d=240,40
    y=rng.integers(0,2,n)
    X=rng.uniform(0,1,(n,d))
    if regime in ['strong','weak']:
        gap=.65 if regime=='strong' else .12
        X[:,:4]=np.clip(.5+gap*(y[:,None]-.5)+rng.normal(0,.12,(n,4)),0,1)
    else:
        first=rng.integers(0,2,n)
        second=first^y
        X[:,0]=np.clip(.15+.7*first+rng.normal(0,.03,n),0,1)
        X[:,1]=np.clip(.15+.7*second+rng.normal(0,.03,n),0,1)
    return X[:120],y[:120],X[120:],y[120:]


records=[]
for regime in ['strong','weak','xor']:
    for seed in range(10):
        A,y,B,truth=generate(seed,regime)
        selected={}
        for mode in ['robust','nominal','accuracy']:
            result=select_crossfit_queries(A,y,mode=mode,rank='relief',
                constraint_tolerance=.01,penalty=.02,budget=24,seed=101)
            selected[mode]=result['features']
        k=len(selected['robust'])
        selected.update({'all':list(range(40)),'MI_matched_k':select_mi(A,y,k),
                         'Relief_matched_k':select_relieff(A,y,k),
                         'generating_features_oracle':[0,1] if regime=='xor' else [0,1,2,3]})
        for method,fs in selected.items():
            model=KNeighborsClassifier(n_neighbors=1,metric='manhattan',algorithm='brute').fit(A[:,fs],y)
            pred=model.predict(B[:,fs])
            cert=certify_queries(A,y,B,pred,fs,eps=.01)
            mask=np.asarray(cert['certified'])
            perturbed=np.clip(B+np.random.default_rng(seed+200).uniform(-.01,.01,B.shape),0,1)
            row={'regime':regime,'seed':seed,'method':method,'features':fs,
                 'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
                 'perturbed_balanced_accuracy':float(balanced_accuracy_score(truth,model.predict(perturbed[:,fs]))),
                 'query_coverage':cert['coverage'],
                 'certified_wrong':float(np.mean(mask&(pred!=truth))),
                 'removed':1-len(fs)/40}
            records.append(row)
        print(json.dumps({'regime':regime,'seed':seed,'status':'complete'}),flush=True)
out=ROOT/'results/synthetic-mechanism'
out.mkdir(parents=True,exist_ok=True)
summary={regime:{method:{metric:float(np.mean([r[metric] for r in records if r['regime']==regime and r['method']==method]))
    for metric in ['balanced_accuracy','perturbed_balanced_accuracy','query_coverage','certified_wrong','removed']}
    for method in selected} for regime in ['strong','weak','xor']}
(out/'results.json').write_text(json.dumps({'status':'development_synthetic_mechanism_not_real_data_confirmation',
    'records':records,'means':summary},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
