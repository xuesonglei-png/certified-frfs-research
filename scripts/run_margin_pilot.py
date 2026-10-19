"""Third-cycle fixed first-fold pilot: robust LOO decision preservation."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import available_datasets,load_dataset
from certified_frfs.selection import deletion_order
from certified_frfs.core import _distance
from certified_frfs.margin import MarginWorkspace

OUTPUT=ROOT/'results/round3/margin-pilot.json'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x))
def hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),ROOT/'research/round3/protocol.md',*sorted((ROOT/'src/certified_frfs').glob('*.py'))]}

def nominal_select(X,y,order,protected):
    features=list(range(X.shape[1])); w=np.ones(X.shape[1])/X.shape[1]
    cross=y[:,None]!=y[None,:]
    same=(~cross)&(~np.eye(len(y),dtype=bool))
    history=[]
    if protected.any():
        for k in order:
            if len(features)==1: break
            candidate=[j for j in features if j!=k]
            dist=_distance(X,candidate,w)
            margin=np.where(cross,dist,np.inf).min(axis=1)-np.where(same,dist,np.inf).min(axis=1)
            accepted=bool(np.all(margin[protected]>1e-10))
            history.append({'removed_candidate':k,'accepted':accepted})
            if accepted: features=candidate
    return dict(features=features,history=history)

def one(name):
    start=perf_counter()
    source=ROOT/'results/primary'/f'{name}-s17-f0.json'
    old=json.loads(source.read_text(encoding='utf-8'))
    X,y=load_dataset(name); train,test=np.array(old['train_indices']),np.array(old['test_indices'])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=MinMaxScaler(clip=True)
    Xtr=scaler.fit_transform(imputer.fit_transform(X[train])); Xte=scaler.transform(imputer.transform(X[test]))
    for a,b in [(scaler.data_min_,old['scaler_min']),(scaler.data_max_,old['scaler_max']),
                (imputer.statistics_,old['imputer_statistics'])]: np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)
    ytr,yte=y[train],y[test]; d=X.shape[1]
    order=deletion_order(Xtr,ytr,17)
    workspace=MarginWorkspace(Xtr,ytr,eps=.01,n_friends=3)
    selected=workspace.select(order)
    protected=np.zeros(len(train),bool); protected[selected['protected_indices']]=True
    nominal=nominal_select(Xtr,ytr,order,protected)
    selected['selection_seconds']=perf_counter()-start
    predictors={'all':list(range(d)),'certified_margin':selected['features'],
                'mi_matched':sorted(order[-len(selected['features']):]),'nominal_margin':nominal['features']}
    models={}
    for key,fs in predictors.items():
        models[key,'l1_1nn']=KNeighborsClassifier(n_neighbors=1,metric='manhattan').fit(Xtr[:,fs],ytr)
        models[key,'euclidean_5nn']=KNeighborsClassifier(n_neighbors=5).fit(Xtr[:,fs],ytr)
    rows=[]
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')+17)
    for noise in [0.,.01,.05]:
        for repeat in range(1 if noise==0 else 5):
            Z=Xte if noise==0 else np.clip(Xte+rng.uniform(-noise,noise,Xte.shape),0.,1.)
            for (key,clf),model in models.items():
                pred=model.predict(Z[:,predictors[key]])
                rows.append(dict(method=key,classifier=clf,test_epsilon=noise,repetition=repeat,
                                 balanced_accuracy=float(balanced_accuracy_score(yte,pred)),
                                 macro_f1=float(f1_score(yte,pred,average='macro',zero_division=0))))
    del workspace
    result=dict(dataset=name,d=d,n_train=len(train),n_test=len(test),train_indices=train,test_indices=test,
        weights=np.ones(d)/d,selected=selected,nominal=nominal,mi_order=order,predictors=predictors,evaluations=rows,
        removed_fraction=1-len(selected['features'])/d,source_run_sha256=sha(source),
        data_sha256=sha(ROOT/'data/processed'/f'{name}.npz'),seconds=perf_counter()-start)
    print(f"{name}: {len(selected['features'])}/{d}, protected={selected['protected_fraction']:.3f}, minclass={selected['min_class_protected_fraction']:.3f}",flush=True)
    return result

def summarize(rows):
    out=dict(datasets=len(rows),mean_removed_fraction=float(np.mean([r['removed_fraction'] for r in rows])),
             mean_protected_fraction=float(np.mean([r['selected']['protected_fraction'] for r in rows])),
             mean_min_class_protected_fraction=float(np.mean([r['selected']['min_class_protected_fraction'] for r in rows])),
             zero_protection_datasets=[r['dataset'] for r in rows if not r['selected']['protected_indices']],
             zero_coverage_classes=[{'dataset':r['dataset'],**c} for r in rows for c in r['selected']['coverage'] if c['protected']==0],
             predictive=[])
    for clf in ['l1_1nn','euclidean_5nn']:
        for key in ['all','certified_margin','mi_matched','nominal_margin']:
            for noise in [0.,.01,.05]:
                values=[np.mean([v['balanced_accuracy'] for v in r['evaluations'] if v['classifier']==clf and v['method']==key and v['test_epsilon']==noise]) for r in rows]
                out['predictive'].append(dict(classifier=clf,method=key,test_epsilon=noise,mean_balanced_accuracy=float(np.mean(values))))
    return out

def main():
    start=perf_counter(); before=hashes()
    result=dict(protocol='margin-first-fold-2026-09-20-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes_before=before,eps=.01,n_friends=3,weights='fixed full table 1/d',rows=[],
        scope='post hoc new target; preserve full-certified robust LOO objects; not held-out accuracy guarantee')
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    try:
        with threadpool_limits(limits=1):
            for name in available_datasets():
                result['rows'].append(one(name)); result['run_status']='running'
                OUTPUT.write_text(json.dumps(result,indent=2,default=plain,allow_nan=False),encoding='utf-8')
        result['source_hashes_after']=hashes()
        assert result['source_hashes_after']==before,'source changed during run'
        result.update(run_status='complete',summary=summarize(result['rows']),seconds=perf_counter()-start)
    except BaseException as err:
        result.update(run_status='failed',error=f'{type(err).__name__}: {err}',seconds=perf_counter()-start)
        OUTPUT.write_text(json.dumps(result,indent=2,default=plain,allow_nan=False),encoding='utf-8')
        raise
    OUTPUT.write_text(json.dumps(result,indent=2,default=plain,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__': main()
