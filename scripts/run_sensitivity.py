"""Prespecified first-split certificate ablations and parameter sensitivity."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from run_experiments import atomic_json
PROTOCOL_VERSION="2026-09-19-v2"

def fingerprint():
    paths=[Path(__file__),*sorted((ROOT/"src"/"certified_frfs").glob("*.py"))]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def run_dataset(name):
    os.environ.setdefault("OMP_NUM_THREADS","1")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score
    from certified_frfs.datasets import load_dataset
    from certified_frfs.selection import deletion_order,select_certified,select_nominal_backward
    from certified_frfs.core import certificate,gamma
    path=ROOT/"results"/"sensitivity"/(name+".json")
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"))
        if old.get("protocol_version") != PROTOCOL_VERSION or old.get("code_fingerprint") != fingerprint():
            raise ValueError("Stale sensitivity cache; archive the previous run first")
        return {"dataset":name,"status":"cached_verified"}
    start=time.perf_counter()
    X,y=load_dataset(name)
    splits=min(3,int(np.unique(y,return_counts=True)[1].min()))
    train,test=next(StratifiedKFold(splits,shuffle=True,random_state=17).split(X,y))
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    scaler=MinMaxScaler(clip=True)
    Xtr=scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte=scaler.transform(imputer.transform(X[test]))
    ytr,yte=y[train],y[test]
    d=Xtr.shape[1]
    order_start=time.perf_counter()
    order=deletion_order(Xtr,ytr,17)
    order_seconds=time.perf_counter()-order_start
    rows=[]
    cache={}
    def compute(eps,eta,method,scale):
        key=(eps,eta,method,scale)
        if key in cache: return cache[key]
        weights=np.full(d,{"sqrt":1/np.sqrt(d),"mean":1/d,"unit":1.0}[scale])
        t=time.perf_counter()
        if method=="nominal_backward":
            feats=select_nominal_backward(Xtr,ytr,eta=eta,weights=weights,order=order)
            cert=certificate(Xtr,ytr,feats,weights,eps)
        else:
            result=select_certified(Xtr,ytr,eps=eps,eta=eta,weights=weights,method=method,order=order)
            feats=result["features"]; cert=result["certificate"]
        seconds=time.perf_counter()-t+order_seconds
        model=KNeighborsClassifier(n_neighbors=5).fit(Xtr[:,feats],ytr)
        score=balanced_accuracy_score(yte,model.predict(Xte[:,feats]))
        row={"dataset":name,"eps":eps,"eta":eta,"method":method,"scale":scale,
             "features":feats,"n_features":len(feats),"d":d,"seconds":seconds,
             "knn_balanced_accuracy":score,"full_dependency":gamma(Xtr,ytr,weights=weights),
             "certificate":cert}
        cache[key]=row; rows.append(row)
        # Retain partial progress separately, never mark it complete.
        atomic_json(path.with_suffix(".partial.json"),{"rows":rows,"complete":False})
        return row
    for eps in [0.0,.005,.01,.02,.05]:
        for eta in [.05,.10,.20]: compute(eps,eta,"hybrid","sqrt")
    for method in ["independent","lipschitz","nominal_backward"]:
        compute(.01,.10,method,"sqrt")
    for scale in ["mean","unit"]:
        compute(.01,.10,"hybrid",scale)
    atomic_json(path,{"dataset":name,"rows":rows,"complete":True,
                      "total_seconds":time.perf_counter()-start,"seed":17,"fold":0,
                      "protocol_version":PROTOCOL_VERSION,"order_seconds":order_seconds,
                      "code_fingerprint":fingerprint()})
    return {"dataset":name,"status":"complete","rows":len(rows),"seconds":round(time.perf_counter()-start,2)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="*")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    from certified_frfs.datasets import available_datasets
    names=args.datasets or available_datasets()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_dataset,name):name for name in names}
        for future in concurrent.futures.as_completed(futures):
            try: print(json.dumps(future.result()),flush=True)
            except Exception as exc:
                print(json.dumps({"dataset":futures[future],"error":repr(exc)}),flush=True)
                raise
