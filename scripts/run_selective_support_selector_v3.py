"""Versioned selective-support experiment with pre-run source fingerprints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.selective_support_selector_v3 import fit_outer,query_scores
from certified_frfs.validated_pool import preprocess_fit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/selective-support-selector-v3"
DATASETS=("banknote","glass","heart_statlog","ionosphere","seeds","wdbc")


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprints():
    files=list((ROOT/"src/certified_frfs").glob("*.py"))+[Path(__file__)]
    files += [ROOT/"data/processed"/(name+".npz") for name in DATASETS]
    files += [ROOT/"research/selective-support-selector-v3-protocol.md"]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def freeze():
    OUT.mkdir(exist_ok=True); p=OUT/"source-freeze.json"; current=fingerprints()
    if p.exists():assert json.loads(p.read_text(encoding="utf-8"))["hashes"]==current,"source changed; use new version"
    else:
        p.write_text(json.dumps({"hashes":current,"created_unix":time.time(),"datasets":DATASETS,
                                  "outer_seed":20261011,"eps":.01,"eta":.1,"retention":.9},indent=2),encoding="utf-8")
    return current


def run(name):
    threadpool_limits(1); hashes=freeze(); p=OUT/(name+".json")
    if p.exists():
        saved=json.loads(p.read_text(encoding="utf-8"));assert saved["source_hashes"]==hashes
        if saved["complete"]:return saved
        rows=saved["rows"]
    else:rows=[]
    X,y=load_dataset(name); start=time.perf_counter()
    outer=StratifiedKFold(3,shuffle=True,random_state=20261011)
    for fold,(ii,jj) in enumerate(outer.split(X,y)):
        if fold<len(rows):continue
        inner_k=min(3,int(np.unique(y[ii],return_counts=True)[1].min()))
        assert inner_k>=2
        fit=fit_outer(X[ii],y[ii],seed=20261011+fold,inner_splits=inner_k)
        A,B=preprocess_fit(X[ii],X[jj]);d=A.shape[1]
        model=fit["selected"];full={"features":list(range(d)),"weights":(np.ones(d)/np.sqrt(d)).tolist()}
        matched={"features":model["features"],"weights":(np.ones(len(model["features"]))/np.sqrt(len(model["features"]))).tolist()}
        results={m:query_scores(A,y[ii],B,y[jj],c) for m,c in (("proposed",model),("full",full),("matched_uniform",matched))}
        noise={m:[] for m in results}
        # These attacks perturb reference and query together; all methods see
        # identical draws.  They never enter inner recipe selection.
        from certified_frfs.relation_support_selector_v1 import weighted_knn
        for draw in range(8):
            rng=np.random.default_rng(20262000+10*fold+draw)
            na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            for m,c in (("proposed",model),("full",full),("matched_uniform",matched)):
                v=weighted_knn(na,y[ii],nb,y[jj],c["features"],c["weights"])
                flags=np.array(results[m]["query_certificate"]["certified"])
                assert not np.any(flags & (np.asarray(v["predictions"])!=results[m]["predictions"]))
                noise[m].append(v)
        row={"fold":fold,"train_indices":ii.tolist(),"test_indices":jj.tolist(),"fit":fit,
             **results,"shared_perturbations":noise,
             "perturbed_BA":{m:float(np.mean([v["clean_BA"] for v in vs])) for m,vs in noise.items()}}
        rows.append(row)
        assert hashes==fingerprints()
        saved={"dataset":name,"d":d,"rows":rows,"complete":len(rows)==3,"source_hashes":hashes,
               "seconds":time.perf_counter()-start,"scope":"exposed_development_nested_recipe_pilot_no_confirmation"}
        p.write_text(json.dumps(saved,separators=(',',':')),encoding="utf-8")
        print(name,fold,"clean",round(100*row["proposed"]["clean_BA"],3),
              "full",round(100*row["full"]["clean_BA"],3),
              "certified",round(100*row["proposed"]["query_certified_BA"],3),
              "full certified",round(100*row["full"]["query_certified_BA"],3),
              "count",len(model["features"]),"/",d,"family",model["family"],"fallback",model["outer_refit_fallback"],flush=True)
    return saved


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--only",choices=DATASETS);args=parser.parse_args()
    for name in ((args.only,) if args.only else DATASETS):run(name)
