"""Attribute selective v3 gains to weighting versus support deletion.

Every relation is refit from outer training only.  All named arms are reported;
there is no endpoint selection from outer labels.  Failed guards are retained
and raw scores are explicitly distinct from certified deployment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.selective_support_selector_v3 import relations,query_scores
from certified_frfs.relation_support_selector_v1 import _candidate,weighted_knn
from certified_frfs.validated_pool import preprocess_fit

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/selective-support-selector-v3"
OUT=ROOT/"results/selective-relation-ablation-v1"


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(name):
    threadpool_limits(1); OUT.mkdir(exist_ok=True)
    p=SOURCE/(name+".json"); source=json.loads(p.read_text(encoding="utf-8"));assert source["complete"]
    freeze=json.loads((SOURCE/"source-freeze.json").read_text(encoding="utf-8"))["hashes"]
    for rel,sha in freeze.items():assert digest(ROOT/rel)==sha
    hashes={**freeze,p.relative_to(ROOT).as_posix():digest(p),Path(__file__).relative_to(ROOT).as_posix():digest(Path(__file__))}
    out=OUT/(name+".json")
    if out.exists():
        stored=json.loads(out.read_text(encoding="utf-8"));assert stored["fingerprints"]==hashes;return stored
    X,y=load_dataset(name);results=[]
    for row in source["rows"]:
        ii=np.array(row["train_indices"]); jj=np.array(row["test_indices"])
        A,B=preprocess_fit(X[ii],X[jj]); fit=row["fit"]; d=A.shape[1]
        fs=fit["selected"]["features"]; families=relations(A,y[ii]); arms={}
        for family,w in families.items():
            for kind,support in (("full",list(range(d))),("fixed_proposed_support",fs)):
                c=_candidate(A,y[ii],w,support,len(support))
                score=query_scores(A,y[ii],B,y[jj],c)
                noisy=[]
                for draw in range(8):
                    rng=np.random.default_rng(20262000+10*row["fold"]+draw)
                    na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
                    nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
                    ev=weighted_knn(na,y[ii],nb,y[jj],c["features"],c["weights"])
                    assert not np.any(np.array(score["query_certificate"]["certified"]) &
                                      (np.array(ev["predictions"])!=np.array(score["predictions"])))
                    noisy.append(ev)
                arms[family+"_"+kind]={"candidate":c,"raw_score":score,"shared_perturbations":noisy,
                                      "raw_perturbed_BA":float(np.mean([v["clean_BA"] for v in noisy])),
                                      "admissible_under_unchanged_dual_guards":bool(c["joint_feasible"])}
        results.append({"fold":row["fold"],"train_indices":ii.tolist(),"test_indices":jj.tolist(),
                        "proposed_family":fit["selected"]["family"],"proposed_support":fs,
                        "arms":arms,"proposed":row["proposed"]})
        print(name,row["fold"],"ablation clean/certified",
              {m:(round(100*v["raw_score"]["clean_BA"],3),round(100*v["raw_score"]["query_certified_BA"],3),v["admissible_under_unchanged_dual_guards"]) for m,v in arms.items()},flush=True)
    assert all(digest(ROOT/rel)==sha for rel,sha in hashes.items())
    result={"dataset":name,"rows":results,"fingerprints":hashes,"complete":True,
            "outer_labels_used_for_selection":False,"scope":"prespecified_relation_and_same_support_ablation_all_raw_arms_not_new_deployment_or_confirmation"}
    out.write_text(json.dumps(result,separators=(',',':')),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--only",required=True);run(parser.parse_args().only)
