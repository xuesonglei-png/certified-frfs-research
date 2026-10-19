"""Prespecified bounded training-only anchor subdivision screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from signed_classwise_dependency import signed_classwise_dependency
from anchor_subdivision_dependency_v1 import refine_signed_guard

NAMES=("banknote","glass","heart_statlog","ionosphere","seeds","wdbc")
SOURCE=ROOT/"results"/"repair-relation-selector-v4"
FOLDER=ROOT/"results"/"anchor-subdivision-dependency-v2"


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    FOLDER.mkdir(exist_ok=True)
    files=[ROOT/"scripts"/p for p in ("anchor_subdivision_dependency_v1.py","run_anchor_subdivision_dependency_v2.py","audit_anchor_subdivision_dependency_v2.py","signed_classwise_dependency.py","classwise_star_dependency.py")]
    files += [ROOT/"research"/"anchor-subdivision-dependency-v2-protocol.md",ROOT/"tests"/"test_anchor_subdivision_dependency_v1.py"]
    files += [SOURCE/(n+".json") for n in NAMES]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    path=FOLDER/"source-freeze.json"
    if path.exists():assert json.loads(path.read_text())["hashes"]==hashes,"versioned run required"
    else:path.write_text(json.dumps({"hashes":hashes,"parameters":{"eps":.01,"eta":.1,"retention":.9,"witness_pool":8,"max_leaves":16,"max_refined_objects":24},"created_unix":time.time()},indent=2))
    return hashes


def choose_candidates(saved):
    groups={"global_dependency":[],"classwise_only_dependency":[]}
    for outer in saved["rows"]:
        for inner,block in enumerate(outer["fit"]["inner"]):
            for family,part in sorted(block["families"].items()):
                unique={tuple(c["features"]):c for row in part["candidates"] for c in row["trace"] if not c["joint_feasible"]}
                for fs,c in sorted(unique.items()):
                    if not all(c[k]["majority_feasible"] for k in ("certificate","family_certificate")):
                        continue
                    global_ok=all(c[k]["dependency_certificate"]["upper_bound"]<=.1 for k in ("certificate","family_certificate"))
                    class_ok=c["classwise"]["feasible"] and c["family_classwise"]["feasible"]
                    if global_ok and class_ok:continue
                    category="classwise_only_dependency" if global_ok else "global_dependency"
                    groups[category].append({"outer_fold":outer["fold"],"inner_fold":inner,"family":family,"features":list(fs),"candidate":c,"category":category})
    # Order follows deterministic outer/inner/family/feature traversal, not outcomes.
    selected=[row for category in ("classwise_only_dependency","global_dependency") for row in groups[category][:2]]
    return selected,{k:len(v) for k,v in groups.items()}


def run(name):
    threadpool_limits(1);hashes=freeze();path=FOLDER/(name+".json")
    if path.exists():
        assert json.loads(path.read_text())["source_hashes"]==hashes
        return
    saved=json.loads((SOURCE/(name+".json")).read_text());selected,availability=choose_candidates(saved)
    X,y=load_dataset(name);rows=[];start=time.perf_counter()
    for selection in selected:
        outer=saved["rows"][selection["outer_fold"]];block=outer["fit"]["inner"][selection["inner_fold"]]
        train=np.asarray(outer["train_indices"])[block["fit_indices"]]
        A,_=preprocess_fit(X[train],X[train]);labels=y[train];c=selection["candidate"];fs=c["features"];wr=np.asarray(c["weights"])
        guards={}
        cache={}
        for guard,cls in (("certificate","classwise"),("family_certificate","family_classwise")):
            dep=c[guard]["dependency_certificate"]
            wf=np.asarray(dep.get("full_weights",wr));key=tuple(wf)
            if key in cache:
                guards[guard]=cache[key];continue
            baseline=signed_classwise_dependency(A,labels,fs,wf,wr,base_dependency=dep)
            baseline_bounds=baseline["class_certificate"]["upper_bounds"]
            refined=refine_signed_guard(A,labels,fs,wf,wr,baseline,baseline_bounds,max_leaves=16,max_objects=24)
            new_global=min(dep["upper_bound"],baseline["global_upper_bound"],refined["global_upper"])
            info={"full_weights":wf.tolist(),"legacy_global_bound":dep["upper_bound"],"legacy_class_bounds":c[cls]["bounds"],
                  "signed_baseline_global_bound":baseline["global_upper_bound"],"signed_baseline_class_bounds":baseline_bounds,
                  "signed_baseline_object_values":baseline["signed_object_report"]["signed_object_upper_bounds"],
                  "subdivision":refined,"refined_global_bound":new_global,"refined_class_bounds":refined["class_bounds"],
                  "baseline_feasible":baseline["global_upper_bound"]<=.1 and all(b<=.1 for b in baseline_bounds),
                  "refined_feasible":new_global<=.1 and all(b<=.1 for b in refined["class_bounds"])}
            cache[key]=info;guards[guard]=info
        record={k:v for k,v in selection.items() if k!="candidate"}
        record.update(weights=wr.tolist(),guards=guards,original_majority_feasible=True,
                      baseline_joint_feasible=all(g["baseline_feasible"] for g in guards.values()),
                      refined_joint_feasible=all(g["refined_feasible"] for g in guards.values()),
                      original_retained_counts={k:c[k]["retained_counts"] for k in guards},
                      original_required_counts={k:c[k]["required"] for k in guards})
        rows.append(record)
        print(name,selection["outer_fold"],selection["inner_fold"],selection["family"],len(fs),selection["category"],"signed",record["baseline_joint_feasible"],"subdiv",record["refined_joint_feasible"],flush=True)
    report={"dataset":name,"source_hashes":hashes,"availability":availability,"rows":rows,"complete":True,"seconds":time.perf_counter()-start,
            "new_outer_scoring":False,"goal_achieved":False,"scope":"bounded_training_only_certificate_screen"}
    assert hashes==freeze()
    path.write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--only",choices=NAMES);args=p.parse_args()
    for name in ((args.only,) if args.only else NAMES):run(name)
