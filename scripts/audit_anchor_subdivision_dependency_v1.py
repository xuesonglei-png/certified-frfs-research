"""Strict scalar replay of preserved anchor subdivision trees and aggregation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_anchor_subdivision_dependency_v1 import NAMES,SOURCE,FOLDER,choose_candidates,digest


def scalar_leaf(A,y,i,left,right,witnesses,wf,wide):
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    enemies=np.flatnonzero(y!=y[i]);lower={},{}
    lows,highs={},{}
    for j in enemies:
        lows[int(j)]=sum(float(wide[a])*max(0.,float(left[a]-hi[j,a]),float(lo[j,a]-right[a])) for a in range(A.shape[1]))
        highs[int(j)]=sum(float(wide[a])*max(abs(float(left[a]-hi[j,a])),abs(float(right[a]-lo[j,a]))) for a in range(A.shape[1]))
    cap=min(highs.values());possible=[int(j) for j in enemies if lows[int(j)]<=cap+1e-12]
    contrasts=[]
    for j in possible:
        each=[]
        for k in witnesses:
            value=0.
            for a in range(A.shape[1]):
                if j==k:
                    coefficient=float(wf[a]-wide[a])
                    near=max(0.,float(left[a]-hi[j,a]),float(lo[j,a]-right[a]))
                    far=max(abs(float(left[a]-hi[j,a])),abs(float(right[a]-lo[j,a])))
                    value+=coefficient*(far if coefficient>=0 else near)
                else:
                    points=[float(left[a]),float(right[a])]
                    for p in (lo[j,a],hi[j,a],lo[k,a],hi[k,a],(lo[k,a]+hi[k,a])/2):
                        points.append(float(np.clip(p,left[a],right[a])))
                    value+=max(float(wf[a])*max(abs(t-float(lo[k,a])),abs(t-float(hi[k,a])))-float(wide[a])*max(0.,float(lo[j,a])-t,t-float(hi[j,a])) for t in points)
            each.append(value)
        contrasts.append(min(1.-lows[j],min(each)))
    raw=max(contrasts);unsaturated=cap<=1.-1e-12
    return {"raw_upper":raw,"upper":raw if unsaturated else max(0.,raw),"unsaturated":unsaturated,"reduced_upper_cap":cap,"possible_enemies":possible}


def audit_tree(A,y,fs,wf,wr,tree,tally):
    i=tree["object"];wide=np.zeros(A.shape[1]);wide[fs]=wr
    enemies=np.flatnonzero(y!=y[i]);dist=lambda k:sum(abs(float(A[i,a]*wf[a])-float(A[k,a]*wf[a])) for a in range(A.shape[1]))
    witnesses=sorted(enemies,key=lambda k:(dist(k),int(k)))[:8]
    assert tree["witnesses"]==[int(k) for k in witnesses]
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    np.testing.assert_array_equal(tree["nodes"][0]["lower"],lo[i]);np.testing.assert_array_equal(tree["nodes"][0]["upper_coordinates"],hi[i])
    parents={child:s["parent"] for s in tree["splits"] for child in s["children"]}
    replay=[]
    for idx,node in enumerate(tree["nodes"]):
        value=scalar_leaf(A,y,i,np.array(node["lower"]),np.array(node["upper_coordinates"]),witnesses,wf,wide)
        if idx in parents:value["upper"]=min(value["upper"],replay[parents[idx]]["upper"])
        for key in ("raw_upper","upper","reduced_upper_cap"):
            np.testing.assert_allclose(node[key],value[key],atol=1e-10,rtol=1e-9)
        for key in ("unsaturated","possible_enemies"):assert node[key]==value[key]
        replay.append(value);tally["scalar_node_bounds"]+=1
    leaves=[0]
    for split in tree["splits"]:
        parent=split["parent"];expected=max(leaves,key=lambda idx:(tree["nodes"][idx]["upper"],-idx));assert parent==expected
        node=tree["nodes"][parent];left,right=np.array(node["lower"]),np.array(node["upper_coordinates"])
        axis=int(np.argmax((right-left)*(wf+wide)));assert split["axis"]==axis
        middle=(left[axis]+right[axis])/2;assert split["midpoint"]==middle
        for direction,child in enumerate(split["children"]):
            child_left,child_right=left.copy(),right.copy()
            if direction==0:child_right[axis]=middle
            else:child_left[axis]=middle
            np.testing.assert_array_equal(tree["nodes"][child]["lower"],child_left)
            np.testing.assert_array_equal(tree["nodes"][child]["upper_coordinates"],child_right)
        leaves.remove(parent);leaves.extend(split["children"])
    assert leaves==tree["leaves"] and len(leaves)<=16
    expected=min(replay[0]["upper"],max(replay[idx]["upper"] for idx in leaves))
    np.testing.assert_allclose(tree["refined_upper"],expected,atol=1e-10,rtol=1e-9)
    tally["objects"]+=1


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+".json");saved=json.loads(path.read_text());source=json.loads((SOURCE/(name+".json")).read_text());selected,_=choose_candidates(source)
    assert len(selected)==len(saved["rows"]);X,y=load_dataset(name);tally=Counter()
    for row,selection in zip(saved["rows"],selected):
        for key in ("outer_fold","inner_fold","family","features","category"):assert row[key]==selection[key]
        outer=source["rows"][row["outer_fold"]];block=outer["fit"]["inner"][row["inner_fold"]]
        ii=np.array(outer["train_indices"])[block["fit_indices"]];A,_=preprocess_fit(X[ii],X[ii]);labels=y[ii];fs=row["features"];wr=np.asarray(row["weights"])
        for guard,info in row["guards"].items():
            wf=np.array(info["full_weights"]);refined=info["subdivision"];values=np.array(info["signed_baseline_object_values"])
            classes,counts=np.unique(labels,return_counts=True)
            failing=[c for c,b in zip(classes,info["signed_baseline_class_bounds"]) if b>.1]
            eligible=np.flatnonzero(np.isin(labels,failing));eligible=eligible[np.lexsort((eligible,-values[eligible]))][:24]
            assert [r["object"] for r in refined["objects"]]==eligible.tolist()
            for tree in refined["objects"]:
                audit_tree(A,labels,fs,wf,wr,tree,tally);i=tree["object"]
                np.testing.assert_allclose(tree["root_upper"],values[i],atol=1e-9,rtol=1e-8)
                values[i]=min(values[i],tree["refined_upper"])
            np.testing.assert_allclose(values,refined["signed_object_values"],atol=1e-10,rtol=1e-9)
            raw=[float(np.mean(values[labels==c])) for c in classes];bounds=[min(old,float(np.clip(v+1e-10*(1+abs(v)),0,1))) for old,v in zip(info["signed_baseline_class_bounds"],raw)]
            np.testing.assert_allclose(bounds,info["refined_class_bounds"],atol=1e-10,rtol=1e-9)
            glob=min(info["legacy_global_bound"],info["signed_baseline_global_bound"],float(np.array(counts)@bounds/len(labels)))
            np.testing.assert_allclose(info["refined_global_bound"],glob,atol=1e-10,rtol=1e-9)
            assert info["refined_feasible"]==(glob<=.1 and all(b<=.1 for b in bounds))
            candidate=selection["candidate"]
            assert row["original_retained_counts"][guard]==candidate[guard]["retained_counts"]
            assert row["original_required_counts"][guard]==candidate[guard]["required"]
            assert candidate[guard]["majority_feasible"]
            assert np.all(np.array(bounds)<=np.array(info["signed_baseline_class_bounds"])+1e-12)
            tally["guards"]+=1
        assert row["refined_joint_feasible"]==all(g["refined_feasible"] for g in row["guards"].values())
        tally["candidates"]+=1
        print(name,tally["candidates"],"scalar subdivision replay passed",flush=True)
    report={"passed":True,"dataset":name,"counts":dict(tally),"artifact_sha256":digest(path),"auditor_sha256":digest(__file__),"source_hashes":saved["source_hashes"],"no_outer_scoring":True,"independent_optimizer_claim":False,"scope":"independent_scalar_subbox_node_and_cover_replay","limitations":["signed_star_baseline_taken_from_existing_implementation_not_rederived","production_preprocessing_reused","ordinary_floating_point_not_formally_verified"]}
    (FOLDER/(name+"-audit.json")).write_text(json.dumps(report,indent=2));return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--only",choices=NAMES);args=p.parse_args()
    for name in ((args.only,) if args.only else NAMES):audit(name)
