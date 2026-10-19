"""Retain exact-oracle comparisons and counterexamples as actual results."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from certified_frfs.core import certificate, gamma, forest_maximum
from certified_frfs.verification import exact_loss_milp, sampled_attack


def safe(obj):
    if isinstance(obj,dict):return {k:safe(v) for k,v in obj.items()}
    if isinstance(obj,(np.ndarray,)):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,list):return [safe(v) for v in obj]
    return obj


def main():
    records=[]
    # Shared-perturbation vs separately minimized dependency.
    X=np.array([[0.,0.],[.3,.3]])
    eps=np.array([[0.,0.],[.1,.2]])
    oracle=exact_loss_milp(X,[0,1],[0],[.5,.5],eps)
    records.append({"case":"separate_minima_counterexample","separate_minima_difference":.05,
                    "actual_expected":.25,"oracle":oracle})
    # Interior maximum: box vertices are not an exact oracle for E.
    X=np.array([[.5],[0.],[1.]])
    oracle=exact_loss_milp(X,[0,1,1],[],[1.],np.array([[.5],[0.],[0.]]))
    records.append({"case":"interior_maximum","vertex_value":1/3,"actual_expected":.5,"oracle":oracle})
    rng=np.random.default_rng(123)
    for sample in range(12):
        X=rng.uniform(.1,.9,(4,3))
        y=np.array([0,0,1,1]);B=[0,1];w=np.ones(3)/np.sqrt(3)
        eps=.03
        cert=certificate(X,y,B,w,eps)
        oracle=exact_loss_milp(X,y,B,w,eps,time_limit=15)
        attack=sampled_attack(X,y,B,w,eps,n_samples=100,seed=sample)
        assert oracle["lower_bound"] <= cert["upper_bound"]+1e-7
        assert attack["lower_bound"] <= oracle["upper_bound"]+1e-7
        records.append({"case":"random_small","index":sample,"X":X,"y":y,"features":B,
                        "weights":w,"eps":eps,"certificate":cert,"oracle":oracle,
                        "attack_lower_bound":attack["lower_bound"]})
    target=ROOT/"results"/"prototypes.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(safe(records),indent=2),encoding="utf-8")
    print(json.dumps({"cases":len(records),"optimal":sum(r["oracle"]["optimal"] for r in records),
                      "max_gap":max(r["oracle"]["gap"] for r in records)}))

if __name__=="__main__":main()
