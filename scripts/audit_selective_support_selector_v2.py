"""Independent votes/query stability/selection replay for the v2 pilot.

Only deployed candidates receive the slow independent training guard replay.
All inner feasible scores receive independent scalar voting and query replay.
Optimizer independence and complete pre-run source freezing are not claimed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from audit_relation_support_selector_v1 import independent_evaluate, check_guard, equivalent
from audit_majority_bound_prediction_v1 import query_certificate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "selective-support-selector-v2"


def check_score(A, y, B, truth, model, saved, counts):
    independent = independent_evaluate(A, y, B, truth, model["features"], model["weights"])
    for key, value in independent.items(): equivalent(saved[key], value)
    query = query_certificate(A, y, B, np.asarray(independent["predictions"]),
                              model["features"], np.asarray(model["weights"]))
    flags = np.asarray(query["certified"])
    correct = (np.asarray(independent["predictions"]) == truth) & flags
    ba = float(np.mean([np.mean(correct[truth == c]) for c in np.unique(truth)]))
    equivalent(saved["query_certified_BA"], ba)
    equivalent(saved["query_coverage"], float(flags.mean()))
    counts["independent_predictions"] += 1


def audit(name):
    threadpool_limits(1)
    p = FOLDER / (name + ".json"); saved = json.loads(p.read_text(encoding="utf-8"))
    X, y = load_dataset(name); counts = Counter()
    splits = list(StratifiedKFold(min(3,int(np.unique(y,return_counts=True)[1].min())),
                                 shuffle=True, random_state=20261011).split(X,y))
    assert len(saved["rows"]) == len(splits)
    for fold, ((ii,jj), row) in enumerate(zip(splits,saved["rows"])):
        assert row["fold"] == fold and row["train_indices"] == ii.tolist() and row["test_indices"] == jj.tolist()
        if row["status"] != "complete":
            assert np.unique(y[ii],return_counts=True)[1].min()<2; continue
        fit = row["fit"]
        assert fit["certificate_parameters"] == {"eps": .01,"eta": .1,"retention": .9}
        assert fit["clean_tolerance"] == .005 and fit["seed"] == 20261011+fold
        inner = list(StratifiedKFold(fit["inner_splits"],shuffle=True,random_state=fit["seed"]).split(X[ii],y[ii]))
        summaries = []
        for block in fit["inner"]:
            assert len(block["folds"]) == len(inner)
            for (aa,bb), part in zip(inner,block["folds"]):
                assert part["fit_indices"] == aa.tolist() and part["validation_indices"] == bb.tolist()
                A,B = preprocess_fit(X[ii][aa],X[ii][bb])
                for candidate in part["candidates"]:
                    c = candidate["candidate"]
                    assert candidate["features"] == c["features"]
                    assert c["joint_feasible"] == (c["certificate"]["feasible"] and c["family_certificate"]["feasible"] and c["classwise"]["feasible"] and c["family_classwise"]["feasible"])
                    w=np.asarray(c["weights"])
                    assert np.all(w >= 0) and np.max(w) <= 2/np.sqrt(saved["d"])+1e-8
                    np.testing.assert_allclose(w.sum(),np.sqrt(saved["d"]),atol=1e-8)
                    if candidate["score"] is not None:
                        assert c["joint_feasible"]
                        check_score(A,y[ii][aa],B,y[ii][bb],c,candidate["score"],counts)
                    else: assert not c["joint_feasible"]
            keys=sorted(set(tuple(c["features"]) for f in block["folds"] for c in f["candidates"]))
            for fs in keys:
                candidates=[next((c for c in f["candidates"] if tuple(c["features"])==fs),None) for f in block["folds"]]
                if any(c is None or c["score"] is None for c in candidates):continue
                summaries.append({"family":block["family"],"features":list(fs),
                                  "mean_BA":float(np.mean([c["score"]["clean_BA"] for c in candidates])),
                                  "mean_certified_BA":float(np.mean([c["score"]["query_certified_BA"] for c in candidates])),
                                  "mean_coverage":float(np.mean([c["score"]["query_coverage"] for c in candidates]))})
        best=max(c["mean_BA"] for c in summaries)
        selected=max([c for c in summaries if c["mean_BA"]>=best-.005-1e-12],
                     key=lambda c:(c["mean_certified_BA"],c["mean_coverage"],c["mean_BA"],-len(c["features"]),c["family"],tuple(c["features"])))
        equivalent(fit["chosen"],selected)
        A,B=preprocess_fit(X[ii],X[jj]); model=fit["selected"]
        uniform=np.ones(saved["d"])/np.sqrt(saved["d"])
        assert model["joint_feasible"]
        assert check_guard(A,y[ii],model["features"],uniform,np.asarray(model["weights"]),model["certificate"],model["classwise"])
        family=uniform if model["outer_refit_fallback"] else np.asarray(fit["refit_reference_weights"])
        assert check_guard(A,y[ii],model["features"],family,np.asarray(model["weights"]),model["family_certificate"],model["family_classwise"])
        for method,c in (("proposed",model),("full",{"features":list(range(saved["d"])),"weights":uniform}),
                         ("matched_uniform",{"features":model["features"],"weights":np.ones(len(model["features"]))/np.sqrt(len(model["features"]))})):
            check_score(A,y[ii],B,y[jj],c,row[method],counts)
        counts["outer_folds"]+=1
        print(name,fold,"independent audit passed",dict(counts),flush=True)
    report={"passed":True,"dataset":name,"counts":dict(counts),
            "artifact_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
            "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations":["no_complete_prerun_source_freeze","learning_and_candidate_generation_not_independently_replayed",
                           "training_guards_replayed_only_for_deployed_candidates","exposed_development_data"]}
    (FOLDER/(name+"-audit.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--only",required=True)
    print(json.dumps(audit(parser.parse_args().only),indent=2),flush=True)
