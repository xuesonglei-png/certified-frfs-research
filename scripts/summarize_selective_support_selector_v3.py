"""Dataset-balanced descriptive summary; never promotes tuning to confirmation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"results/selective-support-selector-v3"
NAMES=("banknote","glass","heart_statlog","ionosphere","seeds","wdbc")
METHODS=("proposed","full","matched_uniform")
METRICS=("clean_BA","macro_F1","balanced_logloss","query_certified_BA",
         "query_coverage","balanced_query_coverage")


def main():
    rows=[];hashes={};audits={};folds=0
    for name in NAMES:
        p=FOLDER/(name+".json")
        if not p.exists():continue
        artifact=json.loads(p.read_text(encoding="utf-8"))
        if not artifact["complete"]:continue
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        a=FOLDER/(name+"-audit.json")
        report=json.loads(a.read_text(encoding="utf-8")) if a.exists() else {}
        audits[name]=bool(report.get("passed") and report.get("artifact_sha256")==hashes[name])
        rr=artifact["rows"];folds+=len(rr)
        row={"dataset":name,"d":artifact["d"],"folds":len(rr),
             "removed":float(np.mean([1-len(r["fit"]["selected"]["features"])/artifact["d"] for r in rr])),
             "fallbacks":sum(r["fit"]["selected"]["outer_refit_fallback"] for r in rr),
             "full_support_folds":sum(len(r["fit"]["selected"]["features"])==artifact["d"] for r in rr)}
        for m in METHODS:
            for k in METRICS:row[m+"_"+k]=float(np.mean([r[m][k] for r in rr]))
            row[m+"_perturbed_BA"]=float(np.mean([r["perturbed_BA"][m] for r in rr]))
            # Pooled selective accuracy uses ordinary counts, not mixed
            # class-balanced and sample-weighted quantities.
            covered=sum(sum(r[m]["class_certified_counts"]) for r in rr)
            correct=sum(sum(r[m]["class_certified_correct_counts"]) for r in rr)
            row[m+"_selective_accuracy"]=None if not covered else correct/covered
        rows.append(row)
    if not rows:raise RuntimeError("no complete artifacts")
    comparisons={}
    for m in ("full","matched_uniform"):
        comparisons[m]={}
        for k in (*METRICS,"perturbed_BA"):
            ds=np.array([r["proposed_"+k]-r[m+"_"+k] for r in rows])
            rng=np.random.default_rng(20261012)
            boot=np.mean(rng.choice(ds,size=(20000,len(ds)),replace=True),axis=1)
            comparisons[m][k]={"mean_delta":float(ds.mean()),"mean_delta_pp":float(100*ds.mean()),
                                "descriptive_bootstrap95_pp":(100*np.quantile(boot,[.025,.975])).tolist(),
                                "wins":int(np.sum(ds>1e-12)),"ties":int(np.sum(abs(ds)<=1e-12)),
                                "losses":int(np.sum(ds< -1e-12))}
    summary={"datasets":len(rows),"expected_datasets":len(NAMES),"complete":len(rows)==len(NAMES),
             "outer_folds":folds,"rows":rows,"comparisons":comparisons,
             "mean_removed":float(np.mean([r["removed"] for r in rows])),
             "fallbacks":sum(r["fallbacks"] for r in rows),
             "full_support_folds":sum(r["full_support_folds"] for r in rows),
             "artifact_hashes":hashes,"audit_status":audits,
             "all_completed_artifacts_audited":all(audits.values()),"goal_achieved":False,
             "classification_superiority_established":False,
             "scope":"dataset_balanced_descriptive_nested_development_not_confirmation",
             "limits":["Dataset bootstrap is descriptive for this exposed small pilot.",
                       "No recent-control superiority is claimed from full/matched-uniform comparison.",
                       "Query certificates prove stability, not truth or population generalization."]}
    (FOLDER/"development-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (FOLDER/"dataset-means.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","artifact_hashes")},indent=2))


if __name__=="__main__":main()
