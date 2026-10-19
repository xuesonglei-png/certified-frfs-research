"""Complete bounded screen including empty strata and negative outcomes."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"results"/"anchor-subdivision-dependency-v2"
NAMES=("banknote","glass","heart_statlog","ionosphere","seeds","wdbc")
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[];totals=Counter()
    for name in NAMES:
        path=FOLDER/(name+".json");saved=json.loads(path.read_text());candidates=saved["rows"]
        guards=[g for row in candidates for g in row["guards"].values()]
        reductions=[max(g["signed_baseline_class_bounds"])-max(g["refined_class_bounds"]) for g in guards]
        objects=[o for g in guards for o in g["subdivision"]["objects"]]
        audit_path=FOLDER/(name+"-audit.json");audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        passed=(audit.get("passed",False) and audit.get("artifact_sha256")==digest(path) and audit.get("auditor_sha256")==digest(ROOT/"scripts"/"audit_anchor_subdivision_dependency_v2.py"))
        if passed:totals.update(audit["counts"])
        rows.append({"dataset":name,"candidates":len(candidates),"availability":saved["availability"],
                     "baseline_joint_admissions":sum(c["baseline_joint_feasible"] for c in candidates),
                     "subdivision_joint_admissions":sum(c["refined_joint_feasible"] for c in candidates),
                     "new_admissions_beyond_signed":sum(c["refined_joint_feasible"] and not c["baseline_joint_feasible"] for c in candidates),
                     "max_worst_class_tightening":max(reductions,default=0.),
                     "mean_worst_class_tightening":float(np.mean(reductions)) if reductions else 0.,
                     "strict_object_tightenings":sum(o["root_upper"]-o["refined_upper"]>1e-10 for o in objects),
                     "refined_object_guard_records":len(objects),"independent_audit_current_hashes_passed":passed,
                     "seconds":saved["seconds"],"artifact_sha256":digest(path)})
    freeze=json.loads((FOLDER/"source-freeze.json").read_text());hash_mismatches=[p for p,h in freeze["hashes"].items() if digest(ROOT/p)!=h]
    report={"goal_achieved":False,"new_outer_scoring":False,"independent_optimizer_claim":False,
            "parameters":freeze["parameters"],"frozen_source_hash_mismatches":hash_mismatches,
            "all_six_audits_passed":all(r["independent_audit_current_hashes_passed"] for r in rows),
            "candidate_models":sum(r["candidates"] for r in rows),"new_admissions_beyond_signed":sum(r["new_admissions_beyond_signed"] for r in rows),
            "existing_signed_admissions":sum(r["baseline_joint_admissions"] for r in rows),
            "max_worst_class_tightening":max(r["max_worst_class_tightening"] for r in rows),
            "strict_object_tightenings":sum(r["strict_object_tightenings"] for r in rows),
            "audit_totals":dict(totals),"datasets":rows,
            "decision":"Retain negative diagnostic; do not integrate as default or claim predictive gain.",
            "limitations":["prespecified_first_candidates_not_exhaustive", "at_most_24_anchor_objects_per_guard_and_16_leaves", "no_test_prediction_scores", "existing_signed_star_baseline_not_independently_rederived_here", "floating_point_not_formally_verified", "zero_screen_admissions_not_proof_no_candidate_can_benefit"]}
    (FOLDER/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="datasets"},indent=2))


if __name__=="__main__":main()
