"""Independent complete-metric merge of two frozen diagnostics, no fitting."""
from __future__ import annotations

from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from certified_frfs.datasets import load_dataset

NAMES=("banknote","glass","heart_statlog","ionosphere","seeds","wdbc")
SOURCE=ROOT/"results"/"repair-relation-selector-v4"
OPP=ROOT/"results"/"rejected-pool-opportunity-v7"
HEAD=ROOT/"results"/"nominal-dependency-headroom-v7"
OUT=ROOT/"results"/"rejected-opportunity-headroom-review-v1"
CATEGORIES=("admitted","dependency_only_rejected","majority_only_rejected","both_rejected")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references(candidate,d):
    result=[]
    for guard in ("certificate","family_certificate"):
        dep=candidate[guard]["dependency_certificate"]
        ref=dep.get("full_weights",candidate["weights"])
        assert len(ref)==d
        result.append(tuple(ref))
    return result


def key(panel,fold,family,candidate,d):
    return (panel,fold,family,tuple(candidate["features"]),tuple(candidate["weights"]),*references(candidate,d))


def key_data(value):
    panel,inner,family,fs,weights,uniform,family_ref=value
    return {"panel":panel,"inner_fold":inner,"family":family,"features":list(fs),"weights":list(weights),
            "original_reference_weights":list(uniform),"family_reference_weights":list(family_ref)}


def summarize_best(records,admitted):
    if not records:return None
    # Deterministic review tie: full key, not family/support alone.
    best=min(records,key=lambda row:(-row["inner_BA"],len(row["features"]),row["metric_key_sha256"]))
    return {k:best[k] for k in ("metric_key_sha256","family","features","inner_BA","category","blockers","nominal_violation_witness")}|{"delta_vs_best_admitted_pp":100*(best["inner_BA"]-admitted)}


def run(name):
    source=json.loads((SOURCE/(name+".json")).read_text());opp=json.loads((OPP/(name+".json")).read_text());head=json.loads((HEAD/(name+".json")).read_text())
    src_hash=digest(SOURCE/(name+".json"));assert src_hash==opp["source_sha256"]==head["source_sha256"]
    X,y=load_dataset(name);d=source["d"]
    reconstruction=[]
    for outer in source["rows"]:
        for fold,block in enumerate(outer["fit"]["inner"]):
            for family,part in block["families"].items():
                unique={}
                for recipe in part["candidates"]:
                    for candidate in recipe["trace"]:
                        ref=references(candidate,d)[1]
                        unique[(tuple(candidate["features"]),tuple(candidate["weights"]),ref)]=candidate
                for candidate in unique.values():
                    reconstruction.append((key(outer["fold"],fold,family,candidate,d),candidate))
    assert len(reconstruction)==len(head["records"])
    indexed={};oldkeys=defaultdict(set)
    for (full_key,candidate),record in zip(reconstruction,head["records"],strict=True):
        assert (record["panel"],record["inner_fold"],record["family"],record["features"])==(full_key[0],full_key[1],full_key[2],list(full_key[3]))
        assert full_key not in indexed
        indexed[full_key]=(record,candidate)
        oldkeys[full_key[:4]].add(full_key)
    collision_groups={str(k):len(v) for k,v in oldkeys.items() if len(v)>1}
    tally=Counter();blocks=[];joined=[]
    for block in opp["records"]:
        panel,fold=block["panel"],block["inner_fold"]
        outer=source["rows"][panel];inner=outer["fit"]["inner"][fold]
        truth=y[np.array(outer["train_indices"])[inner["validation_indices"]]]
        records=[]
        for record in block["candidates"]:
            # Headroom v7 intentionally omitted weights/references. Recover them
            # from exact frozen traversal and match using the complete key.
            original_ref=tuple(np.ones(d)/np.sqrt(d))
            full_key=(panel,fold,record["family"],tuple(record["features"]),tuple(record["weights"]),original_ref,tuple(record["reference_weights"]))
            if full_key not in indexed:
                matches=[k for k in indexed if k[:5]==full_key[:5] and k[6]==full_key[6]]
                assert len(matches)==1
                full_key=matches[0]
            head_record,candidate=indexed.pop(full_key)
            assert record["blockers"]==head_record["blockers"]
            preds=np.asarray(record["predictions"]);assert len(preds)==len(truth)
            ba=float(np.mean([np.mean(preds[truth==c]==c) for c in np.unique(truth)]))
            np.testing.assert_allclose(ba,record["inner_BA"],rtol=0,atol=1e-12)
            sides=head_record["sides"]
            violation=any(s["nominal_global_violation"] or s["nominal_any_class_violation"] for s in sides.values())
            assert violation==head_record["nominal_violation_witness"]
            for side,guard,cls in (("original","certificate","classwise"),("learned","family_certificate","family_classwise")):
                assert sides[side]["saved_global_upper_bound"]==candidate[guard]["dependency_certificate"]["upper_bound"]
                assert sides[side]["saved_class_upper_bounds"]==candidate[cls]["bounds"]
                assert sides[side]["nominal_global_violation"]==(sides[side]["nominal_signed_loss"]>.1+1e-10)
                assert sides[side]["nominal_any_class_violation"]==(max(sides[side]["nominal_class_signed_losses"])>.1+1e-10)
            identity=key_data(full_key);signature=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            row={**identity,"metric_key_sha256":signature,"inner_BA":ba,"category":record["category"],"blockers":record["blockers"],
                 "nominal_violation_witness":violation,"nominal_loss_sides":sides}
            records.append(row);joined.append(row);tally[record["category"]]+=1;tally["complete_key_matches"]+=1
            tally["inner_BA_from_saved_predictions_replays"]+=1
        admitted=max(r["inner_BA"] for r in records if r["category"]=="admitted")
        for row in records:
            row["delta_vs_best_admitted_pp"]=100*(row["inner_BA"]-admitted)
        rejected=[r for r in records if r["category"]!="admitted"]
        higher=[r for r in rejected if r["delta_vs_best_admitted_pp"]>1e-12]
        safe_nominal=[r for r in higher if not r["nominal_violation_witness"]]
        violating=[r for r in rejected if r["nominal_violation_witness"]]
        tally["higher_BA_rejected_candidates"]+=len(higher)
        tally["higher_BA_rejected_without_nominal_violation"]+=len(safe_nominal)
        tally["higher_BA_rejected_with_nominal_violation"]+=sum(r["nominal_violation_witness"] for r in higher)
        for row in safe_nominal:tally["higher_BA_no_nominal_violation_"+row["category"]]+=1
        block_report={"panel":panel,"inner_fold":fold,"best_admitted_BA":admitted,
                      "category_best":{cat:summarize_best([r for r in records if r["category"]==cat],admitted) for cat in CATEGORIES},
                      "best_nominal_violating_rejected":summarize_best(violating,admitted),
                      "best_nonviolating_rejected":summarize_best([r for r in rejected if not r["nominal_violation_witness"]],admitted),
                      "higher_BA_rejected_without_nominal_violation_count":len(safe_nominal),
                      "optimistic_no_nominal_violation_headroom_pp":max([0.]+[r["delta_vs_best_admitted_pp"] for r in safe_nominal]),
                      "optimistic_any_headroom_pp":max([0.]+[r["delta_vs_best_admitted_pp"] for r in rejected])}
        blocks.append(block_report)
    assert not indexed
    report={"dataset":name,"passed":True,"source_sha256":src_hash,"opportunity_sha256":digest(OPP/(name+".json")),"headroom_sha256":digest(HEAD/(name+".json")),
            "counts":dict(tally),"lossy_family_support_collision_groups":collision_groups,"blocks":blocks,"joined_candidates":joined,
            "mean_any_oracle_headroom_pp":float(np.mean([b["optimistic_any_headroom_pp"] for b in blocks])),
            "mean_no_nominal_violation_oracle_headroom_pp":float(np.mean([b["optimistic_no_nominal_violation_headroom_pp"] for b in blocks])),
            "blocks_with_higher_BA_nonviolating_rejected":sum(b["higher_BA_rejected_without_nominal_violation_count"]>0 for b in blocks),
            "new_fit":False,"new_predictions":False,"new_outer_scoring":False,"scope":"exposed_inner_oracle_join_review_not_training_recipe_or_confirmatory_accuracy"}
    (OUT/(name+".json")).write_text(json.dumps(report,indent=2));return report


def main():
    OUT.mkdir(exist_ok=True);freeze_path=OUT/"source-freeze.json"
    paths=[Path(__file__),ROOT/"research"/"rejected-opportunity-headroom-review-v1-protocol.md"]
    paths+=[base/(n+".json") for base in (SOURCE,OPP,HEAD) for n in NAMES]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    if freeze_path.exists():assert json.loads(freeze_path.read_text())["hashes"]==hashes
    else:freeze_path.write_text(json.dumps({"hashes":hashes,"created_unix":time.time()},indent=2))
    reports=[run(n) for n in NAMES];total=sum((Counter(r["counts"]) for r in reports),Counter())
    summary={"passed":True,"datasets":6,"blocks":sum(len(r["blocks"]) for r in reports),"counts":dict(total),
             "dataset_rows":[{k:r[k] for k in ("dataset","mean_any_oracle_headroom_pp","mean_no_nominal_violation_oracle_headroom_pp","blocks_with_higher_BA_nonviolating_rejected","lossy_family_support_collision_groups","counts")} for r in reports],
             "per_category_best_mean_delta_vs_best_admitted_pp":{cat:float(np.mean([b["category_best"][cat]["delta_vs_best_admitted_pp"] for r in reports for b in r["blocks"] if b["category_best"][cat] is not None])) for cat in CATEGORIES},
             "category_best_available_blocks":{cat:sum(b["category_best"][cat] is not None for r in reports for b in r["blocks"]) for cat in CATEGORIES},
             "mean_best_nominal_violating_rejected_delta_pp":float(np.mean([b["best_nominal_violating_rejected"]["delta_vs_best_admitted_pp"] for r in reports for b in r["blocks"] if b["best_nominal_violating_rejected"] is not None])),
             "nominal_violating_best_rejected_available_blocks":sum(b["best_nominal_violating_rejected"] is not None for r in reports for b in r["blocks"]),
             "new_fit":False,"new_predictions":False,"new_outer_scoring":False,"goal_achieved":False,
             "limitations":["All maxima use exposed inner validation and are optimistic oracles; no deployable selection rule is produced.","No nominal violation is only a necessary condition, not robust feasibility or guaranteed loose certificate.","Weights and references omitted in headroom-v7 are recovered from its exact frozen traversal, never joined by support alone.","Best-rejected accuracy may require violating the original guarantees.","No independent training or perturbation certificate recomputation is claimed."]}
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
