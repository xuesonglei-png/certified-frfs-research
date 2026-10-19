"""Finite chord-gap benchmark reusing all previously verified MILP intervals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.core import interval_bounds
from certified_frfs.margin_chord import chord_enemy_interval

INPUT = ROOT / "results/round3/margin-mixture-benchmark.json"
OUTPUT = ROOT / "results/round3/margin-gap-benchmark.json"
PROTOCOL = ROOT / "research/round3/margin-gap-benchmark-protocol.md"
TOLERANCE = 2e-6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for p in [Path(__file__), PROTOCOL, *sorted((ROOT / "src").rglob("*.py"))]}


def plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {str(k): plain(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [plain(x) for x in v]
    return v


def save(result):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary=OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(OUTPUT)


def tight_family(m, positive_width):
    started=time.perf_counter()
    r=1/(4*m)
    centers=(2*np.arange(1,m+1)-1)/(2*m)
    delta_y, eta, zeta = (0.02,0.1,0.05) if positive_width else (0.,0.,0.)
    rho=eta+r+.25 if positive_width else .75
    lower=np.vstack([[0,0],np.column_stack([centers-r,np.zeros(m)]),[0,rho]])
    upper=np.vstack([[1,delta_y],np.column_stack([centers+r,np.full(m,eta)]),[1,rho+zeta]])
    result=chord_enemy_interval(lower,upper,0,m+1,range(1,m+1),[0,1],[.5,.5])
    effective_rho=rho-eta
    true=.5*(effective_rho-r-1/(2*m))
    theoretical_mix=.5*(effective_rho-r-.5)
    theoretical_delta=.25
    measured_mix=result["mixture_result"]["raw_feasible_mixture_value"]
    pure=result["mixture_result"]["onehot_bound"]
    errors={"true_below_mixture": measured_mix>true+TOLERANCE,
            "theoretical_mixture_mismatch": abs(measured_mix-theoretical_mix)>TOLERANCE,
            "pure_mismatch": abs(pure-theoretical_mix)>TOLERANCE,
            "delta_mismatch": abs(result["delta"]-theoretical_delta)>TOLERANCE,
            "attack_below_true": result["upper_bound"]<true-TOLERANCE,
            "posterior_above_delta": result["posterior_gap"]>theoretical_delta+TOLERANCE}
    return {"family":"all_positive_width" if positive_width else "simple",
            "m":m,"r":r,"rho":rho,"query_vertical_width":delta_y,
            "friend_vertical_width":eta,"enemy_vertical_width":zeta,"weights":[.5,.5],
            "lower":lower,"upper":upper,"theoretical_true_M":true,
            "theoretical_P_onehot":theoretical_mix,"theoretical_L_mixture":theoretical_mix,
            "theoretical_Delta":theoretical_delta,"theoretical_gap_ratio":1-1/m,
            "P_onehot":pure,"L_mixture_raw":measured_mix,"L_mixture_safe":result["lower_bound"],
            "Delta":result["delta"],"gap_ratio":(true-measured_mix)/result["delta"],
            "attack_upper":result["upper_bound"],"posterior_gap":result["posterior_gap"],
            "chord_LP_reference_value":result["numerical_opt_chord"],
            "chord_result":result,"all_coordinate_widths_positive":bool(np.all(upper>lower)),
            "checks":errors,"seconds":time.perf_counter()-started}


def random_object(original, table):
    started=time.perf_counter()
    X,y,weights=np.asarray(table["X"]),np.asarray(table["y"]),np.asarray(table["weights"])
    eps=table["heterogeneous_eps"] if original["group"]=="heterogeneous_eps" else .01
    lower,upper=interval_bounds(X,eps)
    target=original["target"]
    friends,enemies=original["friends"],original["enemies"]
    enemy_rows=[]
    checks=[]
    for enemy in enemies:
        start=time.perf_counter()
        result=chord_enemy_interval(lower,upper,target,enemy,friends,range(table["d"]),weights)
        z=result["worst_case"]
        distances=np.abs(z-z[target])@weights
        actual=float(distances[enemies].min()-distances[friends].min())
        pair=float(distances[enemy]-distances[friends].min())
        legal=bool(np.all(z>=lower) and np.all(z<=upper))
        if not legal or abs(actual-result["all_enemy_margin_at_witness"])>1e-12 or abs(pair-result["upper_bound"])>1e-12:
            checks.append(f"invalid_witness_enemy_{enemy}")
        old=next(e["mixture"]["lower_bound"] for e in original["enemy_mixtures"] if e["enemy"]==enemy)
        if abs(result["lower_bound"]-old)>TOLERANCE:
            checks.append(f"mixture_changed_enemy_{enemy}")
        enemy_rows.append({"enemy":enemy,"seconds":time.perf_counter()-start,"result":result,
                           "witness_legal_and_recomputed":legal and abs(actual-result["all_enemy_margin_at_witness"])<=1e-12})
    bound=min(e["result"]["lower_bound"] for e in enemy_rows)
    attack=min(e["result"]["all_enemy_margin_at_witness"] for e in enemy_rows)
    delta=enemy_rows[0]["result"]["delta"]
    if any(abs(e["result"]["delta"]-delta)>1e-12 for e in enemy_rows):
        checks.append("enemy_dependent_delta")
    exact_lo,exact_hi=original["robust_margin_interval"]
    gap_lo,gap_hi=exact_lo-bound,exact_hi-bound
    if gap_hi < -TOLERANCE: checks.append("mixture_above_exact")
    if gap_hi > delta+TOLERANCE: checks.append("exact_gap_above_delta")
    if attack<exact_lo-TOLERANCE: checks.append("attack_below_exact")
    if attack-bound>delta+TOLERANCE: checks.append("posterior_gap_above_delta")
    zero_delta=delta<=1e-12
    if zero_delta and max(abs(gap_lo),abs(gap_hi),abs(attack-bound))>TOLERANCE:
        checks.append("zero_delta_not_numerically_exact")
    return {"table_id":original["table_id"],"group":original["group"],"d":table["d"],
            "target":target,"enemy_results":enemy_rows,"mixture_lower":bound,
            "attack_upper":attack,"posterior_gap":attack-bound,"Delta":delta,
            "existing_MILP_robust_interval":[exact_lo,exact_hi],"exact_gap_interval":[gap_lo,gap_hi],
            "gap_ratio_interval":None if zero_delta else [gap_lo/delta,gap_hi/delta],
            "zero_Delta":zero_delta,"checks_failed":checks,
            "all_chord_LPs_succeeded":all(e["result"]["chord_solver_success"] for e in enemy_rows),
            "all_mixture_LPs_succeeded":all(e["result"]["mixture_result"]["solver_success"] for e in enemy_rows),
            "seconds":time.perf_counter()-started}


def summarize(rows):
    if not rows: return {"objects":0}
    zero=[r for r in rows if r["zero_Delta"]]
    return {"objects":len(rows),"enemy_chord_calls":sum(len(r["enemy_results"]) for r in rows),
            "check_failures":sum(bool(r["checks_failed"]) for r in rows),
            "zero_Delta_count":len(zero),"mean_Delta":float(np.mean([r["Delta"] for r in rows])),
            "mean_exact_gap_interval":np.mean([r["exact_gap_interval"] for r in rows],axis=0),
            "max_exact_gap_upper":max(r["exact_gap_interval"][1] for r in rows),
            "mean_posterior_gap":float(np.mean([r["posterior_gap"] for r in rows])),
            "max_posterior_gap":max(r["posterior_gap"] for r in rows),
            "max_zero_Delta_posterior_gap":max([r["posterior_gap"] for r in zero],default=None),
            "max_exact_gap_minus_Delta":max(r["exact_gap_interval"][1]-r["Delta"] for r in rows),
            "all_chord_LPs_succeeded":all(r["all_chord_LPs_succeeded"] for r in rows),
            "all_mixture_LPs_succeeded":all(r["all_mixture_LPs_succeeded"] for r in rows),
            "seconds":sum(r["seconds"] for r in rows)}


def main():
    if OUTPUT.exists(): raise FileExistsError("Archive existing gap benchmark before rerunning")
    source_before=source_hashes()
    input_hash=digest(INPUT)
    original=json.loads(INPUT.read_text(encoding="utf-8"))
    if original["run_status"]!="complete" or len(original["rows"])!=384:
        raise ValueError("Expected complete fixed 384-object benchmark")
    tables={t["table_id"]:t for t in original["tables"]}
    started=time.perf_counter()
    result={"protocol":"finite_chord_gap_v1","started_utc":datetime.now(timezone.utc).isoformat(),
            "run_status":"running","source_hashes":source_before,"input_path":INPUT.relative_to(ROOT).as_posix(),
            "input_sha256":input_hash,"new_MILP_calls":0,"comparison_tolerance":TOLERANCE,
            "tight_families":[],"rows":[],"planned_objects":384,
            "scope":"Per-object robust LOO on reused random tables; numerical checks, not formal proof"}
    save(result)
    try:
        with threadpool_limits(limits=1):
            for positive in (False,True):
                for m in (3,5,9,17,33):
                    result["tight_families"].append(tight_family(m,positive))
            for row in original["rows"]:
                if time.perf_counter()-started>=120: break
                result["rows"].append(random_object(row,tables[row["table_id"]]))
                if len(result["rows"])%64==0:
                    print(f"chord targets={len(result['rows'])}/384 elapsed={time.perf_counter()-started:.1f}s",flush=True)
                    save(result)
        result["source_hashes_end"]=source_hashes()
        result["source_files_unchanged_during_run"]=source_before==result["source_hashes_end"]
        result["input_unchanged"]=input_hash==digest(INPUT)
        if not result["source_files_unchanged_during_run"] or not result["input_unchanged"]:
            raise RuntimeError("Source/protocol/input changed during benchmark")
        result["summary"]=summarize(result["rows"])
        result["summary_by_group_dimension"]={f"{group}/d{d}":summarize([r for r in result["rows"] if r["group"]==group and r["d"]==d])
            for group in ("heterogeneous_eps","eps_0.01") for d in (2,4)}
        result["plot_data"]={
            "tight_family":[{k:r[k] for k in ("family","m","theoretical_true_M","P_onehot","L_mixture_raw","Delta","gap_ratio","attack_upper","posterior_gap","chord_LP_reference_value")} for r in result["tight_families"]],
            "random_gap":[{k:r[k] for k in ("table_id","group","d","target","Delta","mixture_lower","attack_upper","posterior_gap","exact_gap_interval","zero_Delta")} for r in result["rows"]]}
        result["family_check_failures"]=sum(any(r["checks"].values()) for r in result["tight_families"])
        result["run_status"]="complete" if len(result["rows"])==384 else "partial_time_budget"
        result["seconds"]=time.perf_counter()-started
        result["completed_utc"]=datetime.now(timezone.utc).isoformat()
        save(result)
        print(json.dumps(plain({"status":result["run_status"],"summary":result["summary"],
                               "family_check_failures":result["family_check_failures"]}),indent=2),flush=True)
    except Exception as exc:
        result["run_status"]="failed";result["error"]=repr(exc)
        save(result)
        raise


if __name__=="__main__": main()
