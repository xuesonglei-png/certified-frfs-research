"""Fixed random benchmark of shared witness-mixture LOO certificates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
from datetime import datetime, timezone

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.core import interval_bounds
from certified_frfs.margin import mixture_enemy_bound, triplet_interval_min
from certified_frfs.margin_verification import exact_negative_loo_margin_milp

SEED = 20260923
OUTPUT = ROOT / "results/round3/margin-mixture-benchmark.json"
PROTOCOL = ROOT / "research/round3/mixture-benchmark-protocol.md"
MAX_SECONDS = 120.0


def hashes():
    paths = [Path(__file__), PROTOCOL, *sorted((ROOT / "src").rglob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(result):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def make_tables():
    rng = np.random.default_rng(SEED)
    tables = []
    for d in (2, 4):
        for index in range(48):
            X = rng.uniform(0.1, 0.9, (8, d))
            eps = rng.uniform(0.02, 0.25, (8, d))
            tables.append({"table_id": f"d{d}-t{index:02d}", "d": d, "index": index,
                           "X": X, "heterogeneous_eps": eps, "weights": np.full(d, 1/d),
                           "y": np.array([0]*4+[1]*4)})
    return tables


def benchmark_object(table, group, target):
    started = time.perf_counter()
    X, y, d = table["X"], table["y"], table["d"]
    weights, features = table["weights"], list(range(d))
    eps = table["heterogeneous_eps"] if group == "heterogeneous_eps" else 0.01
    lower, upper = interval_bounds(X, eps, clip=True)
    friends = np.flatnonzero((y == y[target]) & (np.arange(8) != target))
    enemies = np.flatnonzero(y != y[target])
    onehot_start = time.perf_counter()
    onehot_raw = []
    for enemy in enemies:
        individual = [float(weights @ triplet_interval_min(lower[target], upper[target],
                             lower[enemy], upper[enemy], lower[friend], upper[friend]))
                      for friend in friends]
        onehot_raw.append(max(individual))
    onehot_seconds = time.perf_counter() - onehot_start
    enemy_rows = []
    mixture_start = time.perf_counter()
    for enemy, independent_raw in zip(enemies, onehot_raw):
        enemy_start = time.perf_counter()
        mixed = mixture_enemy_bound(lower, upper, target, int(enemy), friends,
                                    features, weights, time_limit=2.0)
        if not np.isclose(mixed["onehot_bound"], independent_raw, atol=1e-12, rtol=0):
            raise AssertionError("Independent triplet and mixture initialization disagree")
        enemy_rows.append({"enemy": int(enemy), "mixture": mixed,
                           "seconds": time.perf_counter()-enemy_start})
    mixture_seconds = time.perf_counter() - mixture_start
    onehot = min(value-1e-10*(1+abs(value)) for value in onehot_raw)
    mixed_bound = min(row["mixture"]["lower_bound"] for row in enemy_rows)
    exact_start = time.perf_counter()
    exact = exact_negative_loo_margin_milp(X, y, target, features, weights,
                                          eps=eps, clip=True, time_limit=2.0)
    exact_seconds = time.perf_counter()-exact_start
    robust_lower, robust_upper = -exact["upper_bound"], -exact["lower_bound"]
    if robust_lower > 1e-8:
        status = "numerically_certified_strict_robust"
    elif robust_upper < -1e-8:
        status = "misclassification_witness"
    else:
        status = "unresolved_or_tie"
    # A feasible point gives an upper bound on the robust minimum. A
    # certificate above that value would be an actual counterexample.
    numerical_inconsistency = mixed_bound > robust_upper+2e-6 or onehot > robust_upper+2e-6
    return {
        "table_id": table["table_id"], "d": d, "group": group, "target": target,
        "friends": friends.tolist(), "enemies": enemies.tolist(),
        "onehot_margin_lower": onehot, "mixture_margin_lower": mixed_bound,
        "bound_improvement": mixed_bound-onehot,
        "onehot_certified": bool(onehot > 1e-10), "mixture_certified": bool(mixed_bound > 1e-10),
        "extra_mixture_certification": bool(mixed_bound > 1e-10 and onehot <= 1e-10),
        "midpoint_separated_all_enemies": all(row["mixture"]["midpoint_separation"] for row in enemy_rows),
        "midpoint_separated_enemy_count": sum(row["mixture"]["midpoint_separation"] for row in enemy_rows),
        "enemy_mixtures": enemy_rows,
        "exact_negative_margin": exact,
        "robust_margin_interval": [robust_lower, robust_upper], "robust_status": status,
        "onehot_to_robust_gap_interval": [robust_lower-onehot, robust_upper-onehot],
        "mixture_to_robust_gap_interval": [robust_lower-mixed_bound, robust_upper-mixed_bound],
        "numerical_certificate_inconsistency": bool(numerical_inconsistency),
        "onehot_seconds": onehot_seconds, "mixture_seconds": mixture_seconds,
        "milp_seconds": exact_seconds, "seconds": time.perf_counter()-started,
    }


def summarize(rows):
    if not rows:
        return {"objects": 0}
    optimal_rows = [row for row in rows if row["exact_negative_margin"]["optimal"]]
    result = {
        "objects": len(rows), "tables_touched": len({r["table_id"] for r in rows}),
        "onehot_certified": sum(row["onehot_certified"] for row in rows),
        "mixture_certified": sum(row["mixture_certified"] for row in rows),
        "extra_mixture_certifications": sum(row["extra_mixture_certification"] for row in rows),
        "strict_bound_improvement_count_tolerance_1e8": sum(row["bound_improvement"] > 1e-8 for row in rows),
        "mean_bound_improvement": float(np.mean([row["bound_improvement"] for row in rows])),
        "max_bound_improvement": max(row["bound_improvement"] for row in rows),
        "robust_status_counts": dict(Counter(row["robust_status"] for row in rows)),
        "milp_status_counts": dict(Counter(str(row["exact_negative_margin"]["status"]) for row in rows)),
        "milp_optimal_count": len(optimal_rows),
        "milp_max_gap": max(row["exact_negative_margin"]["gap"] for row in rows),
        "midpoint_separated_objects": sum(row["midpoint_separated_all_enemies"] for row in rows),
        "midpoint_separated_object_fraction": float(np.mean([row["midpoint_separated_all_enemies"] for row in rows])),
        "midpoint_separated_enemy_fraction": sum(row["midpoint_separated_enemy_count"] for row in rows)/(4*len(rows)),
        "mixture_lp_calls": 4*len(rows),
        "mixture_lp_success_count": sum(e["mixture"]["solver_success"] for row in rows for e in row["enemy_mixtures"]),
        "certificate_inconsistency_count": sum(row["numerical_certificate_inconsistency"] for row in rows),
        "seconds": {key: sum(row[key] for row in rows) for key in ("onehot_seconds", "mixture_seconds", "milp_seconds")},
    }
    for name, subset in (("all", rows), ("milp_optimal_only", optimal_rows)):
        for method in ("onehot", "mixture"):
            gaps = [row[f"{method}_to_robust_gap_interval"] for row in subset]
            result[f"mean_{method}_gap_interval_{name}"] = np.mean(gaps, axis=0).tolist() if gaps else None
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Archive existing output before rerunning: {OUTPUT}")
    start, source_before = time.perf_counter(), hashes()
    tables = make_tables()
    by_id = {table["table_id"]: table for table in tables}
    order = [by_id[f"d{d}-t{index:02d}"] for index in range(48) for d in (2,4)]
    jobs = [(table, group, target) for table in order
            for group in ("heterogeneous_eps", "eps_0.01") for target in (0,4)]
    data = {
        "protocol": "fixed_random_mixture_benchmark_v1", "exploratory": True,
        "seed": SEED, "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "running", "max_seconds": MAX_SECONDS, "milp_time_limit": 2.0,
        "mixture_lp_time_limit": 2.0, "threadpool_limit": 1,
        "tables": tables, "planned_objects": len(jobs), "source_hashes": source_before,
        "inputs_sha256": hashlib.sha256(json.dumps(plain(tables), sort_keys=True,
                              separators=(",", ":")).encode()).hexdigest(),
        "interpretation": "Finite random-table robust LOO benchmark; all generated tables retained. "
                          "No prediction/generalization guarantee and no selection of favorable cases.",
        "rows": [],
    }
    save(data)
    try:
        with threadpool_limits(limits=1):
            for table, group, target in jobs:
                if time.perf_counter()-start >= MAX_SECONDS:
                    break
                row = benchmark_object(table, group, target)
                data["rows"].append(row)
                if len(data["rows"]) % 16 == 0:
                    data["summary"] = summarize(data["rows"])
                    data["seconds"] = time.perf_counter()-start
                    save(data)
                    print(f"objects={len(data['rows'])}/384, elapsed={data['seconds']:.1f}s, "
                          f"extra_certified={data['summary']['extra_mixture_certifications']}, "
                          f"MILP_optimal={data['summary']['milp_optimal_count']}", flush=True)
        data["source_hashes_end"] = hashes()
        data["source_files_unchanged_during_run"] = data["source_hashes_end"] == source_before
        if not data["source_files_unchanged_during_run"]:
            raise RuntimeError("Source or protocol changed during benchmark")
        data["summary"] = summarize(data["rows"])
        data["summary_by_group_dimension"] = {
            f"{group}/d{d}": summarize([r for r in data["rows"] if r["group"] == group and r["d"] == d])
            for group in ("heterogeneous_eps", "eps_0.01") for d in (2,4)}
        data["summary_by_group"] = {
            group: summarize([r for r in data["rows"] if r["group"] == group])
            for group in ("heterogeneous_eps", "eps_0.01")}
        completed = {(r["table_id"],r["group"],r["target"]) for r in data["rows"]}
        data["pending_objects"] = [{"table_id":t["table_id"],"group":g,"target":i}
            for t,g,i in jobs if (t["table_id"],g,i) not in completed]
        data["run_status"] = "complete" if not data["pending_objects"] else "partial_time_budget"
        data["seconds"] = time.perf_counter()-start
        data["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save(data)
        print(json.dumps(plain({"run_status":data["run_status"],"summary":data["summary"],
                               "by_group":data["summary_by_group"]}),indent=2), flush=True)
    except Exception as error:
        data["run_status"] = "failed"
        data["error"] = repr(error)
        data["seconds"] = time.perf_counter()-start
        save(data)
        raise


if __name__ == "__main__":
    main()
