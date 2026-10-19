"""Fixed-split predictive scoring for the majority-bound candidates.

Every row is evaluated on the validation split already frozen by the source
artifact.  No validation score is used to choose a candidate or a method.  The
recent FGS/SFRIP controls are reused only when their native support is stored
for the same reference/validation split; legacy FGS120 artifacts are never
substituted for a mismatched split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_majority_bound_prediction_v1 import (
    CAP, FOLDER, NEWDATA, ROOT, SEARCH, digest, source_split, fresh_training_certificate
)
from certified_frfs.redistributed_loss import transfer_weights
from geometric_vote_evaluation import final_evaluation
from certified_frfs.validated_pool import preprocess_fit


TASKS = tuple(sorted(p for p in SEARCH.glob("*.json") if not p.name.startswith("complete")))


def frozen_metric(features, weights):
    fs = [int(v) for v in features]
    w = np.asarray(weights, float)
    assert len(fs) == len(w) and len(set(fs)) == len(fs)
    order = np.argsort(fs, kind="stable"); fs = [fs[i] for i in order]; w = w[order]
    return {"features": fs, "weights": w.tolist()}


def full_metric(d):
    return frozen_metric(range(d), np.ones(d) / np.sqrt(d))


def compact_candidate(block, candidate):
    c = block["fit"]["candidates"][int(candidate)]
    fs = c["features"]; w = np.asarray(c["weights"], float)
    if len(w) == len(fs):
        reduced = w
    else:
        reduced = w[np.asarray(fs, int)]
    return frozen_metric(fs, reduced), c


def scenario_metric(task, d):
    path = ROOT / "results" / "scenario-repair-diagnostic" / (
        f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}.json")
    if not path.exists():
        return None, {"status": "missing", "reason": f"scenario source missing: {path.relative_to(ROOT)}"}
    data = json.loads(path.read_text(encoding="utf-8")); fit = data["fit"]
    fs = fit["features"]; w = np.asarray(fit["weights"], float)
    if len(w) == d:
        w = w[np.asarray(fs, int)]
    return frozen_metric(fs, w), {"source": str(path.relative_to(ROOT)), "certified": bool(fit.get("certified", False))}


def recent_controls(source, split_index, group, name):
    """Load same-split native FGS/SFRIP supports from audited trajectories."""
    if group == "original15":
        path = ROOT / "results" / "trajectory-recent-comparison" / f"{name}.json"
    else:
        path = NEWDATA / f"{name}.json"
    if not path.exists():
        return {"weighted_fgs": {"status": "missing", "reason": "no same-split recent trajectory"},
                "fgs_uniform_native": {"status": "missing", "reason": "no same-split recent trajectory"},
                "sfrip": {"status": "missing", "reason": "no same-split recent trajectory"},
                "sfrip_uniform_native": {"status": "missing", "reason": "no same-split recent trajectory"}}
    prior = json.loads(path.read_text(encoding="utf-8")); row = prior["splits"][split_index]
    # Confirm the exact split, rather than trusting an equal index.
    if group == "original15":
        ref = source["splits"][split_index]["reference_indices"]
        val = source["splits"][split_index]["validation_indices"]
    else:
        ref = source["splits"][split_index]["train_indices"]
        val = source["splits"][split_index]["test_indices"]
    row_ref = row.get("reference_indices", row.get("train_indices"))
    row_val = row.get("validation_indices", row.get("test_indices"))
    if row_ref != ref or row_val != val:
        return {k: {"status": "missing", "reason": "recent trajectory split provenance mismatch"}
                for k in ("weighted_fgs", "fgs_uniform_native", "sfrip", "sfrip_uniform_native")}
    fgs = row["baselines"]["fgs2023"]["selection"]["features"]
    sfrip = row["baselines"]["sfrip2025"]["selection"]["features"]
    return {
        "weighted_fgs": {"features": fgs, "kind": "transfer"},
        "fgs_uniform_native": {"features": fgs, "kind": "uniform"},
        "sfrip": {"features": sfrip, "kind": "transfer"},
        "sfrip_uniform_native": {"features": sfrip, "kind": "uniform"},
    }


def run(path):
    saved = json.loads(path.read_text(encoding="utf-8")); task = saved["task"]
    X, y, ri, vi, source_path, source, block = source_split(task)
    A, B = preprocess_fit(X[ri], X[vi]); d = A.shape[1]; full = np.ones(d) / np.sqrt(d)
    direct = json.loads((SEARCH / path.name).read_text(encoding="utf-8"))["fit"]
    direct_metric = frozen_metric(direct["features"], direct["weights"])
    original_metric, original = compact_candidate(block, task["candidate"])
    scenario, scenario_info = scenario_metric(task, d)
    methods = {
        "original_candidate": {"metric": original_metric, "provenance": {"source": str(source_path.relative_to(ROOT)), "candidate": int(task["candidate"])}},
        "scenario": {"metric": scenario, "provenance": scenario_info} if scenario else scenario_info,
        "proposed": {"metric": direct_metric, "provenance": {"source": str((SEARCH / path.name).relative_to(ROOT)), "kind": "majority_bound_search_raw"}},
    }
    # Deployment rule is fixed before validation values are computed.
    direct_cert = fresh_training_certificate(A, y[ri], direct_metric["features"], np.asarray(direct_metric["weights"]))
    compact_metric, compact = compact_candidate(block, task["candidate"])
    compact_cert = fresh_training_certificate(A, y[ri], compact_metric["features"], np.asarray(compact_metric["weights"]))
    if direct_cert["feasible"]:
        deployment_metric, deployment_origin = direct_metric, "proposed_certified"
    elif compact_cert["feasible"]:
        deployment_metric, deployment_origin = compact_metric, "source_candidate_certified"
    else:
        deployment_metric, deployment_origin = full_metric(d), "full_fallback"
    methods["deployment"] = {"metric": deployment_metric, "provenance": {"origin": deployment_origin, "certified_before_test": True}}
    methods["full"] = {"metric": full_metric(d), "provenance": {"origin": "uniform_full"}}
    controls = recent_controls(source, int(task["split"]), task["group"], task["dataset"])
    for name, info in controls.items():
        if info.get("status") == "missing":
            methods[name] = info; continue
        fs = info["features"]
        weights = transfer_weights(A, fs, full) if info["kind"] == "transfer" else np.ones(len(fs))
        methods[name] = {"metric": frozen_metric(fs, weights),
                         "provenance": {"same_split": True, "kind": info["kind"]}}
    # Matched-count controls need a complete ranking.  Native support alone
    # cannot be turned into a fair matched-count method, so preserve missing.
    for name in ("fgs_matched", "sfrip_matched", "fgs_uniform_matched", "sfrip_uniform_matched"):
        methods[name] = {"status": "missing", "reason": "native support artifact has no complete ranking for matched-count construction"}
    for name, record in methods.items():
        if record.get("status") == "missing": continue
        metric = record["metric"]
        record["final_evaluation"] = final_evaluation(A, y[ri], B, y[vi], metric)
        record["fresh_certificate"] = fresh_training_certificate(A, y[ri], metric["features"], np.asarray(metric["weights"]))
    source_files = [Path(__file__), ROOT / "scripts" / "audit_majority_bound_prediction_v1.py", source_path,
                    SEARCH / path.name, ROOT / "src" / "certified_frfs" / "validated_pool.py",
                    ROOT / "src" / "certified_frfs" / "redistributed_loss.py"]
    if scenario:
        source_files.append(ROOT / "results" / "scenario-repair-diagnostic" / path.name)
    baseline_path = ROOT / "results" / "trajectory-recent-comparison" / f"{task['dataset']}.json"
    if baseline_path.exists(): source_files.append(baseline_path)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files if p.exists()}
    out = {"task": task, "reference_indices": ri.tolist(), "validation_indices": vi.tolist(),
           "methods": methods, "source_hashes": hashes, "source_fingerprints_verified": True,
           "query_labels_used_for_selection": False, "complete": True,
           "scope": "fixed_candidate_validation_scoring_no_candidate_or_method_selection"}
    out_path = FOLDER / path.name; out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path.stem, "scored", deployment_origin, flush=True)
    return out


def summarize(rows):
    metrics = {}; sizes = {}; datasets = {}
    for row in rows:
        dname = row["task"]["dataset"]; split = int(row["task"]["split"]); datasets.setdefault(dname, []).append(row)
        for name, rec in row["methods"].items():
            if rec.get("status") == "missing": continue
            ev = rec["final_evaluation"]["evaluation"]["metrics"]
            metrics.setdefault(name, []).append(ev); sizes.setdefault(name, []).append(len(rec["metric"]["features"]))
    means = {m: {k: float(np.mean([x[k] for x in vals])) for k in vals[0]} for m, vals in metrics.items()}
    dataset_means = {}
    for name, rs in datasets.items():
        dataset_means[name] = {}
        for method in set().union(*(r["methods"] for r in rs)):
            vals = [r["methods"][method]["final_evaluation"]["evaluation"]["metrics"] for r in rs
                    if method in r["methods"] and r["methods"][method].get("status") != "missing"]
            if vals: dataset_means[name][method] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    paired = {}
    for method in metrics:
        if method == "proposed": continue
        pairs = [(r["methods"]["proposed"]["final_evaluation"]["evaluation"]["metrics"],
                  r["methods"][method]["final_evaluation"]["evaluation"]["metrics"])
                 for r in rows if method in r["methods"] and r["methods"][method].get("status") != "missing"]
        if pairs:
            paired[method] = {"n": len(pairs),
                "proposed_minus_control_clean_BA_pp": float(100*np.mean([a["clean_BA"]-b["clean_BA"] for a,b in pairs])),
                "proposed_minus_control_perturbed_BA_pp": float(100*np.mean([a["mean_perturbed_BA"]-b["mean_perturbed_BA"] for a,b in pairs])),
                "clean_wins": int(sum(a["clean_BA"] > b["clean_BA"] + 1e-12 for a,b in pairs)),
                "clean_ties": int(sum(abs(a["clean_BA"]-b["clean_BA"]) <= 1e-12 for a,b in pairs)),
                "clean_losses": int(sum(a["clean_BA"] < b["clean_BA"] - 1e-12 for a,b in pairs))}
    return {"candidates": len(rows), "datasets": len(datasets), "method_coverage": {m: len(v) for m,v in metrics.items()},
            "means_over_candidate_splits": means, "paired_descriptive_contrasts": paired,
            "mean_features_over_candidate_splits": {m: float(np.mean(v)) for m, v in sizes.items()},
            "dataset_equal_weight_means": dataset_means,
            "scope": "development_exposed_fixed_split_descriptive_no_SOTA_or_pvalue_claim"}


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    rows = [run(p) for p in TASKS]
    complete = summarize(rows)
    complete["source_fingerprints_verified"] = True
    (FOLDER / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in complete.items() if k != "dataset_equal_weight_means"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
