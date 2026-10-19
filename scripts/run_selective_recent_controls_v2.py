"""Append-only FGS/SFRIP controls on existing nested selector splits.

FGS uses the documented 120-configuration superset; SFRIP uses its six-
configuration beta/confidence grid. Native supports are preserved. Fixed-count
rankings and capped weight transfer are explicitly labelled adaptations.
No outer labels enter configuration choice and no parent artifact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.datasets import load_dataset
from certified_frfs.recent_grid_fast import fgs_paths_grid, sfrip_paths_grid
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.selective_support_selector_v3 import query_scores as _query_scores
from certified_frfs.validated_pool import preprocess_fit
from expanded_fgs_grid import BETAS, XIS, CONFIGS as FGS_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "selective-recent-controls-v2"
SFRIP_CONFIGS = tuple({"beta": b, "confidence": c, "k": 10, "support": 3, "alpha": 0.}
                     for b in (2., 3., 5.) for c in (.6, .8))
VARIANTS = ("native_uniform", "native_capped_transfer", "matched_uniform", "matched_capped_transfer")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(source_path):
    files = list((ROOT/"src/certified_frfs").glob("*.py"))
    files += [Path(__file__), ROOT/"scripts/audit_selective_recent_controls_v2.py",
              ROOT/"scripts/expanded_fgs_grid.py", source_path,
              ROOT/"scripts/audit_multiscale_certified_metric_knn_pool.py",
              ROOT/"scripts/audit_relation_support_selector_v1.py",
              source_path.parent/"source-freeze.json",
              ROOT/"data/processed"/(source_path.stem+".npz")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def freeze(name, source_path):
    OUT.mkdir(exist_ok=True)
    hashes = fingerprints(source_path)
    target = OUT/(name+"-source-freeze.json")
    if target.exists():
        assert json.loads(target.read_text())["hashes"] == hashes, "source changed; new version required"
    else:
        target.write_text(json.dumps({"hashes": hashes, "source_folder": source_path.parent.name,
                                     "FGS_configs": list(FGS_CONFIGS), "SFRIP_configs": list(SFRIP_CONFIGS),
                                     "clean_tolerance": .005, "query_eps": .01,
                                     "unit_mass_cap": "2/d", "total_mass": "sqrt(d)"}, indent=2), encoding="utf8")
    return hashes


def paths(A, y):
    # Limit peak memory by batching one xi, with identical arithmetic/configs.
    lookup = {}
    for xi in XIS:
        configs = [{"beta": b, "xi": xi} for b in BETAS]
        for config, path in zip(configs, fgs_paths_grid(A, y, configs), strict=True):
            lookup[(config["beta"], config["xi"])] = path
    return {"fgs": [(dict(c), lookup[(c["beta"], c["xi"])]) for c in FGS_CONFIGS],
            "sfrip": list(zip(SFRIP_CONFIGS, sfrip_paths_grid(A, y, SFRIP_CONFIGS), strict=True))}


def empty_score(y, truth):
    classes = np.unique(y)
    p = (np.array([np.sum(y == c) for c in classes])+1)/(len(y)+len(classes))
    pred = np.repeat(classes[p.argmax()], len(truth))
    own = np.searchsorted(classes, truth)
    groups = [truth == c for c in np.unique(truth)]
    f1 = []
    for c in np.union1d(truth, pred):
        tp = np.sum((truth == c) & (pred == c))
        denominator = 2*tp + np.sum((truth != c) & (pred == c)) + np.sum((truth == c) & (pred != c))
        f1.append(0. if denominator == 0 else 2*tp/denominator)
    return {"clean_BA": float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)])),
            "balanced_logloss": float(np.mean([np.mean(-np.log(p[own[truth == c]])) for c in np.unique(truth)])),
            "macro_F1": float(np.mean(f1)), "balanced_query_coverage": 0., "selective_accuracy": None,
            "class_certified_counts": [0]*len(groups), "class_certified_correct_counts": [0]*len(groups),
            "class_query_counts": [int(g.sum()) for g in groups],
            "query_certificate": {"certified": [False]*len(truth), "lower_bounds": [None]*len(truth),
                                  "coverage": 0., "scope": "empty_support_prior_uncertified"},
            "query_certified_BA": 0., "query_coverage": 0., "predictions": pred.tolist(),
            "probabilities": np.tile(p, (len(truth), 1)).tolist(), "empty_support_prior": True}


def build_metric(A, path, variant, count):
    d = A.shape[1]
    native = sorted(path["native"]["features"])
    fs = sorted(path["ranking"][:count]) if variant.startswith("matched") else native
    record = {"features": fs, "raw_native_support": native, "ranking": path["ranking"],
              "variant": variant, "count": len(fs), "path_adaptation": path.get("adaptation"),
              "weighting_adaptation": "common_5NN_uniform"}
    if not fs:
        return {**record, "weights": [], "status": "empty_native_support_prior"}
    if variant.endswith("capped_transfer"):
        if 2*len(fs) < d:
            return {**record, "weights": None, "status": "infeasible_original_2_over_d_unit_mass_cap",
                    "weighting_adaptation": "unavailable_no_cap_relaxation"}
        mass = np.sqrt(d)
        full = np.ones(d)/mass
        raw = transfer_weights(A, fs, full)
        weights = capped_sparse_projection(raw/mass, len(fs), 2/d)*mass
        assert abs(weights.sum()-mass) <= 1e-9 and weights.max() <= 2*mass/d+1e-9
        return {**record, "raw_transferred_weights": raw.tolist(), "weights": weights.tolist(),
                "status": "available", "unit_mass_cap": 2/d, "total_mass": mass,
                "weighting_adaptation": "training_Pearson_transfer_capped_mass_common_5NN"}
    return {**record, "weights": (np.ones(len(fs))/np.sqrt(len(fs))).tolist(), "status": "available"}


def score(A, y, B, truth, metric, cache):
    if metric["weights"] is None:
        return None
    key = (tuple(metric["features"]), tuple(metric["weights"]))
    if key not in cache:
        cache[key] = (_query_scores(A, y, B, truth, metric) if metric["features"] else empty_score(y, truth))
    return cache[key]


def block(A, y, B, truth, count):
    registry = paths(A, y)
    cache, options = {}, {}
    for algorithm, rows in registry.items():
        for variant in VARIANTS:
            method = algorithm+"_"+variant
            options[method] = []
            for i, (config, path) in enumerate(rows):
                metric = build_metric(A, path, variant, count)
                options[method].append({"config_index": i, "config": dict(config), "metric": metric,
                                        "score": score(A, y, B, truth, metric, cache)})
    return {"matched_proposed_count": count, "raw_paths": {a: [{"config": c, "path": p} for c, p in rows] for a, rows in registry.items()},
            "options": options, "unique_scored_metrics": len(cache)}


def choose(inners, method):
    combined = []
    for i, first in enumerate(inners[0]["options"][method]):
        entries = [row["options"][method][i] for row in inners]
        assert all(e["config"] == first["config"] for e in entries)
        if any(e["score"] is None for e in entries):
            combined.append({"config_index": i, "config": first["config"], "status": "not_available_in_all_inner_folds"})
            continue
        combined.append({"config_index": i, "config": first["config"], "status": "available",
                         "mean_BA": float(np.mean([e["score"]["clean_BA"] for e in entries])),
                         "mean_certified_BA": float(np.mean([e["score"]["query_certified_BA"] for e in entries])),
                         "mean_coverage": float(np.mean([e["score"]["query_coverage"] for e in entries])),
                         "mean_count": float(np.mean([e["metric"]["count"] for e in entries]))})
    valid = [r for r in combined if r["status"] == "available"]
    if not valid:
        return {"summary": combined, "selected_config_index": None, "status": "no_available_nested_config"}
    best = max(r["mean_BA"] for r in valid)
    eligible = [r for r in valid if r["mean_BA"] >= best-.005-1e-12]
    chosen = max(eligible, key=lambda r: (r["mean_certified_BA"], r["mean_coverage"], r["mean_BA"], -r["mean_count"], -r["config_index"]))
    return {"summary": combined, "selected_config_index": chosen["config_index"], "chosen": chosen,
            "selection_rule": "within_.005_of_best_clean_BA_then_max_certified_correct_BA_coverage_clean_BA_smallest_support_index"}


def run(name, source_folder, only_fold=None):
    threadpool_limits(1)
    source_path = ROOT/"results"/source_folder/(name+".json")
    if not source_path.exists():
        raise FileNotFoundError(f"source artifact not ready: {source_path}")
    source_hash = sha(source_path)
    source = json.loads(source_path.read_text(encoding="utf8"))
    if not source.get("complete", False):
        raise ValueError("source is incomplete; baseline trial postponed")
    hashes = freeze(name, source_path)
    X, y = load_dataset(name)
    result_rows = []
    for row in source["rows"]:
        if only_fold is not None and row["fold"] != only_fold:
            continue
        if row.get("status", "complete") != "complete":
            result_rows.append({"fold": row["fold"], "status": "source_fold_unavailable", "source_status": row.get("status")})
            continue
        fit = row["fit"]
        ri, vi = np.array(row["train_indices"]), np.array(row["test_indices"])
        assert not set(ri) & set(vi)
        raw, labels = X[ri], y[ri]
        chosen_family = fit["chosen"]["family"]
        chosen_recipe = fit["chosen"]["recipe_index"]
        inners = []
        for training_fold in fit["inner"]:
            candidate = training_fold["families"][chosen_family]["candidates"][chosen_recipe]
            assert candidate["recipe_index"] == chosen_recipe and candidate["score"] is not None
            count = len(candidate["candidate"]["features"])
            ii, jj = np.array(training_fold["fit_indices"]), np.array(training_fold["validation_indices"])
            assert not set(ii) & set(jj) and len(set(ii)|set(jj)) == len(raw)
            A, B = preprocess_fit(raw[ii], raw[jj])
            inner = block(A, labels[ii], B, labels[jj], count)
            inners.append({"fit_indices": ii.tolist(), "validation_indices": jj.tolist(),
                           "source_selected_family": chosen_family, "source_selected_recipe_index": chosen_recipe,
                           "source_realized_support": candidate["candidate"]["features"], **inner})
            print(name, row["fold"], "inner", len(inners), "recent controls scored", inner["unique_scored_metrics"], flush=True)
        selection = {m: choose(inners, m) for m in inners[0]["options"]}
        A, B = preprocess_fit(raw, X[vi])
        actual_count = len(fit["selected"]["features"])
        outer_paths = paths(A, labels)
        results, outer_cache = {}, {}
        for method, picked in selection.items():
            index = picked["selected_config_index"]
            if index is None:
                results[method] = {"status": "no_available_nested_config", "score": None}
                continue
            algorithm, variant = method.split("_", 1)
            config, path = outer_paths[algorithm][index]
            metric = build_metric(A, path, variant, actual_count)
            results[method] = {"config_index": index, "config": dict(config), "metric": metric,
                               "score": score(A, labels, B, y[vi], metric, outer_cache)}
        result_rows.append({"fold": row["fold"], "status": "complete", "train_indices": ri.tolist(), "test_indices": vi.tolist(),
                            "inner": inners, "selection": selection, "outer": {"matched_proposed_count": actual_count,
                                "raw_paths": {a: [{"config": c, "path": p} for c, p in records] for a, records in outer_paths.items()}, "results": results},
                            "proposed": _query_scores(A, labels, B, y[vi], fit["selected"])})
        print(name, row["fold"], "RECENT CONTROLS COMPLETE", flush=True)
    OUT.mkdir(exist_ok=True)
    tag = name+("-fold"+str(only_fold) if only_fold is not None else "")
    out = OUT/(tag+".json")
    assert sha(source_path) == source_hash, "parent source changed during baseline trial"
    assert fingerprints(source_path) == hashes, "pre-run source changed during baseline trial"
    saved = {"dataset": name, "d": X.shape[1], "rows": result_rows, "source_folder": source_folder,
             "source_hashes": hashes,
             "fingerprints": {"source": source_hash, "data": sha(ROOT/"data"/"processed"/(name+".npz")), "runner": sha(__file__),
                              "recent_grid_fast": sha(ROOT/"src"/"certified_frfs"/"recent_grid_fast.py")},
             "grid_sizes": {"fgs": len(FGS_CONFIGS), "sfrip": len(SFRIP_CONFIGS)},
             "outer_labels_used_for_selection": False, "independent_audit_required": True, "goal_achieved": False, "complete": True,
             "scope": "nested_recent_numeric_reproductions_and_labelled_common_classifier_matched_weight_adaptations_exposed_development"}
    if out.exists():
        raise FileExistsError(f"append-only result already exists: {out}")
    out.write_text(json.dumps(saved, separators=(",", ":")), encoding="utf8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True)
    parser.add_argument("--source-folder", default="selective-support-selector-v3")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    print(run(args.only, args.source_folder, args.fold))
