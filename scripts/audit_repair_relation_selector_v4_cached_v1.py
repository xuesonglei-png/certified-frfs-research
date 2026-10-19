"""Versioned sequential cache driver around the frozen independent v4 auditor.

A hit reuses an already completed independent check only when every numerical
input and the complete saved candidate/score have identical SHA-256 hashes.
Cached occurrences are counted separately from newly replayed checks. All
learning, split, restoration and recipe selections are replayed by the frozen
auditor for every occurrence. No production or frozen auditor file is changed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import audit_repair_relation_selector_v4 as frozen_audit
from audit_relation_support_selector_v1 import equivalent

OUT = ROOT / "results/repair-relation-selector-v4"
TARGETS = ("heart_statlog", "seeds", "ionosphere", "banknote", "wdbc")
ORIGINAL_CANDIDATE = frozen_audit.check_candidate
ORIGINAL_SCORE = frozen_audit.check_score
ORIGINAL_FIT = frozen_audit.check_fit
ORIGINAL_TRACE = frozen_audit.replay_trace
STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    """Typed full-input digest including array shape/dtype/raw data and order."""
    if isinstance(value, np.ndarray):
        header = json.dumps({"type": "ndarray", "dtype": value.dtype.str,
                             "shape": list(value.shape)}, sort_keys=True).encode()
        return hashlib.sha256(header + b"\0" + np.ascontiguousarray(value).tobytes()).hexdigest()
    def typed(v):
        if isinstance(v, np.ndarray):
            return {"__ndarray__": object_hash(v)}
        if isinstance(v, dict):
            return {k: typed(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return {"__sequence_type__": type(v).__name__, "values": [typed(x) for x in v]}
        if isinstance(v, np.generic):
            return {"__numpy_scalar__": str(v.dtype), "value": v.item()}
        return v
    payload = json.dumps(typed(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cached_check(kind, original, inputs, counts):
    components = [object_hash(x) for x in inputs]
    key = hashlib.sha256(json.dumps({"kind": kind, "components": components},
                                    sort_keys=True).encode()).hexdigest()
    cache = STATE["cache"][kind]
    STATE["occurrences"][kind] += 1
    if key in cache:
        assert cache[key]["components"] == components
        STATE["hits"][kind] += 1
        counts["cached_" + kind + "_occurrences"] += 1
        return
    before = dict(counts)
    original(*inputs, counts)
    delta = {k: int(v - before.get(k, 0)) for k, v in counts.items() if v != before.get(k, 0)}
    cache[key] = {"components": components, "count_delta": delta}
    STATE["misses"][kind] += 1


def check_candidate(A, y, reference, fs, candidate, counts):
    return cached_check("candidate", ORIGINAL_CANDIDATE,
                        (A, y, reference, fs, candidate), counts)


def check_score(A, y, B, truth, model, saved, counts):
    return cached_check("score", ORIGINAL_SCORE, (A, y, B, truth, model, saved), counts)


def replay_trace(A, y, reference, rank, requested, trace, model, counts):
    assert trace and requested >= int(np.ceil(A.shape[1] / 2))
    assert isinstance(model["uniform_fallback"], bool)
    assert sorted(rank) == list(range(A.shape[1]))
    count = len(model["features"])
    expected = A.shape[1] - requested + 2 if model["uniform_fallback"] else count - requested + 1
    assert len(trace) == expected
    assert count >= requested and count <= A.shape[1]
    return ORIGINAL_TRACE(A, y, reference, rank, requested, trace, model, counts)


def check_fit(X, y, fit, counts):
    assert fit["seed"] == STATE["expected_seed"]
    assert fit["inner_splits"] == min(3, int(np.unique(y, return_counts=True)[1].min()))
    assert len(fit["inner"]) == fit["inner_splits"]
    for block in fit["inner"]:
        assert set(block["families"]) == set(frozen_audit.FAMILIES)
        for family in frozen_audit.FAMILIES:
            part = block["families"][family]
            assert len(part["candidates"]) == len(frozen_audit.FRACTIONS)
    ORIGINAL_FIT(X, y, fit, counts)
    STATE["expected_seed"] += 1


def dependency_hashes():
    files = [Path(__file__), ROOT / "scripts/audit_repair_relation_selector_v4.py",
             ROOT / "scripts/audit_selective_support_selector_v3.py",
             ROOT / "scripts/audit_relation_support_selector_v1.py",
             ROOT / "scripts/audit_majority_bound_prediction_v1.py"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def audit(name):
    threadpool_limits(1)
    start = time.perf_counter()
    artifact = OUT / (name + ".json")
    failure_path = OUT / (name + "-cached-v1-audit-failure-" + str(time.time_ns()) + ".json")
    report_path = OUT / (name + "-cached-v1-audit.json")
    cache_path = OUT / (name + "-cached-v1-audit-cache.json")
    hashes = dependency_hashes()
    STATE.clear()
    STATE.update({"cache": {"candidate": {}, "score": {}}, "hits": Counter(),
                  "misses": Counter(), "occurrences": Counter(), "expected_seed": 20261011})
    try:
        frozen_hashes = frozen_audit.freeze()
        source = json.loads(artifact.read_text(encoding="utf-8"))
        assert source["dataset"] == name and source["complete"]
        assert source["source_hashes"] == frozen_hashes
        assert len(source["rows"]) == 3
        artifact_sha = digest(artifact)
        for row in source["rows"]:
            assert len(row["train_indices"]) and len(row["test_indices"])
            assert set(row["shared_perturbations"]) == {"proposed", "full", "matched_uniform"}
            for method in row["shared_perturbations"]:
                assert len(row["shared_perturbations"][method]) == 8
                flags = np.asarray(row[method]["query_certificate"]["certified"], bool)
                for perturbed in row["shared_perturbations"][method]:
                    assert not np.any(flags & (np.asarray(perturbed["predictions"]) != row[method]["predictions"]))
        frozen_audit.check_candidate = check_candidate
        frozen_audit.check_score = check_score
        frozen_audit.check_fit = check_fit
        frozen_audit.replay_trace = replay_trace
        frozen_audit.audit(name)
        assert digest(artifact) == artifact_sha
        frozen_audit.freeze()
        assert dependency_hashes() == hashes
        core = json.loads((OUT / (name + "-audit.json")).read_text())
        assert core["passed"] and core["counts"]["outer_folds"] == 3
        ledger = {"key_scheme": "sha256 of typed complete inputs including complete saved record",
                  "entries": STATE["cache"], "dependency_hashes": hashes,
                  "artifact_sha256": artifact_sha}
        cache_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        report = {**core, "driver_sha256": hashes[Path(__file__).relative_to(ROOT).as_posix()],
                  "audit_dependency_hashes": hashes, "driver_seconds": time.perf_counter() - start,
                  "cache_occurrences": dict(STATE["occurrences"]), "cache_hits": dict(STATE["hits"]),
                  "cache_misses": dict(STATE["misses"]), "cache_ledger_sha256": digest(cache_path),
                  "cache_reuse": "exact hashed identical checks within this dataset audit only",
                  "limits": core["limits"] + [
                      "cached checks are previous successful independent replay, not new repeated arithmetic",
                      "optimizer/ranking replay remains production-based",
                      "different metadata in model records prevents score cache reuse even for equal predictions"]}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("CACHED AUDIT COMPLETE", name, "hits", dict(STATE["hits"]), "misses", dict(STATE["misses"]), flush=True)
        return report
    except Exception as error:
        failure = {"passed": False, "dataset": name, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "audit_dependency_hashes": hashes, "cache_hits": dict(STATE["hits"]),
                   "cache_misses": dict(STATE["misses"]), "seconds": time.perf_counter() - start}
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print("CACHED AUDIT FAILED", name, failure_path, flush=True)
        return failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=TARGETS)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    results = []
    for name in ((args.only,) if args.only else TARGETS):
        deadline = time.monotonic() + args.wait_seconds
        while True:
            path = OUT / (name + ".json")
            ready = path.exists() and json.loads(path.read_text())["complete"]
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("artifact not complete: " + name)
            print("waiting for complete artifact", name, flush=True)
            time.sleep(min(30, max(1, deadline - time.monotonic())))
        results.append(audit(name))
    report = {"complete": True, "passed": all(r["passed"] for r in results),
              "datasets": [r["dataset"] for r in results], "results": results,
              "one_sequential_audit_process": True, "BLAS_threads": 1}
    (OUT / "cached-v1-audit-complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
