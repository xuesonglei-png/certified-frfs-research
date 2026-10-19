"""Run a fixed ALL15 development comparison against weighted FGS120."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_multiscale_certified_metric_knn_pool import evaluate
from robustaware_projection_v1 import fit_robustaware_projection
from run_simplex_soft5_knn_pilot import final_evaluation
from continue_soft_topk_knn_pilot import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "fgs120-atomic-count-controls"
OUT = ROOT / "results" / "robustaware-projection-v1"
FREEZE = ROOT / "research" / "robustaware-projection-v1-source-freeze.json"
DATASETS = ("banknote", "ecoli", "glass", "haberman", "heart_statlog", "ionosphere",
            "iris", "seeds", "sonar", "transfusion", "vehicle", "vertebral", "wdbc",
            "wine", "yeast")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources():
    files = sorted((ROOT / "scripts").glob("*.py"))
    files += sorted((ROOT / "src" / "certified_frfs").glob("*.py"))
    files += [ROOT / "research" / "robustaware-projection-v1-protocol.md"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def run(name):
    source = json.loads((BASE / (name + ".json")).read_text())
    source_audit = verify(BASE)
    assert source_audit["artifact_hashes"][name] == digest(BASE / (name + ".json"))
    X, y = load_dataset(name)
    ri, vi = np.asarray(source["reference_indices"]), np.asarray(source["validation_indices"])
    A, B = preprocess_fit(X[ri], X[vi]); labels, truth = y[ri], y[vi]
    metric = source["outer"]["results"]["fgs_weighted120_atomic_count"]["metric"]
    fs, anchor = metric["features"], np.asarray(metric["weights"], float)
    baseline = final_evaluation(A, labels, B, truth, metric)["evaluation"]
    started = time.perf_counter()
    try:
        weights, fit = fit_robustaware_projection(A, labels, fs, anchor, draws=8)
        selected_metric = {"features": fs, "weights": weights.tolist()}
        proposed = final_evaluation(A, labels, B, truth, selected_metric)["evaluation"]
        error = None
    except Exception as exc:
        weights, fit, proposed = anchor.copy(), {"status": "error"}, None
        error = f"{type(exc).__name__}: {exc}"
    row = {
        "dataset": name, "dimension": int(X.shape[1]), "features": fs,
        "anchor": anchor.tolist(), "selected_weights": weights.tolist(),
        "reference_indices": ri.tolist(), "validation_indices": vi.tolist(),
        "source_sha256": digest(BASE / (name + ".json")),
        "source_audit_sha256": digest(BASE / "complete-audit.json"),
        "baseline_metrics": baseline["metrics"],
        "proposed_metrics": None if proposed is None else proposed["metrics"],
        "fit": fit, "error": error, "seconds": time.perf_counter() - started,
        "heldout_labels_used_for_selection": False,
    }
    (OUT / (name + ".json")).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(name, fit.get("status"),
          None if proposed is None else round(100 * proposed["metrics"]["clean_BA"], 4), flush=True)
    return row


def main():
    threadpool_limits(1); OUT.mkdir(exist_ok=True)
    if not FREEZE.exists():
        FREEZE.write_text(json.dumps({"created_unix": time.time(), "fingerprints": freeze_sources(),
            "scope": "fixed_all15_weighted_FGS120_anchor_train_only_robust_projection"}, indent=2), encoding="utf-8")
    frozen = json.loads(FREEZE.read_text()); assert frozen["fingerprints"] == freeze_sources()
    rows = [run(name) for name in DATASETS]
    complete = [r for r in rows if r["proposed_metrics"] is not None]
    deltas = [100 * (r["proposed_metrics"]["clean_BA"] - r["baseline_metrics"]["clean_BA"]) for r in complete]
    robust_deltas = [100 * (r["proposed_metrics"]["mean_perturbed_BA"] - r["baseline_metrics"]["mean_perturbed_BA"])
                     for r in complete if "mean_perturbed_BA" in r["proposed_metrics"] and "mean_perturbed_BA" in r["baseline_metrics"]]
    summary = {"rows": rows, "datasets": len(rows), "complete": len(complete),
               "mean_clean_delta_pp": float(np.mean(deltas)) if deltas else None,
               "clean_wins": int(sum(v > 1e-10 for v in deltas)),
               "clean_losses": int(sum(v < -1e-10 for v in deltas)),
               "mean_perturbed_delta_pp": float(np.mean(robust_deltas)) if robust_deltas else None,
               "freeze_sha256": digest(FREEZE), "goal_achieved": False,
               "scope": "development_only_fixed_weighted_FGS120_anchor_no_confirmation"}
    (OUT / "complete.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

