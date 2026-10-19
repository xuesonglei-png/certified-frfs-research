"""Dataset-equal descriptive v3 summary including failures and audit provenance."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "relation-support-selector-v3"


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(records, d):
    rows = [r for r in records if r.get("status", "complete") == "complete"]
    result = {"available_outer_folds": len(rows), "unavailable_outer_folds": len(records)-len(rows)}
    result["unavailable_reasons"] = dict(Counter(r["status"] for r in records if r not in rows))
    metric_rows = []
    for r in rows:
        selected = r["fit"]["selected"]
        metric_rows.append({
            "proposed_clean_BA": r["proposed"]["clean_BA"],
            "full_clean_BA": r["full"]["clean_BA"],
            "matched_clean_BA": r["matched_uniform"]["clean_BA"],
            "proposed_perturbed_BA": r["perturbed_BA"],
            "full_perturbed_BA": r["full_perturbed_BA"],
            "matched_perturbed_BA": r["matched_perturbed_BA"],
            "clean_delta_full_pp": 100*(r["proposed"]["clean_BA"]-r["full"]["clean_BA"]),
            "clean_delta_matched_pp": 100*(r["proposed"]["clean_BA"]-r["matched_uniform"]["clean_BA"]),
            "perturbed_delta_full_pp": 100*(r["perturbed_BA"]-r["full_perturbed_BA"]),
            "perturbed_delta_matched_pp": 100*(r["perturbed_BA"]-r["matched_perturbed_BA"]),
            "macro_F1_delta_full_pp": 100*(r["proposed"]["macro_F1"]-r["full"]["macro_F1"]),
            "macro_F1_delta_matched_pp": 100*(r["proposed"]["macro_F1"]-r["matched_uniform"]["macro_F1"]),
            "logloss_delta_full": r["proposed"]["balanced_logloss"]-r["full"]["balanced_logloss"],
            "removed_fraction": 1-len(selected["features"])/d,
            "selected_features": len(selected["features"]),
            "fit_evaluation_seconds": r["seconds"],
            "selected_dependency_upper_bound": selected["certificate"]["dependency_certificate"]["upper_bound"],
            "selected_max_classwise_dependency_bound": max(selected["classwise"]["bounds"]),
        })
    for key in metric_rows[0] if metric_rows else []:
        result["mean_"+key] = mean([r[key] for r in metric_rows])
    result["fallback_folds"] = sum(r["fit"]["selected"]["outer_refit_fallback"] for r in rows)
    result["full_support_folds"] = sum(len(r["fit"]["selected"]["features"]) == d for r in rows)
    result["compressed_folds"] = len(rows)-result["full_support_folds"]
    result["selected_joint_certified_folds"] = sum(r["fit"]["selected"]["joint_feasible"] for r in rows)
    result["outer_refit_attempt_joint_feasible_folds"] = sum(r["fit"]["refit_attempt"]["joint_feasible"] for r in rows)
    result["selected_family_counts"] = dict(Counter(r["fit"]["selected"]["family"] for r in rows))
    result["selected_required_majority_counts_sum"] = sum(sum(r["fit"]["selected"]["certificate"]["required"]) for r in rows)
    result["selected_retained_protected_counts_sum"] = sum(sum(r["fit"]["selected"]["certificate"]["retained_counts"]) for r in rows)
    candidates = [c["candidate"] for r in rows for family in r["fit"]["inner"] for fold in family["folds"] for c in fold["candidates"]]
    result["inner_candidates"] = len(candidates)
    result["inner_joint_feasible_candidates"] = sum(c["joint_feasible"] for c in candidates)
    for key in ("clean_delta_full_pp", "clean_delta_matched_pp", "perturbed_delta_full_pp", "perturbed_delta_matched_pp"):
        values = [r[key] for r in metric_rows]
        result[key+"_fold_wins_losses_ties"] = {"wins": sum(v > 1e-12 for v in values), "losses": sum(v < -1e-12 for v in values), "ties": sum(abs(v) <= 1e-12 for v in values)}
    return result


def main():
    status_path = FOLDER/"audit-status-v2.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    statuses = {r["dataset"]: r for r in status.get("datasets", [])}
    datasets = []
    raw = []
    for path in sorted(FOLDER.glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if "dataset" not in saved or "rows" not in saved:
            continue
        summary = summarize(saved["rows"], saved["d"])
        summary.update(dataset=saved["dataset"], dimensions=saved["d"], artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        audit = statuses.get(saved["dataset"], {})
        summary["audit_status"] = audit.get("status", "pending")
        summary["effective_audit_status"] = summary["audit_status"]
        summary["effective_audit_counts"] = audit.get("counts", {}) if summary["audit_status"].startswith("passed") else {}
        tie_path = FOLDER/(saved["dataset"]+"-tie-replay-audit-v2.json")
        if tie_path.exists():
            tie = json.loads(tie_path.read_text(encoding="utf-8"))
            tie_auditor = ROOT/"scripts"/"audit_relation_support_selector_v3_tie_replay_v2.py"
            current = (tie.get("passed") and tie.get("artifact_sha256") == summary["artifact_sha256"]
                       and tie.get("base_auditor_sha256") == status.get("auditor_sha256")
                       and tie.get("versioned_auditor_sha256") == hashlib.sha256(tie_auditor.read_bytes()).hexdigest())
            summary["tie_replay_audit"] = {"current_hashes_valid": current, "path": str(tie_path.relative_to(ROOT)), "counts": tie.get("counts", {})}
            if current:
                summary["effective_audit_status"] = "passed_separate_tie_replay_original_failure_preserved"
                summary["effective_audit_counts"] = tie.get("counts", {})
        if "failure_tail" in audit:
            summary["audit_failure_last_line"] = audit["failure_tail"].strip().splitlines()[-1]
        summary["artifact_fingerprints"] = saved["fingerprints"]
        datasets.append(summary)
        raw.append(saved)
    available = [d for d in datasets if d["available_outer_folds"]]
    passed = [d for d in available if d["effective_audit_status"].startswith("passed")]
    keys = [k for k in available[0] if k.startswith("mean_")] if available else []
    def aggregate(selected):
        return {key: mean([d[key] for d in selected]) for key in keys}
    total_keys = ("available_outer_folds", "unavailable_outer_folds", "fallback_folds", "full_support_folds", "compressed_folds", "selected_joint_certified_folds", "outer_refit_attempt_joint_feasible_folds", "inner_candidates", "inner_joint_feasible_candidates")
    totals = {key: sum(d[key] for d in datasets) for key in total_keys}
    fold_equal = {}
    for key in keys:
        fold_equal[key] = float(sum(d[key]*d["available_outer_folds"] for d in available)/totals["available_outer_folds"])
    for key in ("clean_delta_full_pp", "clean_delta_matched_pp", "perturbed_delta_full_pp", "perturbed_delta_matched_pp"):
        values = [d["mean_"+key] for d in available]
        totals[key+"_dataset_wins_losses_ties"] = {"wins": sum(v>1e-12 for v in values), "losses": sum(v < -1e-12 for v in values), "ties": sum(abs(v) <= 1e-12 for v in values)}
        totals[key+"_fold_wins_losses_ties"] = {outcome: sum(d[key+"_fold_wins_losses_ties"][outcome] for d in datasets) for outcome in ("wins", "losses", "ties")}
    report = {"goal_achieved": False, "scope": "descriptive_development_no_confirmatory_superiority_claim",
              "dataset_count": len(datasets), "datasets_with_available_folds": len(available),
              "audit_batch_complete": status.get("completed", False), "audit_status_counts": dict(Counter(d["audit_status"] for d in datasets)),
              "effective_audit_status_counts": dict(Counter(d["effective_audit_status"] for d in datasets)),
              "effective_audit_count_totals": dict(sum((Counter(d["effective_audit_counts"]) for d in datasets), Counter())),
              "dataset_equal_all_available": aggregate(available), "dataset_equal_currently_passed_audits_only": aggregate(passed),
              "fold_equal_all_available": fold_equal, "totals": totals, "datasets": datasets,
              "certification_coverage": {
                  "selected_joint_feasible_fraction_recorded": totals["selected_joint_certified_folds"]/totals["available_outer_folds"],
                  "current_effective_audited_available_fold_fraction": sum(d["available_outer_folds"] for d in passed)/totals["available_outer_folds"],
                  "compressed_deployed_fold_fraction": totals["compressed_folds"]/totals["available_outer_folds"],
                  "fallback_fraction": totals["fallback_folds"]/totals["available_outer_folds"],
                  "inner_feasible_candidate_fraction": totals["inner_joint_feasible_candidates"]/totals["inner_candidates"],
                  "query_certified_BA": None,
                  "query_certified_coverage": None,
                  "query_metric_status": "not_recorded_in_v3_artifacts_not_inferred_from_training_certificates"},
              "limitations": ["development_data_previously_exposed", "audit_failures_preserved_not_excluded_from_overall_means", "training_certificate_feasibility_is_not_prediction_risk_bound", "query_only_random_perturbations_are_not_shared_table_adversarial_attacks", "no_training_source_freeze_existed_before_v3_run", "selected_required_counts_do_not_identify_exact_original_protected_counts", "fit_evaluation_seconds_include_training_and_evaluation_not_inference_latency"]}
    out = FOLDER/"audited-summary-v2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = ["dataset", "dimensions", "available_outer_folds", "unavailable_outer_folds", "mean_clean_delta_full_pp", "mean_clean_delta_matched_pp", "mean_perturbed_delta_full_pp", "mean_perturbed_delta_matched_pp", "mean_removed_fraction", "fallback_folds", "full_support_folds", "compressed_folds", "inner_candidates", "inner_joint_feasible_candidates", "audit_status", "effective_audit_status"]
    with (FOLDER/"audited-dataset-summary-v2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: d.get(key) for key in columns} for d in datasets)
    brief = {k: v for k, v in report.items() if k not in ("datasets", "fold_equal_all_available", "dataset_equal_currently_passed_audits_only")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
