"""Summarize preserved training-only diagnoses without predictive claims."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"results"/"expanded-majority-admission-v1"
NAMES = ("banknote", "glass", "heart_statlog", "ionosphere", "seeds", "wdbc")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows, totals = [], Counter()
    for name in NAMES:
        path = FOLDER/(name+".json")
        diagnosis = json.loads(path.read_text(encoding="utf-8"))
        audit_path = FOLDER/(name+"-audit-v1.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        current = (audit.get("passed", False) and audit.get("diagnostic_sha256") == digest(path)
                   and audit.get("source_sha256") == diagnosis["source_sha256"]
                   and audit.get("auditor_sha256") == digest(ROOT/"scripts"/"audit_expanded_majority_admission_v1.py"))
        row = {"dataset": name, "majority_only_rejected_candidates": diagnosis["evaluated"],
               "expanded_newly_admitted": diagnosis["newly_admitted"], "additional_object_certificates": diagnosis["additional_certificates"],
               "independent_audit_passed_current_hashes": current, "source_sha256": diagnosis["source_sha256"]}
        if current:
            row.update(audit["counts"])
            totals.update(audit["counts"])
        rows.append(row)
    report = {"goal_achieved": False, "new_outer_scoring": False, "independent_optimizer_claim": False,
              "original_protected_sets_unchanged": True, "parameters": {"eps": .01, "eta": .1, "retention": .9, "friend_pool": [3, 15]},
              "scope": "training_only_diagnostic_of_conservative_majority_admission",
              "all_six_current_audits_passed": all(r["independent_audit_passed_current_hashes"] for r in rows),
              "diagnostic_total_majority_only_rejections": sum(r["majority_only_rejected_candidates"] for r in rows),
              "diagnostic_total_new_admissions": sum(r["expanded_newly_admitted"] for r in rows),
              "audited_totals": dict(totals), "datasets": rows,
              "interpretation": "Witness expansion repairs some majority-only rejections; it cannot admit supports rejected by unchanged dependency/classwise guards and does not show higher test accuracy.",
              "limitations": ["counts_are_unique_candidates_within_training_split_and_family_not_independent_trials", "object_certificate_counts_sum_repeated_objects_across_candidate_models", "no_test_prediction_or_accuracy_measured", "dependency_certificates_preserved_not_independently_rederived_by_this_diagnostic", "original_protected_sets_replayed_using_declared_preweighted_witness_order"]}
    (FOLDER/"summary-v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = sorted(set(k for row in rows for k in row))
    with (FOLDER/"summary-v1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2))


if __name__ == "__main__":
    main()
