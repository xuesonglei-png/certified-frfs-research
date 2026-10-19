"""Sequential audit of preserved v3 artifacts; never changes production code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "relation-support-selector-v3"
AUDITOR = ROOT / "scripts" / "audit_relation_support_selector_v1.py"
STATUS = FOLDER / "audit-status-v2.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
    rows = []
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "completed": False,
              "auditor_sha256": digest(AUDITOR), "datasets": rows,
              "scope": "development_artifact_audit_not_confirmatory_evidence"}
    def save():
        STATUS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    for path in sorted(FOLDER.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "dataset" not in data or "rows" not in data:
            continue
        name = data["dataset"]
        audit_path = FOLDER / (name + "-audit.json")
        prior = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        row = {"dataset": name, "artifact_sha256": digest(path)}
        rows.append(row)
        if prior.get("passed") and prior.get("artifact_sha256") == row["artifact_sha256"] and prior.get("auditor_sha256") == report["auditor_sha256"]:
            row.update(status="passed_existing", counts=prior["counts"])
            print(name, "existing current-hash audit passed", flush=True)
            save()
            continue
        row["status"] = "running"
        save()
        log_path = FOLDER / (name + "-audit-v2.log")
        print(name, "audit starting", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(AUDITOR), "--only", name],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row["exit_code"] = result.returncode
        row["log"] = str(log_path.relative_to(ROOT))
        if result.returncode == 0 and audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            row.update(status="passed" if audit.get("passed") else "failed", counts=audit.get("counts", {}))
        else:
            row.update(status="failed", failure_tail=log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
        print(name, row["status"], flush=True)
        save()
    report["completed"] = True
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed_datasets"] = sum(r["status"].startswith("passed") for r in rows)
    report["failed_datasets"] = sum(r["status"] == "failed" for r in rows)
    save()
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
