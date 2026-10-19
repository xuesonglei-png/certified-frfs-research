"""Versioned auditor repair for scale-before-distance witness semantics.

The original certificate selects fixed witnesses using cityblock(X*w, X*w),
whereas audit v1 selected them with sum(abs(X-X)*w). Those formulas are
mathematically equal but do not always produce equal binary floating-point
ties. This version independently sums scale-before-subtract distances.
The certificate parameters, exact bound formulas and recorded data stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts" / "audit_relation_support_selector_v1.py"


def main(name):
    source = ORIGINAL.read_text(encoding="utf-8")
    patches = {
        "float(np.sum(abs(A[j]-A[i])*weights))": "sum(abs(float(A[j,a]*weights[a])-float(A[i,a]*weights[a])) for a in range(A.shape[1]))",
        "float(np.sum(abs(A[i]-A[j])*full))": "sum(abs(float(A[i,a]*full[a])-float(A[j,a]*full[a])) for a in range(A.shape[1]))",
        "(name+\"-audit.json\")": "(name+\"-tie-replay-audit-v2.json\")",
    }
    for old, new in patches.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = {"__name__": "versioned_independent_tie_replay", "__file__": str(ORIGINAL)}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    report = namespace["audit"](name)
    report["base_auditor_sha256"] = report.pop("auditor_sha256")
    report["versioned_auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["executed_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    report["witness_distance_semantics"] = "independent_scalar_scale_before_subtract_matching_declared_cityblock_preweighted_coordinates"
    report["original_audit_failures_retained"] = True
    path = namespace["FOLDER"] / (name+"-tie-replay-audit-v2.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True)
    main(parser.parse_args().only)
