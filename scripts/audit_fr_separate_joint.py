"""Compare a separable fuzzy-rough certificate with the joint certificate.

This audit uses only frozen round-three interval tables and round-four joint
certificate records. It runs no MILP or new classifier experiment. The
separable value is a sound FR lower bound obtained from
inf lower + inf punctured-upper - 1, with the upper infimum bounded by the
endpoint maximum of one same-class far function.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/round3/margin-mixture-benchmark.json"
JOINT = ROOT / "results/round4/projected-mixture-audit.json"
OUT = ROOT / "results/round4/fr-separate-joint-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(table, group):
    x = np.asarray(table["X"], dtype=float)
    if group == "eps_0.01":
        radius = np.full_like(x, 0.01)
    else:
        radius = np.asarray(table["heterogeneous_eps"], dtype=float)
    return np.maximum(0.0, x - radius), np.minimum(1.0, x + radius)


def interval_dist(qlo, qhi, elo, ehi):
    return np.maximum(elo - qhi, qlo - ehi).clip(min=0.0)


def far(a, lo, hi):
    return np.maximum(np.abs(a - lo), np.abs(a - hi))


def separate_value(lo, hi, target, friends, enemies, weights):
    qlo, qhi = lo[target], hi[target]
    p_minus = min(float(np.dot(weights, interval_dist(qlo, qhi, lo[j], hi[j])))
                  for j in enemies)
    vertices = np.asarray(list(itertools.product(*zip(qlo, qhi))), dtype=float)
    friend_max = []
    for h in friends:
        vals = far(vertices, lo[h], hi[h]) @ weights
        friend_max.append(float(vals.max()))
    # max_a min_h F_h(a) <= min_h max_a F_h(a); therefore this is a
    # certified lower bound on inf_Z(lower+upper-1).
    upper_lower = 1.0 - min(friend_max)
    return p_minus + upper_lower - 1.0, p_minus, upper_lower, friend_max


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    joint = json.loads(JOINT.read_text(encoding="utf-8"))
    table_map = {t["table_id"]: t for t in source["tables"]}
    rows = []
    for row in joint["rows"]:
        table = table_map[row["table_id"]]
        lo, hi = bounds(table, row["group"])
        weights = np.asarray(table["weights"], dtype=float)
        sep, p_minus, u_lower, friend_max = separate_value(
            lo, hi, row["target"], row["pairs"][0]["friends"],
            [p["enemy"] for p in row["pairs"]], weights)
        rows.append({
            "table_id": row["table_id"], "group": row["group"],
            "d": row["d"], "target": row["target"],
            "separate_lower": sep, "joint_lower": row["safe_lower"],
            "joint_upper": row["attack_upper"],
            "saved_exact_interval": row["saved_MILP_robust_interval"],
            "separate_p_lower": p_minus,
            "separate_upper_lower": u_lower,
            "friend_endpoint_maxima": friend_max,
            "separate_certified": bool(sep > 1e-8),
            "joint_certified": bool(row["safe_lower"] > 1e-8),
        })
    sep = np.asarray([r["separate_lower"] for r in rows])
    joint_lo = np.asarray([r["joint_lower"] for r in rows])
    exact = np.asarray([r["saved_exact_interval"][0] for r in rows])
    summary = {
        "objects": len(rows),
        "separate_positive": int(sum(r["separate_certified"] for r in rows)),
        "joint_positive": int(sum(r["joint_certified"] for r in rows)),
        "separate_mean_lower": float(sep.mean()),
        "joint_mean_lower": float(joint_lo.mean()),
        "mean_joint_minus_separate": float((joint_lo - sep).mean()),
        "median_joint_minus_separate": float(np.median(joint_lo - sep)),
        "joint_strictly_tighter": int(np.sum(joint_lo > sep + 1e-8)),
        "separate_strictly_tighter": int(np.sum(sep > joint_lo + 1e-8)),
        "separate_gap_to_exact_mean": float(np.mean(exact - sep)),
        "joint_gap_to_exact_mean": float(np.mean(exact - joint_lo)),
        "separate_ambiguity": int(np.sum((sep <= 0) & (exact > 0))),
        "joint_ambiguity": int(np.sum((joint_lo <= 0) & (exact > 0))),
    }
    result = {
        "protocol": "fuzzy_rough_separate_vs_joint_v1",
        "status": "complete", "new_MILP_calls": 0,
        "source_sha256": sha(SOURCE), "joint_audit_sha256": sha(JOINT),
        "definition": "separate_lower = inf lower + valid lower bound on inf punctured upper - 1",
        "summary": summary, "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ROOT / "research/round4/fr-separate-joint-results.md"
    md.write_text(
        "# Separate versus joint fuzzy-rough certification\n\n"
        "Frozen round-three interval tables and round-four joint records were reused; no MILP or new classifier run was performed. The separable certificate uses the exact worst lower approximation and the valid bound "
        "`max_a min_h F_h(a) <= min_h max_a F_h(a)`, evaluated at query-box vertices. Thus it is a sound but deliberately separable FR lower bound.\n\n"
        f"- objects: {summary['objects']}\n"
        f"- separate positive certificates: {summary['separate_positive']}\n"
        f"- joint positive certificates: {summary['joint_positive']}\n"
        f"- mean joint-minus-separate lower bound: {summary['mean_joint_minus_separate']:.6f}\n"
        f"- objects where joint is strictly tighter: {summary['joint_strictly_tighter']}\n"
        f"- separate ambiguity count among exact-positive objects: {summary['separate_ambiguity']}\n"
        f"- joint ambiguity count among exact-positive objects: {summary['joint_ambiguity']}\n\n"
        "The comparison tests the FR coupling theorem: separate worst-case lower and upper values lose the common realization, whereas the joint certificate retains it. It is a certificate-tightness comparison, not a claim of test-accuracy superiority.\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
