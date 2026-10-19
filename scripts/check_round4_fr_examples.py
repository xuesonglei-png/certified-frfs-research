"""Exact rational audit of the two round-four dependency examples.

Only Python's standard library is used. Box statements follow from exact
pair-distance interval enclosures and, where exact robust minima are claimed,
matching feasible witnesses. No random sampling, LP, MILP, or experiment
rerun is used. Output is separate from every earlier experiment snapshot.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (Q(1, 2), Q(1, 2))
FULL = (0, 1)
REDUCED = (0,)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rational_json(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, dict):
        return {key: rational_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [rational_json(item) for item in value]
    return value


def point_distance(x, z, features):
    return sum((WEIGHTS[k] * abs(x[k] - z[k]) for k in features), Q(0))


def point_values(points, labels, features):
    rows = []
    for i, point in enumerate(points):
        enemies = [j for j in range(len(points)) if labels[j] != labels[i]]
        friends = [h for h in range(len(points)) if h != i and labels[h] == labels[i]]
        require(bool(enemies and friends), "Each audited object needs both reference sets")
        p = min(point_distance(point, points[j], features) for j in enemies)
        s = min(point_distance(point, points[h], features) for h in friends)
        rows.append({"enemy_min": p, "friend_min": s, "margin": p - s})
    return {"objects": rows, "dependency": sum((row["enemy_min"] for row in rows), Q(0)) / len(rows)}


def pair_interval(box_x, box_y, features):
    lower = upper = Q(0)
    for k in features:
        lx, ux = box_x[k]
        ly, uy = box_y[k]
        lower += WEIGHTS[k] * max(Q(0), lx - uy, ly - ux)
        upper += WEIGHTS[k] * max(abs(lx - uy), abs(ux - ly))
    return lower, upper


def margin_enclosures(boxes, labels, features):
    """Safe for every shared realization; separate extrema need not coattain."""
    rows = []
    for i, box in enumerate(boxes):
        enemies = [pair_interval(box, boxes[j], features)
                   for j in range(len(boxes)) if labels[j] != labels[i]]
        friends = [pair_interval(box, boxes[h], features)
                   for h in range(len(boxes)) if h != i and labels[h] == labels[i]]
        p_lower, p_upper = min(row[0] for row in enemies), min(row[1] for row in enemies)
        s_lower, s_upper = min(row[0] for row in friends), min(row[1] for row in friends)
        rows.append({"enemy_min_interval": (p_lower, p_upper),
                     "friend_min_interval": (s_lower, s_upper),
                     "margin_interval": (p_lower - s_upper, p_upper - s_lower)})
    return rows


def clipped_boxes(points, radius):
    return [[(max(Q(0), x - radius), min(Q(1), x + radius)) for x in row] for row in points]


def in_box(points, boxes):
    return all(lo <= points[i][k] <= hi
               for i, row in enumerate(boxes) for k, (lo, hi) in enumerate(row))


def converse_example():
    points = [(Q(0), Q(0)), (Q(0), Q(1)),
              (Q(1), Q(1, 10)), (Q(1), Q(9, 10))]
    labels = ["A", "A", "B", "B"]
    full, reduced = point_values(points, labels, FULL), point_values(points, labels, REDUCED)
    require(full["dependency"] == Q(11, 20), "Converse full dependency")
    require(reduced["dependency"] == Q(1, 2), "Converse reduced dependency")
    require([row["margin"] for row in full["objects"]] == [Q(1, 20)] * 2 + [Q(3, 20)] * 2,
            "Converse nominal full margins")
    require(all(row["margin"] == Q(1, 2) for row in reduced["objects"]),
            "Converse nominal reduced margins")

    radius = Q(1, 100)
    boxes = clipped_boxes(points, radius)
    require(all(lo < hi for row in boxes for lo, hi in row), "Every coordinate interval has positive width")
    full_boxes, reduced_boxes = margin_enclosures(boxes, labels, FULL), margin_enclosures(boxes, labels, REDUCED)
    full_lipschitz = [(row["margin"] - 4 * radius, row["margin"] + 4 * radius)
                      for row in full["objects"]]
    reduced_lipschitz = [(row["margin"] - 2 * radius, row["margin"] + 2 * radius)
                         for row in reduced["objects"]]
    for i in range(len(points)):
        full_lower, full_upper = full_boxes[i]["margin_interval"]
        reduced_lower, _ = reduced_boxes[i]["margin_interval"]
        require(full_lower > 0, "Full-set strict robustness throughout the box")
        require(reduced_lower > full_upper, "Reduced margin strictly exceeds full margin throughout the box")
        require(full_lipschitz[i][0] >= Q(1, 100) and full_lipschitz[i][1] <= Q(19, 100),
                "Published full-set Lipschitz enclosure")
        require(reduced_lipschitz[i][0] >= Q(48, 100), "Published reduced-set Lipschitz lower bound")
    return {"name": "dependency_preservation_is_not_necessary", "points": points, "labels": labels,
            "nominal_full": full, "nominal_reduced": reduced, "radius": radius, "boxes": boxes,
            "full_box_enclosures": full_boxes, "reduced_box_enclosures": reduced_boxes,
            "full_lipschitz_enclosures": full_lipschitz,
            "reduced_lipschitz_enclosures": reduced_lipschitz,
            "certified_statements": ["All full-set margins are strictly positive for every shared realization.",
                                     "Every reduced margin exceeds its corresponding full margin for every realization.",
                                     "The nominal dependency decreases from 11/20 to 1/2."]}


def nominal_equality_example():
    points = [(Q(0), Q(0)), (Q(0), Q(1, 10)), (Q(0), Q(1)),
              (Q(1), Q(0)), (Q(1), Q(1, 10)), (Q(1), Q(1))]
    labels = ["A"] * 3 + ["B"] * 3
    boxes = [[(x, x) for x in row] for row in points]
    boxes[1][0] = boxes[2][0] = (Q(0), Q(1, 10))
    boxes[5][0] = (Q(0), Q(1))
    full, reduced = point_values(points, labels, FULL), point_values(points, labels, REDUCED)
    require(full["dependency"] == reduced["dependency"] == Q(1, 2), "Nominal dependency equality")
    require(all(row["enemy_min"] == Q(1, 2) for row in full["objects"] + reduced["objects"]),
            "Every nominal own-class lower approximation equals 1/2")
    require(all(row["margin"] > 0 for row in full["objects"]), "All nominal full margins are positive")
    full_boxes, reduced_boxes = margin_enclosures(boxes, labels, FULL), margin_enclosures(boxes, labels, REDUCED)

    witness = list(points)
    witness[1] = (Q(1, 10), Q(1, 10))
    witness[2] = (Q(1, 10), Q(1))
    witness[5] = (Q(0), Q(1))
    require(in_box(witness, boxes), "Attack and attainment witness lies in the box")
    witness_full = point_values(witness, labels, FULL)
    witness_reduced = point_values(witness, labels, REDUCED)
    full_lower = full_boxes[0]["margin_interval"][0]
    reduced_lower = reduced_boxes[0]["margin_interval"][0]
    require(full_lower == witness_full["objects"][0]["margin"] == Q(2, 5),
            "Exact query full robust margin: certified lower bound is attained")
    require(reduced_lower == witness_reduced["objects"][0]["margin"] == Q(-1, 20),
            "Exact query reduced robust margin: certified lower bound is attained")
    return {"name": "nominal_dependency_equality_does_not_imply_robust_preservation",
            "points": points, "labels": labels, "boxes": boxes,
            "nominal_full": full, "nominal_reduced": reduced,
            "query_index": 0, "query_full_box_enclosure": full_boxes[0],
            "query_reduced_box_enclosure": reduced_boxes[0],
            "attainment_witness": witness, "witness_full": witness_full, "witness_reduced": witness_reduced,
            "exact_query_robust_margins": {"full": full_lower, "reduced": reduced_lower},
            "scope": "Robust claims concern query a0 only; no assertion that all other objects are robust.",
            "certified_statements": ["Every nominal lower approximation and hence dependency is preserved exactly.",
                                     "Query a0 has full robust margin 2/5 and reduced robust margin -1/20.",
                                     "The reduced failure is strict, without an enemy/friend distance tie."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/round4/fr-examples.json")
    args = parser.parse_args()
    result = {"protocol": "round4-fr-dependency-examples-exact-rational-v1", "status": "passed",
              "arithmetic": "Python fractions.Fraction; no floating point or optimization solver",
              "box_method": "Exact pair-distance interval enclosures; matching feasible witnesses establish claimed minima",
              "weights": WEIGHTS, "full_features": FULL, "reduced_features": REDUCED,
              "examples": [converse_example(), nominal_equality_example()],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Checks the explicit examples and analytic box bounds, not general theorem validity or novelty."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rational_json(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "examples": len(result["examples"]),
                      "arithmetic": result["arithmetic"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
