"""Read-only combinatorial analysis of already classified subset audit rows."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/round2/normalized-subset-audit-triage.json"
OUTPUT = ROOT / "results/round2/normalized-subset-paths.json"
NOTE = ROOT / "research/round2/normalized-subset-paths.md"


def paths(family, full):
    family = set(family) | {full}
    reachable = {full}
    frontier = [full]
    parent = {}
    while frontier:
        current = frontier.pop()
        for feature in current:
            child = current - {feature}
            if child in family and child not in reachable:
                reachable.add(child)
                frontier.append(child)
                parent[child] = current
    unreachable = family - reachable
    shortest = min(reachable, key=lambda x: (len(x), sorted(x)))
    path = [sorted(shortest)]
    current = shortest
    while current != full:
        current = parent[current]
        path.append(sorted(current))
    return {
        "proper_subsets": len(family) - 1,
        "minimum_cardinality_including_full": min(map(len, family)),
        "minimum_reachable_cardinality": len(shortest),
        "unreachable_proper_subsets": len(unreachable),
        "unreachable_feature_sets": sorted([sorted(x) for x in unreachable], key=lambda x: (len(x), x)),
        "one_minimum_reachable_path": list(reversed(path)),
    }


def main():
    raw = SOURCE.read_bytes()
    audit = json.loads(raw)
    assert audit["summary"]["fully_classified"] and audit["valid_frozen_run"]
    out = {
        "protocol": "normalized-subset-feasible-paths-2026-09-20-v1",
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "retention": audit["retention"], "eps": audit["eps"],
        "notes": [
            "No new optimization, source changes, or data resampling. Use the completed first triage audit's 730 classified proper subsets.",
            "Full set is inserted as an algebraically feasible root and fallback; empty subset remains forbidden.",
            "An edge deletes exactly one attribute and both endpoint sets must belong to the same feasibility/certification family.",
            "Reachability considers all deletion paths, not one MI order. Unreachable proven-feasible sets indicate a real feasibility-path obstruction; certified-family obstructions may also reflect conservative bounds.",
            "Feasibility is numerical at the original solver tolerances and only concerns these fixed small tables.",
        ], "datasets": {},
    }
    methods = ("independent", "hybrid_interval", "hybrid_exact")
    for name, data in audit["inputs"].items():
        rows = [row for row in audit["rows"] if row["dataset"] == name]
        full = frozenset(range(data["d"]))
        feasible = [frozenset(row["features"]) for row in rows if row["status"] == "feasible"]
        stats = {"d": data["d"], "true_feasible": paths(feasible, full)}
        for method in methods:
            family = [frozenset(row["features"]) for row in rows if row["certificates"][method]["certified"]]
            stats[method] = paths(family, full)
        out["datasets"][name] = stats
    out["summary"] = {key: {
        "proper_subsets": sum(row[key]["proper_subsets"] for row in out["datasets"].values()),
        "unreachable_proper_subsets": sum(row[key]["unreachable_proper_subsets"] for row in out["datasets"].values()),
        "datasets_with_unreachable_subsets": sum(row[key]["unreachable_proper_subsets"] > 0 for row in out["datasets"].values()),
        "datasets_with_minimum_path_gap": sum(row[key]["minimum_reachable_cardinality"] > row[key]["minimum_cardinality_including_full"]
                                               for row in out["datasets"].values()),
    } for key in ("true_feasible", *methods)}
    out["summary"]["independent_minimum_above_feasible"] = sum(
        row["independent"]["minimum_cardinality_including_full"] > row["true_feasible"]["minimum_cardinality_including_full"]
        for row in out["datasets"].values())
    out["summary"]["exact_stability_minimum_below_independent"] = sum(
        row["hybrid_exact"]["minimum_cardinality_including_full"] < row["independent"]["minimum_cardinality_including_full"]
        for row in out["datasets"].values())
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 小子表最小基数与可行删除路径", "", "日期：2026-09-20。只读分析已完成的730个子集记录，没有新MILP、采样或源码修改。", "",
             "每组加入全属性集作为已知可行根；只允许一次删除一个属性。以下路径检查穷尽所有可行删除顺序，不等于现有MI单次顺序的实际输出。", "",
             "| 数据集 | 全维d | 真可行最小 | 独立最小 | 区间星最小 | 精确稳定星最小 | 真可行但不可达数 | 最小可达 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in out["datasets"].items():
        true = row["true_feasible"]
        mins = [row[k]["minimum_cardinality_including_full"] for k in ("true_feasible", *methods)]
        lines.append(f"| {name} | {row['d']} | " + " | ".join(map(str, mins)) +
                     f" | {true['unreachable_proper_subsets']} | {true['minimum_reachable_cardinality']} |")
    s = out["summary"]
    lines += ["", f"98个已证明可行真子集中，无法通过全程可行单属性删除从完整集到达的有 {s['true_feasible']['unreachable_proper_subsets']} 个，涉及 {s['true_feasible']['datasets_with_unreachable_subsets']} 个数据集。最小可行基数与最小可达基数不同的数据集数为 {s['true_feasible']['datasets_with_minimum_path_gap']}。", "",
              f"独立证书最小基数高于真可行最小基数的数据集数：{s['independent_minimum_above_feasible']}。精确稳定星界将最小认证基数降低的数据集数：{s['exact_stability_minimum_below_independent']}。", "",
              "| 证书族 | 认证真子集数 | 族内不可达数 | 存在不可达的数据集 | 最小基数受路径限制的数据集 |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        r = s[method]
        lines.append(f"| {method} | {r['proper_subsets']} | {r['unreachable_proper_subsets']} | {r['datasets_with_unreachable_subsets']} | {r['datasets_with_minimum_path_gap']} |")
    lines += ["", "归一化模型一般没有属性单调性；这组固定小表是否出现路径障碍由上表决定，不能从一般反例直接推断此处存在障碍。证书族不可达可能来自保守性，不应与真可行族不可达混同。所有不可达集合及一条最小可达路径保存在 `results/round2/normalized-subset-paths.json`。", "",
              "源记录为首轮分层审计快照 `results/round2/normalized-subset-audit-triage.json`，对应脚本原始字节快照为 `scripts/audit_normalized_subsets_triage_snapshot.py`，其SHA256与首轮记录完全一致。后续若存在all-MILP审计，不改变本文件的来源或判定协议。", ""]
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
