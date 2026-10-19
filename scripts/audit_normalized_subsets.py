"""Bounded exhaustive-subset audit on the previously fixed real small tables.

No source algorithms are changed. Numerical MILP is reserved for subsets whose
feasibility is unresolved after certificates and an explicit nominal witness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certified_frfs.normalized_relative import normalized_relative_certificate
from certified_frfs.normalized_verification import exact_normalized_margin_milp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    paths = sorted((ROOT / "src" / "certified_frfs").glob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def classify(groups):
    if all(group["upper_bound"] <= 0 for group in groups):
        return "feasible"
    if any(group["lower_bound"] > 1e-8 for group in groups):
        return "infeasible"
    return "unresolved"


def aggregate(rows):
    keys = ("independent", "hybrid_interval", "hybrid_exact")
    states = Counter(row["status"] for row in rows)
    done = [row for row in rows if "certificates" in row]
    feasible = [row for row in done if row["status"] == "feasible"]
    per_method = {}
    for key in keys:
        per_method[key] = {
            "certified_subsets": sum(row["certificates"][key]["certified"] for row in done),
            "missed_proven_feasible_subsets": sum(not row["certificates"][key]["certified"]
                                                  for row in feasible),
        }
    milps = [group["milp"] for row in done for group in row["groups"] if "milp" in group]
    return {
        "subsets_total": len(rows), "certificates_completed": len(done),
        "status_counts": dict(states), "methods": per_method,
        "hybrid_interval_extra_over_independent": sum(
            row["certificates"]["hybrid_interval"]["certified"] and
            not row["certificates"]["independent"]["certified"] for row in done),
        "hybrid_exact_extra_over_independent": sum(
            row["certificates"]["hybrid_exact"]["certified"] and
            not row["certificates"]["independent"]["certified"] for row in done),
        "exact_stability_extra_over_interval": sum(
            row["certificates"]["hybrid_exact"]["certified"] and
            not row["certificates"]["hybrid_interval"]["certified"] for row in done),
        "milp_class_calls": len(milps),
        "milp_optimal_calls": sum(m["optimal"] for m in milps),
        "milp_max_gap": max((m["gap"] for m in milps), default=None),
        "fully_classified": len(done) == len(rows) and not states.get("unresolved", 0),
        "min_feasible_cardinality": min((len(r["features"]) for r in feasible), default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=float, default=165)
    parser.add_argument("--milp-seconds", type=float, default=2)
    parser.add_argument("--all-milp", action="store_true", help="Solve every class of every subset within the same total budget")
    parser.add_argument("--input", default="results/real-subtable-verification.json")
    parser.add_argument("--output", default="results/round2/normalized-subset-audit.json")
    parser.add_argument("--note", default="research/round2/normalized-subset-audit.md")
    args = parser.parse_args()
    started = perf_counter()
    deadline = started + args.budget_seconds
    source_path, output_path, note_path = (ROOT / args.input, ROOT / args.output, ROOT / args.note)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    before = code_hashes()
    input_before = digest(source_path)
    rows, inputs = [], {}
    for original in source["rows"]:
        name = original["dataset"]
        inputs[name] = {key: original[key] for key in (
            "X", "y", "weights", "n", "d", "original_row_indices", "original_column_indices",
            "construction_seed", "first_fold_seed", "source_processed_sha256")}
        for bits in range(1, (1 << original["d"]) - 1):
            rows.append({"dataset": name, "features": [k for k in range(original["d"]) if bits & (1 << k)],
                         "status": "unresolved", "pending_reason": "certificate_not_evaluated"})
    result = {
        "protocol": "normalized-fixed-real-small-table-all-subsets-audit-2026-09-20-v1",
        "exploratory": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "retention": 0.9, "eps": 0.01, "scope": "classwise",
        "model": "subset_weight_normalized_l1_on_unit_box",
        "input_file": args.input, "input_sha256_before": input_before,
        "code_sha256_before": before, "arguments": vars(args), "inputs": inputs,
        "protocol_notes": [
            "Original 15 six-object at-most-six-attribute subtables are used without resampling or column changes.",
            "Enumerate every nonempty proper subset and compute three certificates against the same original full table.",
            "A nonpositive all-class certificate proves numerical feasibility. Any nominal class margin >1e-8 supplies an explicit infeasibility witness.",
            ("All classes of all subsets are sent to MILP in dataset-round-robin order." if args.all_milp else
             "Only subsets still unresolved after both steps are sent to MILP, every class in each such subset, in dataset-round-robin order."),
            "All groups retain valid initial [nominal witness, certificate upper] intervals even when MILP is skipped or the budget expires.",
            "Feasible requires all class upper bounds <=0; infeasible requires one attained lower bound >1e-8; otherwise unresolved.",
            "MILP optimum and all certificates use ordinary floating-point solver tolerances, not formal interval arithmetic.",
            "This audit establishes small-table subset feasibility, not full-dataset or predictive robustness, nor novelty of normalization.",
        ], "rows": rows,
    }
    for row in rows:
        if perf_counter() > deadline - 5:
            break
        data = inputs[row["dataset"]]
        X, y, weights = np.array(data["X"]), np.array(data["y"]), np.array(data["weights"])
        certificates = {}
        for name, method, stability in (("independent", "independent", "interval"),
                                         ("hybrid_interval", "hybrid", "interval"),
                                         ("hybrid_exact", "hybrid", "exact")):
            certificates[name] = normalized_relative_certificate(
                X, y, row["features"], weights, eps=0.01, retention=0.9,
                scope="classwise", method=method, stability=stability)
        row["certificates"] = certificates
        row["groups"] = []
        for index, group in enumerate(certificates["independent"]["groups"]):
            row["groups"].append({
                "group": group["group"], "n_objects": group["n_objects"],
                "lower_bound": group["nominal_margin"],
                "upper_bound": min(c["groups"][index]["upper_bound"] for c in certificates.values()),
                "witness": X.tolist(), "witness_source": "nominal",
                "milp_skipped_reason": "certificate_or_nominal_witness_decided_subset",
            })
        row["status"] = classify(row["groups"])
        row["initial_status"] = row["status"]
        row.pop("pending_reason", None)
        if args.all_milp or row["status"] == "unresolved":
            for group in row["groups"]:
                group["milp_skipped_reason"] = "awaiting_budgeted_milp"
    print(json.dumps({"phase": "certificates_finished", "elapsed": perf_counter()-started,
                      "summary": aggregate(rows)}), flush=True)
    pending = {name: [row for row in rows if row["dataset"] == name and "initial_status" in row
                     and (args.all_milp or row["initial_status"] == "unresolved")] for name in inputs}
    while any(pending.values()) and perf_counter() < deadline - 5:
        for name, queue in pending.items():
            if not queue or perf_counter() >= deadline - 5:
                continue
            row = queue.pop(0)
            data = inputs[name]
            X, y, weights = np.array(data["X"]), np.array(data["y"]), np.array(data["weights"])
            for group in row["groups"]:
                remaining = deadline - 5 - perf_counter()
                if remaining <= 0.05:
                    break
                object_weights = (y == group["group"]).astype(float)
                solved = exact_normalized_margin_milp(
                    X, y, row["features"], weights, eps=0.01, retention=0.9,
                    object_weights=object_weights, time_limit=min(args.milp_seconds, remaining))
                solved.pop("adversarial_X", None)
                group["milp"] = solved
                group.pop("milp_skipped_reason", None)
                group["upper_bound"] = min(group["upper_bound"], solved["upper_bound"])
                if solved["lower_bound"] >= group["lower_bound"]:
                    group["lower_bound"] = solved["lower_bound"]
                    group["witness"] = solved["worst_case"]
                    group["witness_source"] = "milp_feasible_table"
                if group["lower_bound"] > group["upper_bound"] + 2e-8:
                    raise RuntimeError("certificate upper bound conflicts with a feasible witness")
            row["status"] = classify(row["groups"])
    for row in rows:
        for group in row.get("groups", []):
            if group.get("milp_skipped_reason") == "awaiting_budgeted_milp":
                group["milp_skipped_reason"] = "runtime_budget_exhausted"
    result["summary"] = aggregate(rows)
    result["by_dataset"] = {name: aggregate([r for r in rows if r["dataset"] == name]) for name in inputs}
    result["elapsed_seconds"] = perf_counter() - started
    result["input_sha256_after"] = digest(source_path)
    result["code_sha256_after"] = code_hashes()
    result["source_unchanged"] = before == result["code_sha256_after"]
    result["input_unchanged"] = input_before == result["input_sha256_after"]
    result["valid_frozen_run"] = result["source_unchanged"] and result["input_unchanged"]
    result["completed_all_requested_subset_certificates"] = result["summary"]["certificates_completed"] == len(rows)
    result["pending_milp_classes"] = sum(g.get("milp_skipped_reason") == "runtime_budget_exhausted"
                                         for row in rows for g in row.get("groups", []))
    result["all_classes_numerically_solved"] = (args.all_milp and
        all("milp" in g for row in rows for g in row.get("groups", [])) and
        result["completed_all_requested_subset_certificates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=encode, allow_nan=False), encoding="utf-8")
    summary = result["summary"]
    lines = ["# 冻结小子表的归一化子集穷举审计", "", "日期：2026-09-20。第二轮探索，不替换第一轮实验。", "",
             f"固定 q=.9、ε=.01，直接复用既有15个6对象、最多6属性子表。共 {len(rows)} 个非空真子集，完成 {summary['certificates_completed']} 个三证书对照。",
             f"最终状态：{summary['status_counts']}。MILP按类调用 {summary['milp_class_calls']} 次，其中 {summary['milp_optimal_calls']} 次报告数值最优；总计算 {result['elapsed_seconds']:.2f} 秒。", "",
             ("协议：先对全部子集计算独立界、区间稳定星界、精确稳定星界，再对每个子集所有类求解MILP，按数据集轮转，单次上限2秒、总预算165秒。" if args.all_milp else
              "协议：先对全部子集计算独立界、区间稳定星界、精确稳定星界。已经认证全部类可行或已有名义违反见证的子集不再求解MILP；其每类仍保留名义可行表作为下界和解析上界。仅对剩余未决子集逐类求解MILP，按数据集轮转，单次上限2秒、总预算165秒。") + "正上界只表示未决；有正的可行下界才判不可行。", "",
             "| 方法 | 认证子集 | 漏过的已证明可行子集 |", "|---|---:|---:|"]
    for method, stats in summary["methods"].items():
        lines.append(f"| {method} | {stats['certified_subsets']} | {stats['missed_proven_feasible_subsets']} |")
    lines += ["", f"区间稳定星结构比独立界额外认证 {summary['hybrid_interval_extra_over_independent']} 个子集；精确稳定版比独立界额外认证 {summary['hybrid_exact_extra_over_independent']} 个，比区间版额外认证 {summary['exact_stability_extra_over_interval']} 个。", "",
              "| 数据集 | 子集数 | 可行 | 不可行 | 未决 | 独立认证 | 星形精确稳定认证 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in result["by_dataset"].items():
        status = stats["status_counts"]
        lines.append(f"| {name} | {stats['subsets_total']} | {status.get('feasible',0)} | {status.get('infeasible',0)} | {status.get('unresolved',0)} | {stats['methods']['independent']['certified_subsets']} | {stats['methods']['hybrid_exact']['certified_subsets']} |")
    lines += ["", f"输入与源码前后哈希一致：{result['valid_frozen_run']}。未修改任何 src 文件。所有子集、每类区间、见证和MILP状态保存在 `{args.output}`；复现脚本为 `scripts/audit_normalized_subsets.py`。", "",
              "这里的可行指同一全扰动、逐类相对目标；MILP为容差下的数值求解。未求解的子集不能宣称已知精确最坏违反量。穷举对象仅是既有小子表，不能外推完整数据集最优约简或分类表现。若有未决项，漏过可行数是已验证下限；即使全部状态可判定，也不等于每个子集所有类最坏值都已精确求出。", "",
              "研究决策：只有结构额外认证数及漏检可行子集的恢复足够有实际意义，才支持扩展该结构算法的主实验。单纯归一化压缩率较高不证明结构贡献，也不能将已知的子集归一化定义当新意。", ""]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"phase": "finished", "elapsed_seconds": result["elapsed_seconds"],
                      "valid_frozen_run": result["valid_frozen_run"], "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
