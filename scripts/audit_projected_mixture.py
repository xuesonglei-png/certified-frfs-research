"""Independent projected-box LP audit, reusing saved margins without new MILPs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/round3/margin-mixture-benchmark.json"
OLD_GAP = ROOT / "results/round3/margin-gap-benchmark.json"
PROTOCOL = ROOT / "research/round4/projected-mixture-protocol.md"
OUTPUT = ROOT / "results/round4/projected-mixture-audit.json"
REPORT = ROOT / "research/round4/projected-mixture-results.md"
TOL = 2e-7
ZERO = 1e-12


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    paths = [Path(__file__), PROTOCOL, INPUT, OLD_GAP,
             *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "tests").rglob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def save(result):
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def far(points, lower, upper):
    return np.maximum(np.abs(points-lower), np.abs(points-upper))


def defect(lo, hi, flo, fhi, weights):
    centers = flo / 2 + fhi / 2
    width = hi-lo
    inside = (centers > lo) & (centers < hi) & (width > 0)
    fraction = np.zeros_like(flo)
    np.divide(centers-lo, width, out=fraction, where=inside)
    matrix = np.where(inside, 2*fraction*(hi-centers), 0.)
    return float(np.max(matrix @ weights)), matrix


def original_value(lam, qlo, qhi, elo, ehi, flo, fhi, weights):
    value, count = 0., 0
    for k in range(len(weights)):
        points = np.unique(np.clip(np.r_[qlo[k], qhi[k], elo[k], ehi[k],
                                          flo[:, k]/2+fhi[:, k]/2], qlo[k], qhi[k]))
        near = np.maximum(0., np.maximum(elo[k]-points, points-ehi[k]))
        distances = far(points[:, None], flo[None, :, k], fhi[None, :, k])
        value += weights[k]*np.min(near-distances@lam)
        count += len(points)
    return float(value), count


def projected_pair(lower, upper, target, enemy, friends, enemies, weights, previous=None):
    started = time.perf_counter()
    qlo, qhi = lower[target], upper[target]
    elo, ehi = lower[enemy], upper[enemy]
    flo, fhi = lower[friends], upper[friends]
    alpha, beta = np.clip(elo, qlo, qhi), np.clip(ehi, qlo, qhi)
    gap_to_enemy = np.maximum(0., np.maximum(elo-qhi, qlo-ehi))
    e = float(weights@gap_to_enemy)
    fminus, fplus = far(alpha, flo, fhi), far(beta, flo, fhi)
    r, d = flo.shape

    def projected_value(lam):
        return float(e-weights@np.maximum(lam@fminus, lam@fplus))

    pure_values = np.array([projected_value(lam) for lam in np.eye(r)])
    lam = np.eye(r)[int(pure_values.argmax())]
    candidate_value = float(pure_values.max())
    aub = np.vstack([np.c_[fminus.T, -np.eye(d)], np.c_[fplus.T, -np.eye(d)]])
    equality = np.r_[np.ones(r), np.zeros(d)][None, :]
    mixed = linprog(np.r_[np.zeros(r), weights], A_ub=aub, b_ub=np.zeros(2*d),
                    A_eq=equality, b_eq=[1.], bounds=[(0., 1.)]*r+[(None, None)]*d,
                    method="highs", options={"time_limit": 2.})
    if mixed.x is not None and np.isfinite(mixed.x[:r]).all():
        proposed = np.maximum(mixed.x[:r], 0.)
        if proposed.max() > 0:
            proposed /= proposed.max()
            proposed /= proposed.sum()
            evaluated = projected_value(proposed)
            if evaluated > candidate_value:
                lam, candidate_value = proposed, evaluated

    # Feasible theta gives an upper bound on optimal L, independently of optimality.
    coefficients = (fplus-fminus)*weights
    constants = fminus@weights
    chord = linprog(np.r_[np.zeros(d), -1.], A_ub=np.c_[-coefficients, np.ones(r)],
                    b_ub=constants, bounds=[(0., 1.)]*d+[(None, None)],
                    method="highs", options={"time_limit": 2.})
    theta = np.full(d, .5)
    if chord.x is not None and np.isfinite(chord.x[:d]).all():
        theta = np.clip(chord.x[:d], 0., 1.)
    anchor = alpha + theta*(beta-alpha)
    chord_friend_min = float(np.min(constants + coefficients@theta))
    raw_opt_upper = e-chord_friend_min
    allowance = 1e-10*(1+max(abs(candidate_value), abs(raw_opt_upper)))
    safe_lower = candidate_value-allowance
    safe_opt_upper = raw_opt_upper+allowance
    projected_defect, defect_matrix = defect(alpha, beta, flo, fhi, weights)
    old_defect, _ = defect(qlo, qhi, flo, fhi, weights)

    z = lower/2+upper/2
    z[target] = anchor
    z[enemy] = np.clip(anchor, elo, ehi)
    z[friends] = np.where(np.abs(anchor-flo) >= np.abs(anchor-fhi), flo, fhi)
    distances = np.abs(z-z[target])@weights
    pair_attack = float(distances[enemy]-distances[friends].min())
    full_attack = float(distances[enemies].min()-distances[friends].min())
    actual_e = float(weights@np.maximum(0., np.maximum(elo-anchor, anchor-ehi)))
    identity_value, old_constraints = original_value(lam, qlo, qhi, elo, ehi, flo, fhi, weights)
    prior_value = None if previous is None else previous["raw_feasible_mixture_value"]
    prior_identity_gap = None
    if previous is not None:
        prior_lam = np.array(previous["mixture"])
        prior_original, _ = original_value(prior_lam, qlo, qhi, elo, ehi, flo, fhi, weights)
        prior_identity_gap = abs(prior_original-projected_value(prior_lam))

    checks = []
    if abs(identity_value-candidate_value) > TOL: checks.append("projected_identity_mismatch")
    if prior_identity_gap is not None and prior_identity_gap > TOL: checks.append("old_lambda_identity_mismatch")
    if prior_value is not None and abs(candidate_value-prior_value) > TOL: checks.append("old_new_optimized_value_mismatch")
    if projected_defect > old_defect+TOL: checks.append("new_defect_exceeds_old")
    if safe_lower > safe_opt_upper+TOL: checks.append("optimization_interval_conflict")
    if np.any(z < lower) or np.any(z > upper): checks.append("illegal_witness")
    if abs(actual_e-e) > TOL: checks.append("enemy_distance_not_constant")
    if pair_attack < safe_lower-TOL: checks.append("attack_below_lower")
    if pair_attack-safe_lower > projected_defect+(safe_opt_upper-safe_lower)+TOL:
        checks.append("defect_plus_optimization_gap_violated")
    if previous is not None and abs(float(pure_values.max())-previous["onehot_bound"]) > TOL:
        checks.append("pure_bound_mismatch")
    return {
        "enemy": enemy, "friends": friends, "projected_lower": alpha, "projected_upper": beta,
        "enemy_distance_constant": e, "mixture": lam, "theta": theta, "anchor": anchor,
        "pure_bound": float(pure_values.max()), "raw_feasible_lower": candidate_value,
        "safe_lower": safe_lower, "raw_optimization_upper": raw_opt_upper,
        "safe_optimization_upper": safe_opt_upper, "optimization_gap": safe_opt_upper-safe_lower,
        "raw_optimization_gap": raw_opt_upper-candidate_value,
        "old_mixture_raw": prior_value,
        "old_new_value_abs_gap": None if prior_value is None else abs(prior_value-candidate_value),
        "original_projected_identity_abs_gap": abs(identity_value-candidate_value),
        "old_lambda_identity_abs_gap": prior_identity_gap,
        "old_defect": old_defect, "new_defect": projected_defect,
        "new_defect_by_friend_attribute": defect_matrix,
        "old_zero_defect": old_defect <= ZERO, "new_zero_defect": projected_defect <= ZERO,
        "mixture_solver_success": bool(mixed.success), "mixture_solver_status": int(mixed.status),
        "chord_solver_success": bool(chord.success), "chord_solver_status": int(chord.status),
        "mixture_inequality_count": 2*d, "old_breakpoint_inequality_count": old_constraints,
        "worst_case": z, "pair_attack_upper": pair_attack, "all_enemy_attack_upper": full_attack,
        "pair_posterior_gap": pair_attack-safe_lower,
        "posterior_minus_defect_plus_optgap": pair_attack-safe_opt_upper-projected_defect,
        "checks_failed": checks, "seconds": time.perf_counter()-started,
    }


def object_record(original, table, old_record):
    X, weights = np.asarray(table["X"]), np.asarray(table["weights"])
    eps = np.asarray(table["heterogeneous_eps"]) if original["group"] == "heterogeneous_eps" else .01
    lower, upper = np.maximum(0., X-eps), np.minimum(1., X+eps)
    pairs = [projected_pair(lower, upper, original["target"], p["enemy"],
                           original["friends"], original["enemies"], weights, p["mixture"])
             for p in original["enemy_mixtures"]]
    lower_bound = min(p["safe_lower"] for p in pairs)
    attack = min(p["all_enemy_attack_upper"] for p in pairs)
    error_envelope_upper = min(p["safe_optimization_upper"]+p["new_defect"] for p in pairs)
    exact_lo, exact_hi = original["robust_margin_interval"]
    checks = []
    if lower_bound > exact_hi+TOL: checks.append("lower_above_saved_MILP_interval")
    if attack < exact_lo-TOL: checks.append("attack_below_saved_MILP_interval")
    if error_envelope_upper < exact_lo-TOL: checks.append("defect_envelope_below_saved_MILP_interval")
    if abs(pairs[0]["old_defect"]-old_record["Delta"]) > TOL: checks.append("old_defect_record_mismatch")
    if max(p["new_defect"] for p in pairs) > old_record["Delta"]+TOL: checks.append("aggregated_defect_worse")
    return {
        "table_id": original["table_id"], "group": original["group"], "d": original["d"],
        "target": original["target"], "pairs": pairs,
        "old_mixture_safe_lower": original["mixture_margin_lower"], "safe_lower": lower_bound,
        "attack_upper": attack, "posterior_gap": attack-lower_bound,
        "defect_envelope_upper": error_envelope_upper,
        "defect_envelope_gap": error_envelope_upper-lower_bound,
        "old_defect": old_record["Delta"], "max_new_defect": max(p["new_defect"] for p in pairs),
        "all_enemies_old_zero": all(p["old_zero_defect"] for p in pairs),
        "all_enemies_new_zero": all(p["new_zero_defect"] for p in pairs),
        "saved_MILP_robust_interval": original["robust_margin_interval"],
        "checks_failed": checks,
    }


def constructed_example():
    lower = np.array([[0., 0.], [.25, 0.], [.75, 0.], [.4, .3]])
    upper = np.array([[1., 0.], [.25, 0.], [.75, 0.], [.6, .3]])
    result = projected_pair(lower, upper, 0, 3, [1, 2], [3], np.array([.5, .5]))
    expected = {"pure_bound": -.025, "raw_feasible_lower": .025, "pair_attack_upper": .025,
                "old_defect": .1875, "new_defect": 0.}
    result["expected"] = expected
    result["expected_checks_failed"] = [key for key, value in expected.items() if abs(result[key]-value) > TOL]
    result["analytic_true_margin"] = .025
    result["input_lower"], result["input_upper"], result["weights"] = lower, upper, [.5, .5]
    return result


def summarize(rows):
    if not rows: return {"objects": 0, "enemy_pairs": 0}
    pairs = [p for r in rows for p in r["pairs"]]
    return {
        "objects": len(rows), "enemy_pairs": len(pairs),
        "mixture_successes": sum(p["mixture_solver_success"] for p in pairs),
        "chord_successes": sum(p["chord_solver_success"] for p in pairs),
        "pair_check_failures": sum(bool(p["checks_failed"]) for p in pairs),
        "object_check_failures": sum(bool(r["checks_failed"]) for r in rows),
        "max_old_new_value_abs_gap": max(p["old_new_value_abs_gap"] for p in pairs),
        "max_original_projected_identity_gap": max(p["original_projected_identity_abs_gap"] for p in pairs),
        "max_old_lambda_identity_gap": max(p["old_lambda_identity_abs_gap"] for p in pairs),
        "mean_old_pair_defect": float(np.mean([p["old_defect"] for p in pairs])),
        "mean_new_pair_defect": float(np.mean([p["new_defect"] for p in pairs])),
        "mean_old_object_defect": float(np.mean([r["old_defect"] for r in rows])),
        "mean_max_new_object_defect": float(np.mean([r["max_new_defect"] for r in rows])),
        "mean_new_object_envelope_gap": float(np.mean([r["defect_envelope_gap"] for r in rows])),
        "mean_new_object_posterior_gap": float(np.mean([r["posterior_gap"] for r in rows])),
        "max_new_defect_minus_old": max(p["new_defect"]-p["old_defect"] for p in pairs),
        "old_zero_defect_pairs": sum(p["old_zero_defect"] for p in pairs),
        "new_zero_defect_pairs": sum(p["new_zero_defect"] for p in pairs),
        "objects_all_enemies_old_zero": sum(r["all_enemies_old_zero"] for r in rows),
        "objects_all_enemies_new_zero": sum(r["all_enemies_new_zero"] for r in rows),
        "max_raw_optimization_gap": max(p["raw_optimization_gap"] for p in pairs),
        "max_safe_optimization_gap": max(p["optimization_gap"] for p in pairs),
        "max_posterior_minus_defect_plus_optgap": max(p["posterior_minus_defect_plus_optgap"] for p in pairs),
        "original_breakpoint_inequalities": sum(p["old_breakpoint_inequality_count"] for p in pairs),
        "projected_mixture_inequalities": sum(p["mixture_inequality_count"] for p in pairs),
        "seconds_pair_solves": sum(p["seconds"] for p in pairs),
    }


def write_report(result):
    s = result["summary"]
    report = f"""# 投影盒 2d 混合 LP：独立数值核验结果

日期：2026-09-20。状态：{result['run_status']}。固定协议见 `projected-mixture-protocol.md`，完整记录为 `results/round4/projected-mixture-audit.json`。只复用原样本与保存的 MILP 界，新增 MILP 次数为 0；src/tests 未修改，未重跑 pytest。

## 结论

在 {s['objects']} 个对象的 {s['enemy_pairs']} 个敌对象比较中，新 2d LP 与原分段 LP 的混合值最大差为 {s['max_old_new_value_abs_gap']:.12g}。投影前后对同一可行混合向量重算的最大差为 {s['max_original_projected_identity_gap']:.12g}。新理论优化了计算表达和误差界，不改变本批已优化混合证书的目标值，不能称为新增认证成功。

| 指标 | 原查询盒 | 投影敌盒 |
|---|---:|---:|
| 逐敌比较平均缺陷 | {s['mean_old_pair_defect']:.12g} | {s['mean_new_pair_defect']:.12g} |
| 每对象保守缺陷均值（新取最大敌缺陷） | {s['mean_old_object_defect']:.12g} | {s['mean_max_new_object_defect']:.12g} |
| 零缺陷敌比较数 | {s['old_zero_defect_pairs']} | {s['new_zero_defect_pairs']} |
| 全部敌比较均零缺陷的对象数 | {s['objects_all_enemies_old_zero']} | {s['objects_all_enemies_new_zero']} |
| 混合 LP 不等式数总计（不含变量界/单纯形） | {s['original_breakpoint_inequalities']} | {s['projected_mixture_inequalities']} |

新缺陷从未超过旧缺陷（最大差 {s['max_new_defect_minus_old']:.12g}）。全部对象的新数值误差包络宽度均值为 {s['mean_new_object_envelope_gap']:.12g}，可行攻击后验区间均值为 {s['mean_new_object_posterior_gap']:.12g}。

## 数值状态与边界

- 混合 LP 成功 {s['mixture_successes']}/{s['enemy_pairs']}；独立弦 LP 成功 {s['chord_successes']}/{s['enemy_pairs']}。
- 敌比较检查失败 {s['pair_check_failures']}；对象检查失败 {s['object_check_failures']}。完整未通过项目保留在逐行 `checks_failed`。
- 可行 lambda 与 theta 重算的原始优化间隙最大 {s['max_raw_optimization_gap']:.12g}；加入两侧浮点安全余量后的最大区间宽度为 {s['max_safe_optimization_gap']:.12g}。
- 每个单敌攻击使用同一个完整扰动表，检查 `U−ell <= D_ij+(u_L−ell)`；最大残差为 {s['max_posterior_minus_defect_plus_optgap']:.12g}。对象层面分别对下界、全敌攻击上界取最小。
- `u_L` 来自可行弦锚点，是混合最优值的数值上界；没有将可能次优的可行 lambda 下界直接加 D_ij 当成鲁棒间隔上界。旧 MILP 是全敌对象值，仅在相同聚合作用域比较。
- 所有计算仍是普通浮点 LP，不是形式认证。本次数值一致性不是一般恒等式的证明；理论证明需独立阅读。

## 指定有理构造

查询 `[0,1]×{{0}}`，朋友 `(.25,0),(.75,0)`，敌 `[.4,.6]×{{.3}}`，权重 `(.5,.5)`。得到纯界 −.025，混合值/合法最坏间隔 +.025；旧缺陷 .1875、新缺陷 0。期望值检查失败项目：{result['constructed_example']['expected_checks_failed']}。该例说明投影后的零缺陷条件比原查询盒中点分离更宽；并非一般数据上分类效能的证据。

## 复现

运行 `.venv/Scripts/python.exe scripts/audit_projected_mixture.py`。输出和报告已存在时默认拒绝覆盖；新复验请另行归档/使用独立副本。两份第三轮输入、脚本、协议、全部 src/tests 保存前后 SHA-256，`inputs_sources_tests_unchanged={result['inputs_sources_tests_unchanged']}`。本次合计 {result['mixture_LP_calls']} 个混合 LP、{result['chord_LP_calls']} 个弦 LP（包括构造例），0 个新 MILP。计时仅为本机探索记录，不比较硬件独立加速。
"""
    REPORT.write_text(report, encoding="utf-8")


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Preserve existing projected-mixture results before rerunning")
    before = hashes()
    original = json.loads(INPUT.read_text(encoding="utf-8"))
    old_gap = json.loads(OLD_GAP.read_text(encoding="utf-8"))
    if original["run_status"] != "complete" or len(original["rows"]) != 384:
        raise ValueError("Require the complete fixed 384-object saved benchmark")
    tables = {t["table_id"]: t for t in original["tables"]}
    by_key = {(r["table_id"], r["group"], r["target"]): r for r in old_gap["rows"]}
    started = time.perf_counter()
    result = {"protocol": "projected_enemy_box_2d_mixture_v1", "run_status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": before,
              "new_MILP_calls": 0, "comparison_tolerance": TOL, "zero_defect_tolerance": ZERO,
              "planned_objects": 384, "planned_enemy_pairs": 1536, "rows": [],
              "scope": "Fixed-sample identity/error-bound audit; floating arithmetic, not formal certification"}
    try:
        with threadpool_limits(limits=1):
            result["constructed_example"] = constructed_example()
            for row in original["rows"]:
                if time.perf_counter()-started >= 120.: break
                key = row["table_id"], row["group"], row["target"]
                result["rows"].append(object_record(row, tables[row["table_id"]], by_key[key]))
                if len(result["rows"]) % 64 == 0:
                    save(result)
                    print(f"projected objects={len(result['rows'])}/384 elapsed={time.perf_counter()-started:.2f}s", flush=True)
        result["source_hashes_end"] = hashes()
        result["inputs_sources_tests_unchanged"] = result["source_hashes_end"] == before
        if not result["inputs_sources_tests_unchanged"]: raise RuntimeError("Frozen inputs/source/tests changed during run")
        result["summary"] = summarize(result["rows"])
        result["summary_by_group_dimension"] = {
            f"{group}/d{d}": summarize([r for r in result["rows"] if r["group"] == group and r["d"] == d])
            for group in ("heterogeneous_eps", "eps_0.01") for d in (2, 4)}
        result["pending_objects"] = [{k:r[k] for k in ("table_id", "group", "target")}
                                     for r in original["rows"][len(result["rows"]):]]
        result["mixture_LP_calls"] = result["chord_LP_calls"] = result["summary"]["enemy_pairs"]+1
        result["run_status"] = "complete" if not result["pending_objects"] else "partial_time_budget"
        result["seconds"] = time.perf_counter()-started
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save(result)
        write_report(result)
        print(json.dumps(plain(result["summary"]), indent=2))
    except Exception as error:
        result.update(run_status="failed", error=repr(error), seconds=time.perf_counter()-started)
        save(result)
        raise


if __name__ == "__main__": main()
