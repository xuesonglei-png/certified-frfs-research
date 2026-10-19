"""Dataset-level recent-comparator summary; no split treated as independent."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
groups = {}
for path in sorted((ROOT / "results/recent-sfrip").glob("*-s*-f*.json")):
    result = json.loads(path.read_text())
    original = json.loads((ROOT / "results/primary" / path.name).read_text())
    group = groups.setdefault(result["dataset"], {"removed": [], "budget": [], "rows": {}})
    group["removed"].append(1 - result["n_features"] / result["d"])
    group["budget"].append(result["meets_our_certificate_budget"])
    for row in result["rows"]:
        key = f'{row["classifier"]}/{row["test_epsilon"]}'
        match = next(r for r in original["rows"] if r["method"] == "certified"
                     and r["classifier"] == row["classifier"]
                     and r["test_epsilon"] == row["test_epsilon"])
        values = group["rows"].setdefault(key, [])
        values.append([row["balanced_accuracy"], match["balanced_accuracy"], row["macro_f1"]])
dataset_results = {}
for dataset, group in groups.items():
    dataset_results[dataset] = {
        "removed_fraction": float(np.mean(group["removed"])),
        "budget_pass_fraction": float(np.mean(group["budget"])),
        "rows": {k: np.mean(v, axis=0).tolist() for k, v in group["rows"].items()}}
summary = {"completed_splits": sum(len(g["removed"]) for g in groups.values()),
           "n_datasets": len(groups), "datasets": dataset_results,
           "mean_removed_fraction": float(np.mean([g["removed_fraction"] for g in dataset_results.values()])),
           "budget_pass_splits": sum(sum(g["budget"]) for g in groups.values()),
           "performance": {}}
for key in next(iter(dataset_results.values()))["rows"]:
    values = np.asarray([g["rows"][key] for g in dataset_results.values()])
    delta = values[:, 1] - values[:, 0]
    summary["performance"][key] = {
        "recent_balanced_accuracy": float(values[:, 0].mean()),
        "certified_balanced_accuracy": float(values[:, 1].mean()),
        "certified_minus_recent": float(delta.mean()),
        "paired_wilcoxon_two_sided_raw": float(wilcoxon(delta).pvalue),
        "recent_macro_f1": float(values[:, 2].mean())}
out = ROOT / "results/recent-sfrip/summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "datasets"}, indent=2))
