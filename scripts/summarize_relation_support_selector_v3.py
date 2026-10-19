"""Regenerate the descriptive summary from the completed v3 artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "relation-support-selector-v3"

def main():
    rows = [json.loads(p.read_text()) for p in sorted(FOLDER.glob("*.json"))
            if p.name != "complete.json" and not p.name.endswith("-audit.json")]
    deltas, robust, matched, removed = [], [], [], []
    failed = 0
    for row in rows:
        for r in row["rows"]:
            if r.get("status", "complete") != "complete":
                failed += 1; continue
            d = r["proposed"]["clean_BA"] - r["full"]["clean_BA"]
            deltas.append(100*d)
            robust.append(100*(r["perturbed_BA"]-r["full_perturbed_BA"]))
            matched.append(100*(r["proposed"]["clean_BA"]-r["matched_uniform"]["clean_BA"]))
            removed.append(1-len(r["fit"]["selected"]["features"])/row["d"])
    out = {"datasets": len(rows), "outer_folds": len(deltas), "failed_folds": failed,
           "mean_clean_delta_vs_full_pp": float(np.mean(deltas)),
           "mean_perturbed_delta_vs_full_pp": float(np.mean(robust)),
           "mean_clean_delta_vs_matched_uniform_pp": float(np.mean(matched)),
           "wins_vs_full": int(sum(x > 1e-12 for x in deltas)),
           "losses_vs_full": int(sum(x < -1e-12 for x in deltas)),
           "wins_vs_matched_uniform": int(sum(x > 1e-12 for x in matched)),
           "mean_removed": float(np.mean(removed)), "goal_achieved": False,
           "independent_audit_required": True,
           "scope": "descriptive_nested_development_summary_no_SOTA_claim"}
    (FOLDER/"complete.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
