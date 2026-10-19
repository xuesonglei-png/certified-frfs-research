"""Generate transparent, publication-ready figures for the compression trade-off."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper-round4" / "figures"
OUT.mkdir(exist_ok=True)

avg = pd.read_csv(ROOT / "results" / "dataset-means.csv")
clean = avg[(avg.classifier == "knn") & (avg.test_epsilon == 0)].pivot(
    index="dataset", columns="method", values="balanced_accuracy")
cert = avg[(avg.method == "certified") & (avg.classifier == "knn") &
           (avg.test_epsilon == 0)].set_index("dataset")
trade = pd.DataFrame({
    "dataset": clean.index,
    "removed": [100 * (1 - cert.loc[d, "n_features"] / cert.loc[d, "d"]) for d in clean.index],
    "delta": [100 * (clean.loc[d, "certified"] - clean.loc[d, "all"]) for d in clean.index],
    "bound": [cert.loc[d, "certificate_upper"] for d in clean.index],
})

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=220)
ax.axhline(0, color="#444444", lw=0.8)
ax.scatter(trade.removed, trade.delta, c=trade.bound, cmap="viridis", s=58,
           edgecolor="white", linewidth=0.5, zorder=3)
for r in trade.itertuples():
    ax.annotate(r.dataset, (r.removed, r.delta), xytext=(4, 4),
                textcoords="offset points", fontsize=7)
ax.set_xlabel("Removed attributes (%)")
ax.set_ylabel("Certified − all-attribute kNN accuracy (percentage points)")
ax.set_title("Compression–prediction trade-off across 15 datasets")
cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.02)
cbar.set_label("Certificate upper bound")
fig.tight_layout()
fig.savefig(OUT / "compression-accuracy-tradeoff.pdf", bbox_inches="tight")
fig.savefig(OUT / "compression-accuracy-tradeoff.png", dpi=240, bbox_inches="tight")
plt.close(fig)

audit = json.loads((ROOT / "results" / "round4" / "fr-separate-joint-audit.json").read_text())
rows = audit["rows"]
gap = pd.DataFrame({"separate": [r["separate_lower"] for r in rows],
                    "joint": [r["joint_lower"] for r in rows]})
fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=220)
ax.boxplot([gap.separate, gap.joint], tick_labels=["Separate", "Joint"],
           patch_artist=True, boxprops={"facecolor": "#8ecae6"},
           medianprops={"color": "#222222"})
ax.axhline(0, color="#444444", lw=0.8)
ax.set_ylabel("Certified lower bound")
ax.set_title("Joint fuzzy-rough certification tightens the lower bound")
fig.tight_layout()
fig.savefig(OUT / "joint-certification-tightening.pdf", bbox_inches="tight")
fig.savefig(OUT / "joint-certification-tightening.png", dpi=240, bbox_inches="tight")
plt.close(fig)

trade.to_csv(ROOT / "results" / "round4" / "compression-accuracy-tradeoff.csv", index=False)
print("generated", OUT / "compression-accuracy-tradeoff.pdf", OUT / "joint-certification-tightening.pdf")
