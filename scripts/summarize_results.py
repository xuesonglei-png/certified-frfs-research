"""Produce tables, figures, statistics and factual text from actual run files."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
FIGURES=ROOT/"paper"/"figures"
FIGURES.mkdir(parents=True,exist_ok=True)


def main():
    runs=[]
    for path in sorted((RESULTS/"primary").glob("*-s*-f*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs: raise SystemExit("No completed primary runs")
    rows=pd.DataFrame([row for run in runs for row in run["rows"]])
    rows.to_csv(RESULTS/"primary-results.csv",index=False)
    # Folds are averaged within dataset before any statistical comparison.
    avg=rows.groupby(["dataset","method","classifier","test_epsilon"],as_index=False).agg(
        balanced_accuracy=("balanced_accuracy","mean"),macro_f1=("macro_f1","mean"),
        n_features=("n_features","mean"),d=("d","first"),
        certificate_upper=("certificate_upper","mean"),stable_fraction=("stable_fraction","mean"),
        selection_seconds=("selection_seconds","mean"))
    avg.to_csv(RESULTS/"dataset-means.csv",index=False)
    overall=avg.groupby(["method","classifier","test_epsilon"],as_index=False).mean(numeric_only=True)
    overall.to_csv(RESULTS/"overall-means.csv",index=False)
    clean=avg.query("test_epsilon == 0 and classifier == 'knn'")
    pivot=clean.pivot(index="dataset",columns="method",values="balanced_accuracy")
    comparisons=[]
    rng=np.random.default_rng(20260919)
    for baseline in [x for x in pivot if x!="certified"]:
        diff=(pivot.certified-pivot[baseline]).dropna().to_numpy()
        p=1.0 if np.allclose(diff,0) else float(wilcoxon(diff,zero_method="wilcox").pvalue)
        boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
        comparisons.append({"baseline":baseline,"datasets":len(diff),"mean_delta":diff.mean(),
                            "median_delta":np.median(diff),"ci_low":np.quantile(boot,.025),
                            "ci_high":np.quantile(boot,.975),"p_raw":p})
    rank=np.argsort([r["p_raw"] for r in comparisons]); previous=0
    for i,k in enumerate(rank):
        adjusted=min(1,(len(rank)-i)*comparisons[k]["p_raw"])
        previous=max(previous,adjusted); comparisons[k]["p_holm"]=previous
    pd.DataFrame(comparisons).to_csv(RESULTS/"statistics-knn-clean.csv",index=False)
    cert=clean.query("method == 'certified'").set_index("dataset")
    cert["removed_fraction"]=1-cert.n_features/cert.d
    cert.to_csv(RESULTS/"certified-by-dataset.csv")
    fig,ax=plt.subplots(figsize=(9,4.8))
    cert.removed_fraction.sort_values().plot.barh(ax=ax,color="#2878a0")
    ax.set_xlabel("Fraction of attributes removed");ax.set_ylabel("")
    ax.set_xlim(0,1);fig.tight_layout();fig.savefig(FIGURES/"compression.pdf");fig.savefig(FIGURES/"compression.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for method,group in avg.query("classifier == 'knn'").groupby("method"):
        means=group.groupby("test_epsilon").balanced_accuracy.mean()
        ax.plot(means.index,means.values,marker="o",label=method)
    ax.set_xlabel("Held-out input perturbation radius");ax.set_ylabel("Mean balanced accuracy")
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(FIGURES/"classification.pdf");fig.savefig(FIGURES/"classification.png",dpi=180);plt.close(fig)
    sensitivity=[]
    for path in sorted((RESULTS/"sensitivity").glob("*.json")):
        if ".partial." in path.name: continue
        payload=json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"): continue
        for row in payload["rows"]:
            flat={k:v for k,v in row.items() if k not in ["certificate","features"]}
            flat.update({"cert_"+k:v for k,v in row["certificate"].items()
                         if isinstance(v,(int,float))})
            sensitivity.append(flat)
    if sensitivity:
        sens=pd.DataFrame(sensitivity)
        sens.to_csv(RESULTS/"sensitivity-results.csv",index=False)
        ab=sens.query("eps == .01 and eta == .1 and scale == 'sqrt'")
        ab.groupby("method").mean(numeric_only=True).to_csv(RESULTS/"ablation-means.csv")
        fig,ax=plt.subplots(figsize=(7,4.5))
        sweep=sens.query("method == 'hybrid' and scale == 'sqrt'").copy()
        sweep["removed_fraction"]=1-sweep.n_features/sweep.d
        for eta,group in sweep.groupby("eta"):
            curve=group.groupby("eps").removed_fraction.mean()
            ax.plot(curve.index,curve.values,marker="o",label=f"eta={eta:.2f}")
        ax.set_xlabel("Training-table perturbation radius");ax.set_ylabel("Mean fraction removed")
        ax.legend();fig.tight_layout();fig.savefig(FIGURES/"sensitivity.pdf");fig.savefig(FIGURES/"sensitivity.png",dpi=180);plt.close(fig)
    summary={"completed_runs":len(runs),"datasets":len(cert),"evaluation_rows":len(rows),
             "mean_removed_fraction":float(cert.removed_fraction.mean()),
             "mean_certificate_upper":float(cert.certificate_upper.mean()),
             "max_certificate_upper":float(rows.query("method == 'certified'").certificate_upper.max()),
             "mean_stable_fraction":float(cert.stable_fraction.mean()),
             "mean_selection_seconds":float(cert.selection_seconds.mean()),
             "zero_removal_datasets":cert[cert.removed_fraction==0].index.tolist(),
             "knn_clean_comparisons":comparisons}
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
