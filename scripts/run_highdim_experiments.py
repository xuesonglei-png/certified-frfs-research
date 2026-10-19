"""High-dimensional stress experiment using the current certified selector.

Datasets are fetched from OpenML and saved under results/highdim. The selector
and classifier are the same implementations used by the primary study. The
first release runs Madelon, Isolet and Gisette when downloads are available;
each fold is atomic and records failures/timeouts rather than replacing the
certified method with an approximation.
"""
from __future__ import annotations
import json, os, sys, time, traceback, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "highdim"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

DATASETS = {"madelon": {"version": 1}, "isolet": {"version": 1}, "gisette": {"version": 1}}

def fetch_and_save(name):
    from sklearn.datasets import fetch_openml
    cache = OUT / f"{name}.npz"
    meta = OUT / f"{name}.json"
    if cache.exists() and meta.exists():
        with np.load(cache) as z:
            return z["X"], z["y"]
    data = fetch_openml(name=name, version=1, as_frame=False, parser="auto")
    X = np.asarray(data.data, dtype=np.float64)
    yraw = np.asarray(data.target)
    labels, y = np.unique(yraw, return_inverse=True)
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite].astype(np.int64)
    np.savez_compressed(cache, X=X, y=y)
    meta.write_text(json.dumps({"name": name, "source": "OpenML",
                                "openml_version": 1, "shape": list(X.shape),
                                "labels": labels.tolist(),
                                "removed_nonfinite_rows": int((~finite).sum())}, indent=2), encoding="utf-8")
    return X, y

def run_one(name, X, y, seed=17, max_samples=None, output_suffix=""):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score
    from certified_frfs.selection import deletion_order, select_certified
    if max_samples is not None and len(X) > max_samples:
        # Stratified deterministic cap: preserves the original feature space
        # and selector, while making a high-dimensional feasibility test finite.
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(X)), train_size=max_samples,
                                   stratify=y, random_state=seed)
        X, y = X[keep], y[keep]
    counts = np.unique(y, return_counts=True)[1]
    nfold = min(3, int(counts.min()))
    train, test = list(StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(X, y))[0]
    imputer, scaler = SimpleImputer(strategy="median"), MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    weights = np.ones(Xtr.shape[1]) / np.sqrt(Xtr.shape[1])
    start = time.perf_counter()
    order = deletion_order(Xtr, y[train], seed)
    order_seconds = time.perf_counter() - start
    selected = select_certified(Xtr, y[train], eps=.01, eta=.10, weights=weights, order=order)
    select_seconds = time.perf_counter() - start
    feats = np.asarray(selected["features"], dtype=int)
    timings = {}
    scores = {}
    for method, fs in [("all", np.arange(Xtr.shape[1])), ("certified", feats)]:
        model = KNeighborsClassifier(n_neighbors=min(5, len(train))).fit(Xtr[:, fs], y[train])
        for _ in range(2): model.predict(Xte[:, fs])
        vals=[]
        for _ in range(7):
            t=time.perf_counter_ns(); model.predict(Xte[:,fs]); vals.append(time.perf_counter_ns()-t)
        med=float(np.median(vals)); timings[method]={"batch_ns":med,"per_sample_us":med/len(test)/1000,
                                                     "speedup_vs_all":1.0}
        scores[method]=float(balanced_accuracy_score(y[test],model.predict(Xte[:,fs])))
    timings["certified"]["speedup_vs_all"] = timings["all"]["per_sample_us"] / timings["certified"]["per_sample_us"]
    result={"dataset":name,"seed":seed,"shape":list(X.shape),"n_train":len(train),"n_test":len(test),
            "sample_cap":max_samples,
            "d":int(Xtr.shape[1]),"n_features_certified":int(len(feats)),
            "removed_fraction":float(1-len(feats)/Xtr.shape[1]),"certificate_upper":float(selected["certificate"]["upper_bound"]),
            "balanced_accuracy":scores,"timings":timings,"order_seconds":order_seconds,
            "selection_seconds":select_seconds,"status":"complete"}
    (OUT/f"{name}-s{seed}{output_suffix}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=list(DATASETS))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[17,29,43])
    args = ap.parse_args()
    for name in args.names:
        try:
            print("FETCH",name,flush=True); X,y=fetch_and_save(name)
            print("DATA",name,X.shape,flush=True)
            for seed in args.seeds:
                suffix = f"-n{args.max_samples}" if args.max_samples else ""
                out=OUT/f"{name}-s{seed}{suffix}.json"
                if out.exists(): print("CACHED",out,flush=True); continue
                print("RUN",name,seed,flush=True)
                print(json.dumps(run_one(name,X,y,seed,args.max_samples,suffix)),flush=True)
        except Exception as exc:
            (OUT/f"{name}-error.json").write_text(json.dumps({"status":"error","error":repr(exc),"traceback":traceback.format_exc()},indent=2),encoding="utf-8")
            print("ERROR",name,repr(exc),flush=True)

if __name__ == "__main__": main()
