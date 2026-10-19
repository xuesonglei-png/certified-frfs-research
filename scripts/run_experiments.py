"""Actual reproducible experiments; each completed fold is an atomic JSON file."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import traceback
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL_VERSION = "2026-09-19-v2"


def code_fingerprint():
    paths = [Path(__file__), *sorted((ROOT/"src"/"certified_frfs").glob("*.py"))]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def plain(value):
    import numpy as np
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(plain(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def run_fold(job):
    # Avoid oversubscribing numerical libraries across worker processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from certified_frfs.datasets import load_dataset
    from certified_frfs import baselines
    from certified_frfs.core import certificate, gamma
    from certified_frfs.selection import select_certified, deletion_order, select_nominal_backward
    from certified_frfs.classifiers import FuzzyRoughScoreClassifier
    name, seed, fold, eps, eta, output, mode, fingerprint = job
    suffix = f"{name}-s{seed}-f{fold}"
    path = Path(output) / (suffix + ".json")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        expected = {"eps":eps,"eta":eta,"mode":mode,"protocol_version":PROTOCOL_VERSION,
                    "code_fingerprint":fingerprint}
        if any(cached.get(k) != v for k,v in expected.items()):
            raise ValueError(f"Stale cache {path}; use a fresh output directory or archive this run")
        return {"job": suffix, "status": "cached_verified"}
    start = time.perf_counter()
    X, y = load_dataset(name)
    splits = min(3, int(np.unique(y, return_counts=True)[1].min()))
    split = list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed).split(X,y))[fold]
    train, test = split
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    Xtr = scaler.fit_transform(imputer.fit_transform(X[train]))
    Xte = scaler.transform(imputer.transform(X[test]))
    ytr, yte = y[train], y[test]
    weights = np.ones(Xtr.shape[1]) / np.sqrt(Xtr.shape[1])
    order_start=time.perf_counter()
    order = deletion_order(Xtr, ytr, seed)
    order_seconds=time.perf_counter()-order_start
    selected = select_certified(Xtr,ytr,eps=eps,eta=eta,weights=weights,order=order)
    k = len(selected["features"])
    subsets = {"all": list(range(Xtr.shape[1])), "certified": selected["features"]}
    selection_times = {"all": 0.0, "certified": selected["seconds"]+order_seconds}
    for method, fn in [
        ("nominal", lambda: baselines.select_nominal_frfs(Xtr,ytr,eta=eta,weights=weights)),
        ("mi_matched", lambda: baselines.select_mi(Xtr,ytr,k=k,random_state=seed)),
        ("mrmr_matched", lambda: baselines.select_mrmr(Xtr,ytr,k=k,random_state=seed)),
        ("relieff_matched", lambda: baselines.select_relieff(Xtr,ytr,k=k,random_state=seed)),
        ("maf_framework", lambda: baselines.select_mafrfs_additive(Xtr,ytr,k=k,weights=weights,sop=2,margin_type="local")),
    ]:
        t = time.perf_counter()
        subsets[method] = list(map(int, fn())) or [int(order[-1])]
        selection_times[method] = time.perf_counter()-t
    rng = np.random.default_rng(seed * 1000 + fold)
    # All methods/classifiers see identical perturbation realizations.
    perturbations = {0.0: [Xte]}
    for radius in [0.01, 0.05]:
        perturbations[radius] = [np.clip(Xte+rng.uniform(-radius,radius,Xte.shape),0,1)
                                 for _ in range(5)]
    training_attacks = [np.clip(Xtr+rng.uniform(-eps,eps,Xtr.shape),0,1) for _ in range(5)]
    attack_full = [gamma(z,ytr,weights=weights) for z in training_attacks]
    rows = []
    subset_results = {}
    for method, feats in subsets.items():
        cert = selected["certificate"] if method == "certified" else certificate(Xtr,ytr,feats,weights,eps)
        observed = max([float(cert["nominal_loss"])] +
                       [a-gamma(z,ytr,feats,weights) for a,z in zip(attack_full,training_attacks)])
        if observed > cert["upper_bound"] + 1e-8:
            raise AssertionError(f"Certificate violated for {suffix}/{method}: {observed}>{cert['upper_bound']}")
        subset_results[method] = {"features": feats, "certificate": cert,
                                  "selection_seconds": selection_times[method],
                                  "attack_lower_bound": observed}
        models = {"knn": KNeighborsClassifier(n_neighbors=min(5,len(train))),
                  "svm": SVC(C=1.0,kernel="rbf",gamma="scale"),
                  "fuzzy_rough": FuzzyRoughScoreClassifier(weights=weights[feats])}
        for model_name, model in models.items():
            fit_start=time.perf_counter()
            model.fit(Xtr[:,feats],ytr)
            fit_seconds=time.perf_counter()-fit_start
            for radius, perturbed in perturbations.items():
                bacc, f1 = [], []
                for z in perturbed:
                    pred=model.predict(z[:,feats])
                    bacc.append(balanced_accuracy_score(yte,pred))
                    f1.append(f1_score(yte,pred,average="macro",zero_division=0))
                rows.append({"dataset":name,"seed":seed,"fold":fold,"method":method,
                             "classifier":model_name,"test_epsilon":radius,
                             "balanced_accuracy":float(np.mean(bacc)),"macro_f1":float(np.mean(f1)),
                             "n_features":len(feats),"d":Xtr.shape[1],"n_train":len(train),
                             "certificate_upper":cert["upper_bound"],
                             "stable_fraction":cert["stable_fraction"] if cert.get("stability_evaluated",True) else None,
                             "selection_seconds":selection_times[method],"fit_seconds":fit_seconds})
    result={"dataset":name,"seed":seed,"fold":fold,"eps":eps,"eta":eta,
            "train_indices":train,"test_indices":test,"weights":weights,
            "scaler_min":scaler.data_min_,"scaler_max":scaler.data_max_,
            "imputer_statistics":imputer.statistics_,"subsets":subset_results,
            "deletion_history":selected["history"],"rows":rows,"order_seconds":order_seconds,
            "protocol_version":PROTOCOL_VERSION,"code_fingerprint":fingerprint,
            "total_seconds":time.perf_counter()-start,"mode":mode,"n_splits":splits}
    atomic_json(path,result)
    return {"job":suffix,"status":"complete","features":k,"d":Xtr.shape[1],
            "seconds":round(result["total_seconds"],2)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="*")
    parser.add_argument("--seeds",nargs="+",type=int,default=[17,29,43])
    parser.add_argument("--folds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--eps",type=float,default=.01)
    parser.add_argument("--eta",type=float,default=.10)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--output",default="results/primary")
    parser.add_argument("--mode",default="primary")
    args=parser.parse_args()
    import numpy as np
    from certified_frfs.datasets import available_datasets, load_dataset
    names=args.datasets or available_datasets()
    out=str(ROOT/args.output)
    fingerprint=code_fingerprint()
    config=vars(args)|{"datasets":names,"protocol_version":PROTOCOL_VERSION,"code_fingerprint":fingerprint}
    config_path=Path(out)/"config.json"
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding="utf-8"))
        for key in ["eps","eta","mode","protocol_version","code_fingerprint"]:
            if old.get(key)!=config[key]:
                raise ValueError(f"Conflicting run configuration ({key}); select a fresh output directory")
    atomic_json(config_path,config)
    jobs=[]
    for name in names:
        _, labels=load_dataset(name)
        splits=min(3,int(np.unique(labels,return_counts=True)[1].min()))
        jobs.extend((name,seed,fold,args.eps,args.eta,out,args.mode,fingerprint)
                    for seed in args.seeds for fold in args.folds if fold < splits)
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_fold,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try: print(json.dumps(future.result()),flush=True)
            except Exception as exc:
                record={"job":futures[future],"error":repr(exc),"traceback":traceback.format_exc()}
                errors.append(record)
                print(json.dumps(record),flush=True)
    atomic_json(Path(out)/"errors.json",errors)
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
