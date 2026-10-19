"""Posthoc query stability/correctness audit of fixed iteration24/25 models."""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.knn_query import certify_knn_queries


def balanced(mask,y):
    return float(np.mean([np.asarray(mask)[y==label].mean() for label in np.unique(y)]))


def run(job):
    path,iteration=job;threadpool_limits(1);obj=json.loads(Path(path).read_text())
    X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
    A,B=preprocess_fit(X[tr],X[te]);rows=[]
    for row in obj['rows']:
        fs=row['features'];w=np.array(row['weights'])
        model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
        def predict(ref,query):
            return model.fit(ref[:,fs]*w[fs],y[tr]).predict(query[:,fs]*w[fs])
        nominal=predict(A,B);correct=nominal==y[te]
        assert abs(balanced_accuracy_score(y[te],nominal)-row['metrics']['clean'])<1e-12
        old=certify_knn_queries(A,y[tr],B,nominal,fs,weights=w,allowed_enemies=0)
        new=certify_knn_queries(A,y[tr],B,nominal,fs,weights=w,allowed_enemies=2)
        flags=np.array(new['certified']);assert np.all(~np.array(old['certified'])|flags)
        rng=np.random.default_rng(20261009+obj['fold']);values=[]
        for _ in range(20):
            ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            query=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            pred=predict(ref,query);assert not np.any(flags&(pred!=nominal))
            values.append(float(balanced_accuracy_score(y[te],pred)))
        rows.append({'method':row['method'],'original_certificate':old,'budget_certificate':new,
            'balanced_certified_correct':balanced(flags&correct,y[te]),
            'balanced_certified_wrong':balanced(flags&~correct,y[te]),
            'balanced_original_certified_correct':balanced(np.array(old['certified'])&correct,y[te]),
            'joint_mean_BA':float(np.mean(values)),'joint_draw_BA':values})
    out=ROOT/f'results/majority-query-audit{iteration}'/Path(path).name
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('preserve query audits')
    out.write_text(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'rows':rows,
        'input_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'scope':'nominal_label_stability_certificate_not_query_truth_or_population_risk'},indent=2),encoding='utf-8')
    print(obj['dataset'],obj['fold'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[24,25,33],default=25)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    files=sorted((ROOT/f'results/predictive-iteration{a.iteration}').glob('*-s101-f*.json'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(str(path),a.iteration)) for path in files]
        for future in as_completed(futures):future.result()
