"""Frozen iteration20 models: joint-reference/query perturbation diagnostic."""
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


def balanced(mask,truth):
    return float(np.mean([np.asarray(mask)[truth==c].mean() for c in np.unique(truth)]))


def run(path,iteration=20):
    threadpool_limits(1);path=Path(path)
    obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
    tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
    A,B=preprocess_fit(X[tr],X[te]);rows=[]
    out=ROOT/f'results/joint-perturbation{iteration}'/path.name
    if out.exists():
        previous=json.loads(out.read_text())
        assert previous['input_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        for relative in ['src/certified_frfs/knn_query.py','src/certified_frfs/margin.py']:
            key=str(Path(relative))
            assert previous['source_sha256'][key]==hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        return
    for row in obj['rows']:
        detail=row['detail'];fs=detail['features'];w=np.array(detail['weights'])
        def predict(ref,queries):
            return KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute').fit(
                ref[:,fs]*w[fs],y[tr]).predict(queries[:,fs]*w[fs])
        nominal=predict(A,B)
        assert np.isclose(balanced_accuracy_score(y[te],nominal),row['metrics']['0.0']['balanced_accuracy'])
        records=[]
        for eps in [.01,.05]:
            cert=certify_knn_queries(A,y[tr],B,nominal,fs,weights=w,eps=eps,k=5)
            flags=np.array(cert['certified']);correct=nominal==y[te]
            rng=np.random.default_rng(20261009+obj['fold'])
            values=[];any_wrong=~correct;any_change=np.zeros(len(B),bool)
            for draw in range(20):
                # One coherent reference-table perturbation per draw; common
                # random numbers for all models on a given split/radius.
                ref=np.clip(A+rng.uniform(-eps,eps,A.shape),0,1)
                queries=np.clip(B+rng.uniform(-eps,eps,B.shape),0,1)
                pred=predict(ref,queries)
                if np.any(flags & (pred!=nominal)):
                    raise AssertionError('observed query certificate violation')
                values.append(float(balanced_accuracy_score(y[te],pred)))
                any_wrong|=pred!=y[te];any_change|=pred!=nominal
            records.append({'eps':eps,'certificate':cert,
                'balanced_certified_correct':balanced(flags&correct,y[te]),
                'balanced_certified_wrong':balanced(flags&~correct,y[te]),
                'balanced_coverage':balanced(flags,y[te]),
                'joint_draw_balanced_accuracy':values,'joint_mean':float(np.mean(values)),
                'joint_min_over20':min(values),'pointwise_correct_all20_plus_nominal':balanced(~any_wrong,y[te]),
                'balanced_observed_instability':balanced(any_change,y[te])})
        rows.append({'method':row['method'],'mode':row['mode'],'features':fs,'records':records})
        print(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'method':row['method'],
                          'mode':row['mode'],'correct_cert01':records[0]['balanced_certified_correct']}),flush=True)
    sources=[Path(__file__),ROOT/'src/certified_frfs/knn_query.py',ROOT/'src/certified_frfs/margin.py']
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'rows':rows,
        'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'status':'posthoc_stress_diagnostic_random_draws_not_exact_worst_case'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-folds',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--iteration',type=int,choices=[20,21],default=20);a=p.parse_args()
    files=sorted((ROOT/f'results/predictive-iteration{a.iteration}').glob('*-s101-f*.json' if a.all_folds else '*-s101-f0.json'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,str(path),a.iteration) for path in files]
        for f in as_completed(futures):f.result()
