"""Posthoc development prototype of an expanded certified candidate pool."""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from run_decision_iteration import ROOT,scaled_split
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs import baselines
from certified_frfs.graph_recent import select_fgs,combined_certified_candidates
from certified_frfs.knn_decision import certified_knn_coverage
from certified_frfs.core import certificate
from threadpoolctl import threadpool_limits


def run(name):
    threadpool_limits(1)
    old=ROOT/'results/predictive-iteration18'/f'{name}-s101-f0.json'
    obj=json.loads(old.read_text());X,y=load_dataset(name)
    tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
    A,B=scaled_split(X[tr],X[te]);w=np.array(obj['rows'][0]['weights'])
    out=ROOT/'results/pool-prototype'/f'{name}-s101-f0.json'
    if out.exists():raise ValueError('preserve existing prototype outputs')
    start=time.perf_counter();base=select_fgs(A*w,y[tr],beta=.8,xi=1.)
    relevance=baselines.mi_scores(A*w,y[tr],random_state=101)
    ranking=np.argsort(relevance,kind='stable').tolist()
    pool=combined_certified_candidates(A,y[tr],base,ranking,eps=.01,eta=.1,weights=w)
    trace=[]
    for fs in pool['candidates']:
        loss=certificate(A,y[tr],fs,weights=w,eps=.01)
        assert loss['upper_bound']<=.1+1e-10
        nominal=certified_knn_coverage(A,y[tr],fs,k=5,eps=0,weights=w)
        robust=certified_knn_coverage(A,y[tr],fs,k=5,eps=.01,weights=w)
        trace.append({'features':fs,'loss_certificate':loss,'nominal':nominal,'robust':robust})
    rows=[]
    for mode in ['nominal','robust','compression']:
        def objective(row):
            if mode=='compression':return -len(row['features'])
            coverage=row[mode]
            return (coverage['balanced_coverage']+coverage['min_class_coverage'])/2-.02*len(row['features'])/A.shape[1]
        selected=max(trace,key=lambda r:(round(objective(r),12),-len(r['features'])))
        fs=selected['features'];metrics=scores(A,y[tr],B,y[te],fs,104,w)
        rows.append({'mode':mode,'features':fs,'metrics':metrics,'loss_certificate':selected['loss_certificate']})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'seed':101,'fold':0,'d':A.shape[1],
        'train_indices':tr.tolist(),'test_indices':te.tolist(),'weights':w.tolist(),
        'pool':pool,'candidate_trace':trace,'rows':rows,'seconds':time.perf_counter()-start,
        'status':'posthoc_development_prototype_fixed_parameters_not_confirmatory'},indent=2),encoding='utf-8')
    print(json.dumps({'dataset':name,'candidates':len(trace),'rows':[{**r,'loss_certificate':r['loss_certificate']['upper_bound']} for r in rows]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere']);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,n) for n in args.datasets]
        for f in as_completed(futures):f.result()
