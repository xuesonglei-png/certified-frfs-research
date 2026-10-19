"""Training-only fixed diagnostic: one lowest-bound rejection per task/fold.

Uses all final-path dependency rejections, picks the closest to .1 using
old training bounds only. Not a random sample or a predictive experiment.
"""
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import gamma
from certified_frfs.mixture_loss import mixture_loss_bound


def run(job):
    threadpool_limits(1);name,fold=job
    source=ROOT/'results/predictive-iteration36'/f'{name}-s101-f{fold}.json'
    o=json.loads(source.read_text());r=next(r for r in o['rows'] if r['method']=='protected')
    rejected=[a for s in r['path']['trace'] for a in s['certification_attempts']
              if a['majority_feasible'] and not a['feasible']]
    if not rejected:return {'dataset':name,'fold':fold,'status':'no_dependency_rejection'}
    chosen=min(rejected,key=lambda a:(a['dependency_certificate']['upper_bound'],tuple(a['features'])))
    X,y=load_dataset(name);tr=np.array(o['train_indices']);A,_=preprocess_fit(X[tr],X[tr]);w=np.array(r['weights'])
    start=time.perf_counter();cert=mixture_loss_bound(A,y[tr],chosen['features'],w);seconds=time.perf_counter()-start
    rng=np.random.default_rng(937);losses=[]
    for _ in range(20):
        Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        losses.append(gamma(Z,y[tr],weights=w)-gamma(Z,y[tr],chosen['features'],weights=w))
    assert max(losses)<=cert['upper_bound']+1e-10
    old=chosen['dependency_certificate']['upper_bound'];assert cert['upper_bound']<=old+1e-12
    result={'dataset':name,'fold':fold,'status':'training_certificate_diagnostic',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'features':chosen['features'],
        'old_bound':old,'new_bound':cert['upper_bound'],'improvement':old-cert['upper_bound'],
        'newly_certified':cert['upper_bound']<=.1,'certificate':cert,'seconds':seconds,
        'sampled_losses':losses,'sampled_violations':0}
    print(name,fold,old,cert['upper_bound'],'seconds',seconds,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--folds',nargs='+',type=int,default=[0]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=[f.result() for f in as_completed([pool.submit(run,(n,k)) for n in a.datasets for k in a.folds])]
    tested=[r for r in rows if r['status']=='training_certificate_diagnostic']
    result={'scope':'fixed_closest_training_rejection_per_dataset_fold_not_representative_or_predictive',
        'rows':rows,'tested':len(tested),'newly_certified':sum(r['newly_certified'] for r in tested),
        'sampled_checks':20*len(tested),'sampled_violations':0}
    directory=ROOT/'results/mixture-loss-diagnostic';directory.mkdir(exist_ok=True)
    out=directory/('folds-'+ '-'.join(map(str,a.folds))+'.json')
    if out.exists():raise ValueError('preserve prior evidence')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
