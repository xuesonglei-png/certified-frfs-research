"""Iteration24: task-aligned majority metric/protection development experiment."""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs import baselines
from certified_frfs.recent import select_sfrip
from certified_frfs.validated_pool import preprocess_fit,metric_weights,validation_scores
from certified_frfs.majority_metric import learn_majority_weights,protected_majority_path,MajorityTerms


def weights(A,y,family):
    return learn_majority_weights(A,y) if family=='majority' else metric_weights(A,y,family,101)


def discovery_split(y):
    y=np.asarray(y);labels,counts=np.unique(y,return_counts=True)
    singleton=np.isin(y,labels[counts==1]);ids=np.flatnonzero(~singleton)
    ii,jj=train_test_split(ids,test_size=.3,stratify=y[ids],random_state=142)
    # A singleton cannot be stratified; keep it in discovery for all methods.
    return np.r_[ii,np.flatnonzero(singleton)],jj


def run(job):
    name,fold,iteration=job;threadpool_limits(1)
    allowed_enemies=2 if iteration==25 else 0
    out=ROOT/f'results/predictive-iteration{iteration}'/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError('preserve iteration24')
    sources=[Path(__file__),*sorted((ROOT/'src/certified_frfs').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    X,y=load_dataset(name);d=X.shape[1]
    tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[fold]
    ii,jj=discovery_split(y[tr])
    A,B=preprocess_fit(X[tr][ii],X[tr][jj]);F,Q=preprocess_fit(X[tr],X[te])
    traces={method:[] for method in ['protected','all','mi','relief','mrmr','sfrip']}
    paths=[];final_weights={}
    for family in ['uniform','triplet','majority']:
        w=weights(A,y[tr][ii],family);final_weights[family]=weights(F,y[tr],family)
        candidates={tuple(range(d))}
        for rank in ['mi','relief','weights']:
            relevance=(baselines.mi_scores(A*w,y[tr][ii],random_state=101) if rank=='mi'
                else baselines.relieff_scores(A*w,y[tr][ii],random_state=101) if rank=='relief' else w)
            path=protected_majority_path(A,y[tr][ii],np.argsort(relevance,kind='stable').tolist(),w,allowed_enemies=allowed_enemies)
            candidates.update(tuple(fs) for fs in path['candidates'])
            paths.append({'family':family,'rank':rank,**path})
        for method in traces:
            if method=='protected':subsets=sorted(candidates)
            elif method=='all':subsets=[tuple(range(d))]
            elif method=='sfrip':subsets=[tuple(select_sfrip(A*w,y[tr][ii],beta=b,confidence=c)['features']) for b in [2.,3.,5.] for c in [.6,.8]]
            else:
                function={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}[method]
                subsets=[tuple(function(A*w,y[tr][ii],k=max(1,int(np.ceil(f*d))))) for f in [.2,.35,.5,.65,.8,1.]]
            for fs in sorted(set(subsets)):
                if not fs:continue
                val=validation_scores(A,y[tr][ii],B,y[tr][jj],list(fs),w,144,joint=True)
                traces[method].append({'family':family,'features':list(fs),'validation':val,
                    'score':(val['clean']+val['noise01'])/2})
    rows=[]
    for method,trace in traces.items():
        chosen=max(trace,key=lambda r:(r['score'],-len(r['features']),r['family']=='uniform'))
        w=final_weights[chosen['family']];fs=chosen['features'].copy();restoration=[]
        workspace=MajorityTerms(F,y[tr],weights=w)
        full_bounds=workspace.bounds(range(d),allowed_enemies=allowed_enemies);protected=full_bounds>1e-10
        groups=[(y[tr]==label)&protected for label in np.unique(y)]
        required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
        order=np.argsort(-w,kind='stable').tolist()
        while True:
            final_bounds=workspace.bounds(fs,allowed_enemies=allowed_enemies)
            counts=[int(np.sum((final_bounds>1e-10)&g)) for g in groups]
            if method!='protected' or all(c>=r for c,r in zip(counts,required)):break
            a=next(a for a in order if a not in fs);fs=sorted(fs+[a]);restoration.append(a)
        metric=validation_scores(F,y[tr],Q,y[te],fs,w,104,joint=True)
        rows.append({'method':method,'chosen':chosen,'trace':trace,'features':fs,
            'weights':w.tolist(),'restoration':restoration,'metrics':metric,
            'certificate':{'protected_counts':[int(g.sum()) for g in groups],
                'retained_counts':counts,'required':required,
                'lower_bounds':np.nan_to_num(final_bounds,posinf=1e100,neginf=-1e100).tolist(),
                'allowed_enemies':allowed_enemies,
                'scope':'finite_training_true_label_majority_retention_not_dependency_or_test_risk'}})
        print(json.dumps({'dataset':name,'fold':fold,'method':method,'family':chosen['family'],
            'k':len(fs),**metric}),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),
        'test_indices':te.tolist(),'discovery_indices':ii.tolist(),'validation_indices':jj.tolist(),
        'rows':rows,'paths':paths,'source_hashes':hashes,
        'split_policy':'stratified_70_30_with_training_singletons_kept_in_discovery',
        'status':'development_not_confirmatory'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--iteration',type=int,choices=[24,25],default=24)
    p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere'])
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(name,fold,a.iteration)) for name in a.datasets for fold in a.folds]
        for future in as_completed(futures):future.result()
