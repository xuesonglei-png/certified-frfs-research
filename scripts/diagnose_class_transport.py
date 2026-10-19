"""Fixed-subset training subfold class-aware transport gate, no outer scores."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace
from certified_frfs.class_transport import fit_class_transport


def main():
    threadpool_limits(1);records=json.loads((ROOT/'results/redistributed-metric-diagnostic/fold0-training.json').read_text())['rows'];rows=[]
    for r in records:
        o=json.loads((ROOT/'results/predictive-iteration36'/f"{r['dataset']}-s101-f0.json").read_text())
        family=next(p for p in o['rows'] if p['method']=='protected')['chosen']['family']
        X,y=load_dataset(r['dataset']);tr=np.array(o['train_indices']);fs=r['features']
        old=CachedCrossfitTransferUtility(X[tr],y[tr],family=family).evaluate(fs)
        source=CrossfitRedistributedUtilityWorkspace(X[tr],y[tr],family=family);scores=[];fits=[]
        for index,(A,B,yt,yv,w) in enumerate(source.folds):
            v,info=fit_class_transport(A,yt,fs,w);rng=np.random.default_rng(144+index)
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            labels=np.unique(yt);values=[]
            for ref,query in [(A,B),(Z,Q)]:
                D=abs(query[:,None,fs]-ref[None,:,fs])@v
                nn=np.argsort(D,axis=1,kind='stable')[:,:min(5,len(yt))]
                votes=np.stack([(yt[nn]==q).sum(axis=1) for q in labels],axis=1);pred=labels[votes.argmax(axis=1)]
                values.append(float(np.mean([(pred[yv==q]==q).mean() for q in np.unique(yv)])))
            scores.append(values);fits.append({'weights':v.tolist(),'fit':info})
        clean,noisy=np.mean(scores,axis=0);utility=float((clean+noisy)/2)
        row={'dataset':r['dataset'],'features':fs,'family':family,'old_score':old,
            'new_score':{'clean':float(clean),'noisy':float(noisy),'utility':utility},
            'utility_delta':utility-old['utility'],'fits':fits,'splits':source.splits}
        rows.append(row);print(r['dataset'],'utility delta',row['utility_delta'],flush=True)
    gates={'mean_gain_at_least_half_point':bool(np.mean([r['utility_delta'] for r in rows])>=.005),
           'at_least_three_positive':sum(r['utility_delta']>1e-12 for r in rows)>=3}
    directory=ROOT/'results/class-transport-diagnostic';directory.mkdir(exist_ok=True);out=directory/'training-subfold-gate.json';assert not out.exists()
    out.write_text(json.dumps({'scope':'reused_training_subfold_fixed_subset_not_test_superiority',
        'rows':rows,'gates':gates,'gate_passed':all(gates.values()),
        'mean_utility_delta':float(np.mean([r['utility_delta'] for r in rows]))},indent=2),encoding='utf-8')
    print(json.dumps(gates))

if __name__=='__main__':main()
