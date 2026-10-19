"""Recompute every held-out score, summarize paired development outcomes."""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_decision_iteration import ROOT, scaled_split, scores
from certified_frfs.datasets import load_dataset
from threadpoolctl import threadpool_limits


def main(iteration=15):
    threadpool_limits(1)
    directory=ROOT/f'results/predictive-iteration{iteration}'
    files=sorted(directory.glob('*-s*-f*.json'))
    grouped={}
    audited=0
    baseline_checks=0
    for path in files:
        obj=json.loads(path.read_text())
        X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        assert not set(tr)&set(te)
        assert sorted(np.r_[tr,te].tolist())==list(range(len(y)))
        A,B=scaled_split(X[tr],X[te])
        old_path=ROOT/'results/predictive-iteration14'/path.name
        old=json.loads(old_path.read_text()) if old_path.exists() else None
        if old:
            assert old['train_indices']==tr.tolist() and old['test_indices']==te.tolist()
        for row in obj['rows']:
            fs=row['features']
            assert fs and len(set(fs))==len(fs) and min(fs)>=0 and max(fs)<obj['d']
            actual=scores(A,y[tr],B,y[te],fs,obj['seed']+3,k=5)
            for eps,metrics in actual.items():
                for key,value in metrics.items():
                    assert np.isclose(value,row['metrics'][eps][key],rtol=0,atol=1e-12)
            audited+=1
            if iteration==16 and row['method'] in ['graph_robust_majority','graph_nominal_majority']:
                from certified_frfs.core import certificate
                cert=certificate(A,y[tr],fs,eps=.01,method='hybrid')
                assert cert['upper_bound']<=.1+1e-10
                assert cert==row['detail']['loss_certificate']
            if old and row['method'] in ['all','sfrip','mi','relief','mrmr']:
                previous=next(r for r in old['rows'] if r['method']==row['method'])
                assert fs==previous['features']
                assert actual==previous['metrics']
                baseline_checks+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                'clean':actual['0.0']['balanced_accuracy'],
                'noise01':actual['0.01']['balanced_accuracy'],
                'noise05':actual['0.05']['balanced_accuracy'],
                'macro_f1':actual['0.0']['macro_f1'],
                'removed':1-len(fs)/obj['d']})
    summary={'status':'development_only','completed_splits':len(files),
             'independently_recomputed_method_scores':audited,
             'identical_historical_baseline_checks':baseline_checks,'methods':{},'pairwise':[]}
    for method,datasets in grouped.items():
        means={name:{key:float(np.mean([v[key] for v in values])) for key in values[0]}
               for name,values in datasets.items()}
        summary['methods'][method]={'datasets':means,'means':{
            key:float(np.mean([v[key] for v in means.values()])) for key in next(iter(means.values()))}}
    target='graph_robust_majority'
    for comparator in grouped:
        if comparator==target:
            continue
        common=sorted(set(grouped[target])&set(grouped[comparator]))
        delta=np.array([summary['methods'][target]['datasets'][n]['clean']-
                        summary['methods'][comparator]['datasets'][n]['clean'] for n in common])
        summary['pairwise'].append({'comparator':comparator,'datasets':common,
            'mean_delta':float(delta.mean()),'wins_ties_losses':[int((delta>1e-12).sum()),
            int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],
            'raw_p':float(wilcoxon(delta).pvalue) if np.any(delta) else 1.})
    comparisons=summary['pairwise']
    order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['raw_p'])
    maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,min(1.,(len(order)-rank)*comparisons[i]['raw_p']))
        comparisons[i]['holm_p']=maximum
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iteration',type=int,choices=[15,16],default=15)
    main(parser.parse_args().iteration)
