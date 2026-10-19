"""Independent score/weight/certificate reconstruction for iteration18."""
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT,scaled_split
from certified_frfs.datasets import load_dataset
from certified_frfs.triplet_metric import learn_triplet_weights
from certified_frfs.core import certificate


def main(iteration=18):
    threadpool_limits(1)
    directory=ROOT/f'results/predictive-iteration{iteration}'
    grouped={}; count=0; certs=0; norm_checks=0
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(X)))
        A,B=scaled_split(X[tr],X[te]);w=learn_triplet_weights(A,y[tr],seed=obj['seed'])
        assert np.isclose(np.linalg.norm(w),1) and np.all(w>0)
        for row in obj['rows']:
            assert np.allclose(w,row['weights'],rtol=0,atol=1e-12)
            fs=row['features'];model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute').fit(A[:,fs]*w[fs],y[tr])
            rng=np.random.default_rng(obj['seed']+3)
            for eps in [0.,.01,.05]:
                versions=[B] if eps==0 else [np.clip(B+rng.uniform(-eps,eps,B.shape),0,1) for _ in range(3)]
                values=[]
                for Z in versions:
                    pred=model.predict(Z[:,fs]*w[fs]);values.append([balanced_accuracy_score(y[te],pred),f1_score(y[te],pred,average='macro',zero_division=0)])
                for value,key in zip(np.mean(values,axis=0),['balanced_accuracy','macro_f1']):
                    assert np.isclose(value,row['metrics'][str(eps)][key],rtol=0,atol=1e-12)
            if row['method'].startswith('metric_graph'):
                actual=certificate(A,y[tr],fs,weights=w,eps=.01)
                assert actual==row['detail']['loss_certificate'] and actual['upper_bound']<=.1+1e-10
                certs+=1
            count+=1;norm_checks+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                'clean':row['metrics']['0.0']['balanced_accuracy'],
                'noise05':row['metrics']['0.05']['balanced_accuracy'],
                'removed':1-len(fs)/obj['d']})
    summary={'status':'development_only','verified_model_scores':count,'verified_certificates':certs,'weight_checks':norm_checks,'methods':{},'pairwise':[]}
    for method,datasets in grouped.items():
        means={name:{key:float(np.mean([r[key] for r in rows])) for key in rows[0]} for name,rows in datasets.items()}
        summary['methods'][method]={'datasets':means,'means':{key:float(np.mean([r[key] for r in means.values()])) for key in next(iter(means.values()))}}
    for comparator in grouped:
        if comparator=='metric_graph_robust':continue
        common=sorted(grouped[comparator]);delta=np.array([summary['methods']['metric_graph_robust']['datasets'][n]['clean']-summary['methods'][comparator]['datasets'][n]['clean'] for n in common])
        summary['pairwise'].append({'comparator':comparator,'delta':float(delta.mean()),'wins_ties_losses':[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],'raw_p':float(wilcoxon(delta).pvalue) if np.any(abs(delta)>1e-12) else 1.})
    pairs=summary['pairwise'];order=sorted(range(len(pairs)),key=lambda i:pairs[i]['raw_p']);maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,min(1.,(len(order)-rank)*pairs[i]['raw_p']));pairs[i]['holm_p']=maximum
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'verified':count,'certificates':certs,'means':{m:r['means'] for m,r in summary['methods'].items()},'pairwise':pairs},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iteration',type=int,choices=[18,19],default=18)
    main(parser.parse_args().iteration)
