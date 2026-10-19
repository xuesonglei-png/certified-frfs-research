"""Reconstruct splits, holdout metrics, final weights, predictions and bounds."""
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_metric_graph_iteration import scores
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,metric_weights
from certified_frfs.core import certificate


def main(iteration=20):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}'
    grouped={};checks=0;validation_checks=0;bound_checks=0
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        if iteration==23:
            previous=json.loads((ROOT/'results/predictive-iteration22'/path.name).read_text())
            assert previous['train_indices']==obj['train_indices'] and previous['test_indices']==obj['test_indices']
            obj['rows']=obj['rows']+previous['rows']
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(y)))
        A,B=preprocess_fit(X[tr],X[te])
        for row in obj['rows']:
            detail=row['detail'];ii=np.array(detail['discovery_indices']);jj=np.array(detail['validation_indices'])
            assert not set(ii)&set(jj) and sorted(np.r_[ii,jj])==list(range(len(tr)))
            Ia,Ib=preprocess_fit(X[tr][ii],X[tr][jj])
            for candidate in detail['trace']:
                w=metric_weights(Ia,y[tr][ii],candidate['family'],101);fs=candidate['features']
                model=KNeighborsClassifier(n_neighbors=min(5,len(ii)),metric='manhattan',algorithm='brute').fit(Ia[:,fs]*w[fs],y[tr][ii])
                clean=balanced_accuracy_score(y[tr][jj],model.predict(Ib[:,fs]*w[fs]))
                rng=np.random.default_rng(144);noisy=[]
                for _ in range(3):
                    if detail.get('joint_validation',False):
                        ref=np.clip(Ia+rng.uniform(-.01,.01,Ia.shape),0,1)
                        model.fit(ref[:,fs]*w[fs],y[tr][ii])
                    Z=np.clip(Ib+rng.uniform(-.01,.01,Ib.shape),0,1)
                    noisy.append(balanced_accuracy_score(y[tr][jj],model.predict(Z[:,fs]*w[fs])))
                assert np.isclose(clean,candidate['validation']['clean'],atol=1e-12,rtol=0)
                assert np.isclose(np.mean(noisy),candidate['validation']['noise01'],atol=1e-12,rtol=0)
                validation_checks+=1
            candidates=detail['trace']
            for candidate in candidates:
                val=candidate['validation']
                expected=val['clean'] if row['mode']=='nominal' else (val['clean']+val['noise01'])/2
                assert np.isclose(candidate['score'],expected,rtol=0,atol=1e-12)
            if row['mode']=='compression':
                threshold=max(c['score'] for c in candidates)-.01-1e-12
                eligible=[c for c in candidates if c['score']>=threshold]
                expected=max(eligible,key=lambda c:(-len(c['features']),c['score'],c['family']=='uniform'))
            else:
                expected=max(candidates,key=lambda c:(c['score'],-len(c['features']),c['family']=='uniform'))
            assert expected==detail['chosen']
            w=metric_weights(A,y[tr],detail['family'],101)
            assert np.allclose(w,detail['weights'],rtol=0,atol=1e-12)
            fs=detail['features'];assert scores(A,y[tr],B,y[te],fs,104,w)==row['metrics']
            if row['method'] in ('pool','pool_ldfrs','pool_ldfrs_exact'):
                cert=certificate(A,y[tr],fs,weights=w,eps=.01,stability=detail.get('stability','interval'))
                assert cert==detail['certificate'] and cert['upper_bound']<=.1+1e-10
                assert sorted(set(detail['chosen']['features'])|set(detail['restoration']))==sorted(fs)
                bound_checks+=1
            key=row['method']+'_'+row['mode']
            grouped.setdefault(key,[]).append({'dataset':obj['dataset'],'fold':obj['fold'],
                'clean':row['metrics']['0.0']['balanced_accuracy'],'removed':1-len(fs)/obj['d'],
                'restored':len(detail['restoration']),'candidate_count':len(detail['trace'])})
            checks+=1
    # Average folds within each dataset before calculating macro dataset means.
    summary={}
    for key,rows in grouped.items():
        datasets={}
        for r in rows:datasets.setdefault(r['dataset'],[]).append(r)
        means={n:{k:float(np.mean([v[k] for v in values])) for k in ['clean','removed','restored','candidate_count']} for n,values in datasets.items()}
        summary[key]={'datasets':means,'means':{k:float(np.mean([v[k] for v in means.values()])) for k in ['clean','removed','restored','candidate_count']}}
    output={'status':'development_only','verified_outer_scores':checks,'verified_holdout_scores':validation_checks,
            'verified_final_certificates':bound_checks,'methods':summary}
    target={22:'pool_ldfrs_robust',23:'pool_ldfrs_exact_robust'}.get(iteration,'pool_robust')
    pairs=[]
    for comparator in summary:
        if comparator==target:continue
        common=sorted(set(summary[target]['datasets'])&set(summary[comparator]['datasets']))
        delta=np.array([summary[target]['datasets'][n]['clean']-
                        summary[comparator]['datasets'][n]['clean'] for n in common])
        pairs.append({'comparator':comparator,'datasets':common,'mean_delta':float(delta.mean()),
                      'wins_ties_losses':[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],
                      'raw_p':float(wilcoxon(delta).pvalue) if np.any(abs(delta)>1e-12) else 1.})
    order=sorted(range(len(pairs)),key=lambda i:pairs[i]['raw_p']);maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,min(1.,(len(order)-rank)*pairs[i]['raw_p']));pairs[i]['holm_p']=maximum
    output['pairwise']=pairs
    (directory/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({**{k:v for k,v in output.items() if k!='methods'},
                     'means':{k:v['means'] for k,v in summary.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--iteration',type=int,choices=[20,21,22,23],default=20)
    main(parser.parse_args().iteration)
