"""Reconstruct iteration24 metric fits, holdout scores and majority retention."""
import argparse,json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_majority_metric_iteration import weights,discovery_split
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms


def score(A,y,B,truth,fs,w,seed):
    model=KNeighborsClassifier(n_neighbors=min(5,len(y)),metric='manhattan',algorithm='brute')
    model.fit(A[:,fs]*w[fs],y)
    clean=float(balanced_accuracy_score(truth,model.predict(B[:,fs]*w[fs])))
    rng=np.random.default_rng(seed);noisy=[]
    for _ in range(3):
        ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        query=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        model.fit(ref[:,fs]*w[fs],y)
        noisy.append(balanced_accuracy_score(truth,model.predict(query[:,fs]*w[fs])))
    return {'clean':clean,'noise01':float(np.mean(noisy))}


def main(iteration=24):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}'
    grouped={};outer=inner=certs=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        ii,jj=np.array(obj['discovery_indices']),np.array(obj['validation_indices'])
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(y)))
        assert not set(ii)&set(jj) and sorted(np.r_[ii,jj])==list(range(len(tr)))
        expected_ii,expected_jj=discovery_split(y[tr])
        assert np.array_equal(ii,expected_ii) and np.array_equal(jj,expected_jj)
        A,B=preprocess_fit(X[tr][ii],X[tr][jj]);F,Q=preprocess_fit(X[tr],X[te])
        IW={family:weights(A,y[tr][ii],family) for family in ['uniform','triplet','majority']}
        FW={family:weights(F,y[tr],family) for family in IW}
        workspaces={family:MajorityTerms(F,y[tr],weights=w) for family,w in FW.items()}
        for row in obj['rows']:
            for candidate in row['trace']:
                result=score(A,y[tr][ii],B,y[tr][jj],candidate['features'],IW[candidate['family']],144)
                assert result==candidate['validation']
                assert abs(candidate['score']-(result['clean']+result['noise01'])/2)<1e-12
                inner+=1
            chosen=max(row['trace'],key=lambda r:(r['score'],-len(r['features']),r['family']=='uniform'))
            assert chosen==row['chosen']
            w=FW[chosen['family']];fs=row['features'];workspace=workspaces[chosen['family']]
            assert np.allclose(w,row['weights'],rtol=0,atol=1e-12)
            assert score(F,y[tr],Q,y[te],fs,w,104)==row['metrics'];outer+=1
            allowed=row['certificate'].get('allowed_enemies',0)
            bounds=workspace.bounds(fs,allowed_enemies=allowed);full=workspace.bounds(range(obj['d']),allowed_enemies=allowed)
            protected=full>1e-10;groups=[(y[tr]==label)&protected for label in np.unique(y)]
            counts=[int(np.sum((bounds>1e-10)&g)) for g in groups]
            required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            assert np.allclose(np.nan_to_num(bounds,posinf=1e100,neginf=-1e100),row['certificate']['lower_bounds'])
            assert counts==row['certificate']['retained_counts']
            assert required==row['certificate']['required']
            if row['method']=='protected':assert all(c>=r for c,r in zip(counts,required))
            certs+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                **row['metrics'],'removed':1-len(fs)/obj['d'],
                'protected_fraction':float(protected.mean()),
                'retained_fraction':float(np.mean((bounds>1e-10)&protected))})
    summary={}
    for method,data in grouped.items():
        means={name:{key:float(np.mean([r[key] for r in rows])) for key in rows[0]} for name,rows in data.items()}
        summary[method]={'datasets':means,'means':{key:float(np.mean([r[key] for r in means.values()])) for key in next(iter(means.values()))}}
    pairs=[]
    for method in summary:
        if method=='protected':continue
        names=sorted(summary[method]['datasets'])
        delta=np.array([summary['protected']['datasets'][n]['clean']-summary[method]['datasets'][n]['clean'] for n in names])
        pairs.append({'comparator':method,'mean_delta':float(delta.mean()),
            'wins_ties_losses':[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],
            'raw_p':float(wilcoxon(delta).pvalue) if np.any(abs(delta)>1e-12) else 1.})
    maximum=0.
    for rank,index in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]['raw_p'])):
        maximum=max(maximum,min(1.,(len(pairs)-rank)*pairs[index]['raw_p']));pairs[index]['holm_p']=maximum
    result={'status':'development_only_unequal_candidate_budgets','verified_outer_scores':outer,
        'verified_holdout_scores':inner,'verified_training_certificates':certs,'methods':summary,'pairwise':pairs}
    (directory/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({**{k:v for k,v in result.items() if k!='methods'},'means':{k:v['means'] for k,v in summary.items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[24,25],default=24)
    main(p.parse_args().iteration)
