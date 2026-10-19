"""Full iteration31 outer/retention audit; does not rebuild inner selectors."""
import argparse,json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_crossfit_majority_iteration import inner_splits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.flexible_witness import flexible_bounds
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main(iteration=31,expected_count=45):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';paths=sorted(directory.glob('*-s101-f*.json'))
    assert len(paths)==expected_count;grouped={};outer=certs=0;coverage=[]
    for path in paths:
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        a,b=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[obj['fold']]
        assert np.array_equal(a,tr) and np.array_equal(b,te)
        splits=inner_splits(y[tr]);assert len(splits)==len(obj['inner_splits'])
        for (ii,jj),record in zip(splits,obj['inner_splits']):
            assert ii.tolist()==record['train'] and jj.tolist()==record['validation']
        A,B=preprocess_fit(X[tr],X[te]);d=obj['d']
        previous={}
        if iteration==32:
            old=json.loads((ROOT/'results/predictive-iteration31'/path.name).read_text())
            assert old['train_indices']==obj['train_indices'] and old['test_indices']==obj['test_indices']
            previous={r['method']:r for r in old['rows']}
        for row in obj['rows']:
            if iteration==32 and row['method']!='protected':
                p=previous[row['method']]
                assert row['features']==p['features'] and row['weights']==p['weights']
                assert row['chosen']==p['chosen'] and row['metrics']==p['metrics']
            for c in row['trace']:
                assert len(c['folds'])==len(splits)
                assert abs(c['score']-np.mean([f['score'] for f in c['folds']]))<1e-12
                assert abs(c['mean_size']-np.mean([len(f['features']) for f in c['folds']]))<1e-12
            assert row['chosen']==max(row['trace'],key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
            fs=row['features'];w=np.array(row['weights']);assert len(fs)==len(set(fs)) and all(0<=a<d for a in fs)
            assert score(A,y[tr],B,y[te],fs,w,104)==row['metrics'];outer+=1
            if row['method']=='protected':
                ws=MajorityTerms(A,y[tr],weights=w);protected=ws.bounds(range(d),allowed_enemies=2)>1e-10
                bounds=np.maximum(ws.bounds(fs,allowed_enemies=2),adaptive_bounds(A,y[tr],fs,w))
                if iteration==32:bounds=np.maximum(bounds,flexible_bounds(A,y[tr],fs,w,pool_size=8))
                groups=[(y[tr]==label)&protected for label in np.unique(y[tr])]
                counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
                required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups];cert=row['certificate']
                assert counts==cert['retained_counts'] and required==cert['required'] and all(c>=r for c,r in zip(counts,required))
                assert cert['protected_counts']==[int(g.sum()) for g in groups]
                np.testing.assert_allclose(np.nan_to_num(bounds,posinf=1e100,neginf=-1e100),cert['lower_bounds'],rtol=0,atol=1e-12)
                coverage.append({'dataset':obj['dataset'],'fold':obj['fold'],'full_subset':len(fs)==d,
                    'protected_fraction':float(protected.mean()),'zero_protected_classes':sum(not g.any() for g in groups)})
                certs+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                **row['metrics'],'removed':1-len(fs)/d})
        print('outer verified',obj['dataset'],obj['fold'],flush=True)
    assert len(grouped['protected'])==expected_count//3 and all(len(v)==3 for data in grouped.values() for v in data.values())
    result={'status':'development_outer_and_certificates_verified_inner_selector_rebuild_not_claimed',
        'verified_outer_scores':outer,'verified_certificates':certs,'coverage':coverage,
        'all15':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli'])}
    (directory/'full-verified-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'outer':outer,'certificates':certs,'means':{m:r['means'] for m,r in result['all15']['methods'].items()},
        'pairs':result['all15']['pairs']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[31,32],default=31)
    p.add_argument('--expected-count',type=int,default=45);a=p.parse_args()
    main(a.iteration,a.expected_count)
