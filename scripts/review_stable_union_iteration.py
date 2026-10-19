"""Independent reconstruction and dataset-level analysis of iteration28."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration28'
    grouped={};verified=certificates=0
    paths=sorted(directory.glob('*-s101-f*.json'));assert len(paths)==45
    for path in paths:
        obj=json.loads(path.read_text());source=ROOT/'results/predictive-iteration26'/path.name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==obj['input_sha256']
        old=json.loads(source.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        assert obj['train_indices']==old['train_indices'] and obj['test_indices']==old['test_indices']
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(y)))
        A,B=preprocess_fit(X[tr],X[te])
        for row,previous in zip(obj['rows'],old['rows']):
            assert row['method']==previous['method'] and row['chosen']==previous['chosen']
            assert row['weights']==previous['weights']
            union=sorted(set(a for fold in previous['chosen']['folds'] for a in fold['features']))
            assert union==row['union_features'];w=np.array(row['weights']);fs=union[:]
            if row['method']=='protected':
                workspace=MajorityTerms(A,y[tr],weights=w)
                protected=workspace.bounds(range(obj['d']),allowed_enemies=2)>1e-10
                groups=[(y[tr]==label)&protected for label in np.unique(y[tr])]
                required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups];restored=[]
                order=np.argsort(-w,kind='stable').tolist()
                while True:
                    bounds=workspace.bounds(fs,allowed_enemies=2)
                    counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
                    if all(c>=r for c,r in zip(counts,required)):break
                    a=next(a for a in order if a not in fs);fs=sorted(fs+[a]);restored.append(a)
                cert=row['certificate']
                assert cert['restoration']==restored and cert['required']==required
                assert cert['retained_counts']==counts and cert['protected_counts']==[int(g.sum()) for g in groups]
                np.testing.assert_allclose(np.nan_to_num(bounds,posinf=1e100,neginf=-1e100),cert['lower_bounds'],rtol=0,atol=1e-12)
                certificates+=1
            assert fs==row['features']
            assert score(A,y[tr],B,y[te],fs,w,104)==row['metrics'];verified+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({
                **row['metrics'],'removed':1-len(fs)/obj['d']})
    assert all(len(rows)==3 for data in grouped.values() for rows in data.values())
    result={'status':'development_union_rule_not_directly_inner_validated',
        'verified_outer_scores':verified,'verified_certificates':certificates,
        'all15':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli'])}
    (directory/'verified-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'verified':verified,'certificates':certificates,
        'means':{k:v['means'] for k,v in result['all15']['methods'].items()},
        'pairs':result['all15']['pairs']},indent=2))

if __name__=='__main__':main()
