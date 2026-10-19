"""Reconstruct new margin candidates and verify unchanged comparator records."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_majority_metric_iteration import weights
from review_majority_metric_iteration import score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.crossfit_majority import recipe_candidates
from certified_frfs.majority_metric import MajorityTerms


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration27';rows=[];matched=new_scores=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());old=json.loads((ROOT/'results/predictive-iteration26'/path.name).read_text())
        assert obj['train_indices']==old['train_indices'] and obj['test_indices']==old['test_indices']
        assert obj['inner_splits']==old['inner_splits']
        X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        row=next(r for r in obj['rows'] if r['method']=='protected')
        previous=next(r for r in old['rows'] if r['method']=='protected')
        # Previous recipe traces are unchanged; verify only the added rank here.
        old_trace=[r for r in row['trace'] if r['config']['rank']!='margin']
        assert old_trace==previous['trace']
        for other in obj['rows']:
            if other['method']=='protected':continue
            assert other==next(r for r in old['rows'] if r['method']==other['method']);matched+=1
        for i,split in enumerate(obj['inner_splits']):
            ii,jj=np.array(split['train']),np.array(split['validation'])
            A,B=preprocess_fit(X[tr][ii],X[tr][jj])
            for family in ('uniform','triplet','majority'):
                w=weights(A,y[tr][ii],family)
                candidates,_=recipe_candidates(A,y[tr][ii],w,'protected',include_margin=True)
                for candidate in row['trace']:
                    if candidate['family']!=family or candidate['config']['rank']!='margin':continue
                    expected=next(r['features'] for r in candidates if r['config']==candidate['config'])
                    fold=candidate['folds'][i];assert expected==fold['features']
                    assert score(A,y[tr][ii],B,y[tr][jj],expected,w,144+i)==fold['validation'];new_scores+=1
        for candidate in row['trace']:
            assert abs(candidate['score']-np.mean([f['score'] for f in candidate['folds']]))<1e-12
        chosen=max(row['trace'],key=lambda r:(r['score'],-r['mean_size'],r['family']=='uniform'))
        assert chosen==row['chosen']
        F,Q=preprocess_fit(X[tr],X[te]);w=weights(F,y[tr],chosen['family'])
        candidates,_=recipe_candidates(F,y[tr],w,'protected',include_margin=True)
        fs=next(r['features'] for r in candidates if r['config']==chosen['config'])
        assert fs==row['features'] and np.allclose(w,row['weights'])
        assert score(F,y[tr],Q,y[te],fs,w,104)==row['metrics']
        workspace=MajorityTerms(F,y[tr],weights=w);full=workspace.bounds(range(obj['d']),allowed_enemies=2)
        bounds=workspace.bounds(fs,allowed_enemies=2)
        for label in np.unique(y):
            group=(y[tr]==label)&(full>1e-10)
            assert np.sum(group&(bounds>1e-10))>=np.ceil(.9*group.sum()-1e-12)
        rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'rank':chosen['config']['rank'],
            'clean':row['metrics']['clean'],'old_clean':previous['metrics']['clean'],
            'removed':1-len(fs)/obj['d'],'old_removed':1-len(previous['features'])/obj['d']})
        print(obj['dataset'],'verified',flush=True)
    output={'rows':rows,'matched_baselines':matched,'verified_new_inner_candidates':new_scores,
        'scope':'five_dataset_fold0_development_incremental_verification_not_significance'}
    (directory/'pilot-review.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
