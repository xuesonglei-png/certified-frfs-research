"""Reconstruct nested feature fits, independent scores, and certificates."""
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_crossfit_majority_iteration import inner_splits
from run_nested_union_iteration import fit_candidates
from run_softvote_iteration import metric
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.crossfit_majority import recipe_candidates
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.flexible_witness import flexible_bounds
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main(iteration,folds=None):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}'
    grouped={};inner_count=outer_count=cert_count=0;coverage=[]
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        if folds is not None and obj['fold'] not in folds:continue
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices']);d=obj['d']
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(y)))
        weights={};fits={};splits=inner_splits(y[tr])
        assert len(splits)==len(obj['inner_splits'])
        for index,((ii,jj),record) in enumerate(zip(splits,obj['inner_splits'])):
            assert ii.tolist()==record['train'] and jj.tolist()==record['validation']
            A,B=preprocess_fit(X[tr][ii],X[tr][jj])
            if iteration==29:
                fitted,nested=fit_candidates(X[tr][ii],y[tr][ii]);assert nested==record['aggregation_splits']
                fits[index]=fitted;weights[index]={'uniform':np.ones(d)/np.sqrt(d)}
            else:
                weights[index]={f:metric(A,y[tr][ii],f) for f in ('uniform','softvote')}
                fits[index]={}
                for family,w in weights[index].items():
                    np.testing.assert_allclose(w,record['weights'][family],rtol=0,atol=1e-12)
                    fits[index][family]={m:recipe_candidates(A,y[tr][ii],w,m,refresh_witness=iteration in (31,32),flexible_witness=iteration==32)[0] for m in grouped_methods(obj)}
        F,Q=preprocess_fit(X[tr],X[te]);FW={}
        if iteration==29:
            final,nested=fit_candidates(X[tr],y[tr]);assert nested==obj['final_aggregation_splits']
        score_cache={}
        for row in obj['rows']:
            method=row['method']
            for c in row['trace']:
                family=c.get('family','uniform');scores=[];sizes=[]
                assert len(c['folds'])==len(splits)
                for f in c['folds']:
                    index=f['inner'];ii,jj=splits[index];A,B=preprocess_fit(X[tr][ii],X[tr][jj])
                    candidates=fits[index][method] if iteration==29 else fits[index][family][method]
                    expected=next(r for r in candidates if r['config']==c['config'])
                    assert f['features']==expected['features']
                    if iteration==29:assert f['fold_subsets']==expected['fold_subsets']
                    cache_key=(index,family,tuple(f['features']))
                    if cache_key not in score_cache:
                        score_cache[cache_key]=score(A,y[tr][ii],B,y[tr][jj],f['features'],weights[index][family],144+index)
                    metrics=score_cache[cache_key]
                    assert metrics==f['validation'];scores.append((metrics['clean']+metrics['noise01'])/2)
                    assert abs(scores[-1]-f['score'])<1e-12
                    sizes.append(len(f['features']));inner_count+=1
                assert abs(np.mean(scores)-c['score'])<1e-12 and abs(np.mean(sizes)-c['mean_size'])<1e-12
            chosen=max(row['trace'],key=lambda c:(c['score'],-c['mean_size'],c.get('family','uniform')=='uniform'))
            assert chosen==row['chosen'];family=chosen.get('family','uniform')
            if family not in FW:FW[family]=metric(F,y[tr],family)
            w=FW[family];np.testing.assert_allclose(w,row['weights'],rtol=0,atol=1e-12)
            candidates=final[method] if iteration==29 else recipe_candidates(F,y[tr],w,method,refresh_witness=iteration in (31,32),flexible_witness=iteration==32)[0]
            expected=next(c for c in candidates if c['config']==chosen['config'])
            assert row['features']==expected['features']
            assert score(F,y[tr],Q,y[te],row['features'],w,104)==row['metrics'];outer_count+=1
            if method=='protected':
                ws=MajorityTerms(F,y[tr],weights=w);protected=ws.bounds(range(d),allowed_enemies=2)>1e-10
                bounds=ws.bounds(row['features'],allowed_enemies=2)
                if iteration in (31,32):bounds=np.maximum(bounds,adaptive_bounds(F,y[tr],row['features'],w))
                if iteration==32:bounds=np.maximum(bounds,flexible_bounds(F,y[tr],row['features'],w,pool_size=8))
                groups=[(y[tr]==label)&protected for label in np.unique(y[tr])]
                counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
                required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups];cert=row['certificate']
                assert counts==cert['retained_counts'] and required==cert['required']
                assert cert['protected_counts']==[int(g.sum()) for g in groups]
                assert all(c>=r for c,r in zip(counts,required))
                np.testing.assert_allclose(np.nan_to_num(bounds,posinf=1e100,neginf=-1e100),cert['lower_bounds'],rtol=0,atol=1e-12)
                coverage.append({'dataset':obj['dataset'],'fold':obj['fold'],'protected_fraction':float(protected.mean()),
                    'zero_protected_classes':sum(not g.any() for g in groups),'full_features':len(row['features'])==d})
                cert_count+=1
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({
                **row['metrics'],'removed':1-len(row['features'])/d})
        print('verified',iteration,obj['dataset'],obj['fold'],flush=True)
    result={'iteration':iteration,'status':'reused_development_tests_not_confirmatory',
        'verified_inner_scores':inner_count,'verified_outer_scores':outer_count,'verified_certificates':cert_count,
        'coverage':coverage,'analysis':analyze(grouped,[])}
    filename='verified-analysis.json' if folds is None else 'verified-folds-'+ '-'.join(map(str,folds))+'.json'
    (directory/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'iteration':iteration,'inner':inner_count,'outer':outer_count,'certificates':cert_count,
        'means':{m:r['means'] for m,r in result['analysis']['methods'].items()},'pairs':result['analysis']['pairs']},indent=2))


def grouped_methods(obj):return [r['method'] for r in obj['rows']]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[29,30,31,32],required=True)
    p.add_argument('--folds',type=int,nargs='+');a=p.parse_args()
    main(a.iteration,a.folds)
