"""Reconstruct iteration26 recipes, selection, final scores and certificates."""
import argparse,json
import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_crossfit_majority_iteration import inner_splits
from run_majority_metric_iteration import weights
from review_majority_metric_iteration import score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.crossfit_majority import recipe_candidates
from certified_frfs.majority_metric import MajorityTerms


def main(datasets=None,iteration=26):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}'
    grouped={};outer=inner=generated=certs=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        if datasets is not None and obj['dataset'] not in datasets:continue
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        assert not set(tr)&set(te) and sorted(np.r_[tr,te])==list(range(len(y)))
        splits=inner_splits(y[tr]);cache={};recipe_cache={};score_cache={}
        F,Q=preprocess_fit(X[tr],X[te]);final_weights={};final_recipes={}
        for i,(ii,jj) in enumerate(splits):
            assert ii.tolist()==obj['inner_splits'][i]['train'] and jj.tolist()==obj['inner_splits'][i]['validation']
            assert not set(ii)&set(jj) and sorted(np.r_[ii,jj])==list(range(len(tr)))
            A,B=preprocess_fit(X[tr][ii],X[tr][jj])
            for family in ('uniform','triplet','majority'):
                cache[(i,family)]=(A,B,weights(A,y[tr][ii],family),ii,jj)
        for row in obj['rows']:
            eligible={};empty_records=[]
            for family in ('uniform','triplet','majority'):
                configs=None
                for i in range(len(splits)):
                    A,B,w,ii,jj=cache[(i,family)];rk=(i,family,row['method'])
                    if rk not in recipe_cache:recipe_cache[rk]=recipe_candidates(A,y[tr][ii],w,row['method'],include_margin=iteration==27)[0]
                    valid={json.dumps(r['config'],sort_keys=True) for r in recipe_cache[rk] if r['features']}
                    configs=valid if configs is None else configs&valid
                    empty_records.extend({'method':row['method'],'family':family,'config':r['config'],'inner':i}
                                         for r in recipe_cache[rk] if not r['features'])
                eligible[family]=configs
            assert {(r['family'],json.dumps(r['config'],sort_keys=True)) for r in row['trace']}=={
                (family,config) for family,configs in eligible.items() for config in configs}
            if obj.get('empty_inner_outputs') is not None:
                actual=[r for r in obj['empty_inner_outputs'] if r['method']==row['method']]
                assert sorted(map(lambda r:json.dumps(r,sort_keys=True),actual))==sorted(map(lambda r:json.dumps(r,sort_keys=True),empty_records))
            for candidate in row['trace']:
                family=candidate['family']
                for fold in candidate['folds']:
                    i=fold['inner'];A,B,w,ii,jj=cache[(i,family)]
                    rk=(i,family,row['method'])
                    if rk not in recipe_cache:recipe_cache[rk]=recipe_candidates(A,y[tr][ii],w,row['method'],include_margin=iteration==27)[0]
                    expected=next(r['features'] for r in recipe_cache[rk] if r['config']==candidate['config'])
                    assert expected==fold['features'];generated+=1
                    sk=(i,family,tuple(expected))
                    if sk not in score_cache:score_cache[sk]=score(A,y[tr][ii],B,y[tr][jj],expected,w,144+i)
                    result=score_cache[sk]
                    assert result==fold['validation']
                    assert abs(fold['score']-(result['clean']+result['noise01'])/2)<1e-12;inner+=1
                assert abs(candidate['score']-np.mean([f['score'] for f in candidate['folds']]))<1e-12
                assert abs(candidate['mean_size']-np.mean([len(f['features']) for f in candidate['folds']]))<1e-12
            chosen=max(row['trace'],key=lambda r:(r['score'],-r['mean_size'],r['family']=='uniform'))
            assert chosen==row['chosen']
            family=chosen['family']
            if family not in final_weights:final_weights[family]=weights(F,y[tr],family)
            w=final_weights[family]
            assert np.allclose(w,row['weights'],rtol=0,atol=1e-12)
            rk=(family,row['method'])
            if rk not in final_recipes:final_recipes[rk]=recipe_candidates(F,y[tr],w,row['method'],include_margin=iteration==27)[0]
            candidates=final_recipes[rk]
            fs=next(r['features'] for r in candidates if r['config']==chosen['config'])
            assert fs==row['features'] and row['restoration']==[]
            assert score(F,y[tr],Q,y[te],fs,w,104)==row['metrics'];outer+=1
            workspace=MajorityTerms(F,y[tr],weights=w);full=workspace.bounds(range(obj['d']),allowed_enemies=2)
            final=workspace.bounds(fs,allowed_enemies=2);protected=full>1e-10
            groups=[(y[tr]==label)&protected for label in np.unique(y)]
            required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
            counts=[int(np.sum((final>1e-10)&g)) for g in groups]
            assert required==row['certificate']['required'] and counts==row['certificate']['retained_counts']
            assert np.allclose(np.nan_to_num(final,posinf=1e100,neginf=-1e100),row['certificate']['lower_bounds'])
            if row['method']=='protected':assert all(c>=r for c,r in zip(counts,required))
            certs+=1
            grouped.setdefault(row['method'],{}).setdefault(obj['dataset'],[]).append({**row['metrics'],
                'removed':1-len(fs)/obj['d'],'protected_fraction':float(protected.mean())})
        print(json.dumps({'verified_dataset':obj['dataset'],'fold':obj['fold'],
            'outer_checks_so_far':outer,'inner_checks_so_far':inner}),flush=True)
    summary={}
    for method,data in grouped.items():
        datasets={n:{k:float(np.mean([r[k] for r in rs])) for k in rs[0]} for n,rs in data.items()}
        summary[method]={'datasets':datasets,'means':{k:float(np.mean([r[k] for r in datasets.values()])) for k in next(iter(datasets.values()))}}
    pairs=[]
    for method in summary:
        if method=='protected':continue
        delta=np.array([summary['protected']['datasets'][n]['clean']-r['clean'] for n,r in summary[method]['datasets'].items()])
        pairs.append({'comparator':method,'mean_delta':float(delta.mean()),
            'wins_ties_losses':[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta< -1e-12).sum())],
            'raw_p':float(wilcoxon(delta).pvalue) if np.any(abs(delta)>1e-12) else 1.})
    maximum=0.
    for rank,i in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]['raw_p'])):
        maximum=max(maximum,min(1.,(len(pairs)-rank)*pairs[i]['raw_p']));pairs[i]['holm_p']=maximum
    result={'status':'development_only','verified_outer_scores':outer,'verified_inner_scores':inner,
        'verified_inner_recipe_subsets':generated,'verified_final_certificates':certs,'methods':summary,'pairwise':pairs}
    filename='summary.json' if datasets is None else 'expansion-verification.json'
    (directory/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({**{k:v for k,v in result.items() if k!='methods'},'means':{m:r['means'] for m,r in summary.items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');p.add_argument('--iteration',type=int,choices=[26,27],default=26);a=p.parse_args()
    main(a.datasets,a.iteration)
