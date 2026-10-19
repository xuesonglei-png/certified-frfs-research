"""Replay every stored inner logistic score and selected outer prediction."""
import hashlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_common_fold_recent import ROOT,GRIDS,CS,MODES,features,score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters

FOLDER=ROOT/'results/common-fold-recent'

def independent_fit(A,y,fs,C):
    classes,counts=np.unique(y,return_counts=True)
    if fs:
        model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A[:,fs],y)
        w,b,classes=linear_parameters(model);W=np.zeros((len(classes),A.shape[1]));W[:,fs]=w
    else:W=np.zeros((len(classes),A.shape[1]));b=np.log(counts/counts.sum())
    return {'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist()}

def verify_trace(choices):
    for c in choices:
        np.testing.assert_allclose(c['mean_clean'],np.mean([f['clean'] for f in c['folds']]),atol=1e-12)
        assert c['mean_size']==np.mean([len(f['features']) for f in c['folds']])
    return max(range(len(choices)),key=lambda i:(choices[i]['mean_clean'],-choices[i]['mean_size'],-i))

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    counts={'splits':0,'inner_splits':0,'inner_scores':0,'unique_inner_fits':0,'outer_models':0,'prediction_replays':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        source=json.loads((ROOT/'results/nested-teacher-regularized'/f'{name}.json').read_text())
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri]
            for j,inner in enumerate(s['inner']):
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices'])
                assert a.tolist()==source['splits'][index]['inner'][j]['reference_indices']
                assert b.tolist()==source['splits'][index]['inner'][j]['validation_indices'] and set(a).isdisjoint(b)
                A,B=preprocess_fit(raw[a],raw[b]);cache={}
                for method,modes in s['trace'].items():
                    for mode,choices in modes.items():
                        for k,c in enumerate(choices):
                            fs=features(inner['paths'][method][k//len(CS)],mode,p['d']);stored=c['folds'][j]
                            assert fs==stored['features']
                            identity=(tuple(fs),c['C'])
                            if identity not in cache:cache[identity]=score(B,labels[b],independent_fit(A,labels[a],fs,c['C']))
                            np.testing.assert_allclose(cache[identity],stored['clean'],atol=1e-12);counts['inner_scores']+=1
                for choices in s['simple_trace'].values():
                    for c in choices:
                        f=c['folds'][j];identity=(tuple(f['features']),c['C'])
                        if identity not in cache:cache[identity]=score(B,labels[b],independent_fit(A,labels[a],f['features'],c['C']))
                        np.testing.assert_allclose(cache[identity],f['clean'],atol=1e-12);counts['inner_scores']+=1
                counts['unique_inner_fits']+=len(cache);counts['inner_splits']+=1
            A,B=preprocess_fit(raw,X[vi]);results=[]
            for method,modes in s['trace'].items():
                for mode,choices in modes.items():
                    selected=verify_trace(choices);r=s['results'][method][mode];assert r['chosen_index']==selected
                    c=choices[selected];assert r['C']==c['C'] and r['config']==c['config']
                    path=s['outer_paths'][method][json.dumps(c['config'],sort_keys=True)]
                    assert r['features']==features(path,mode,p['d']);results.append(r)
            for method,choices in s['simple_trace'].items():
                selected=verify_trace(choices);r=s['results'][method];assert r['chosen_index']==selected and r['C']==choices[selected]['C'];results.append(r)
            for r in results:
                fit=independent_fit(A,labels,r['features'],r['C'])
                np.testing.assert_allclose(fit['coef'],r['fit']['coef'],atol=1e-12);np.testing.assert_allclose(fit['intercept'],r['fit']['intercept'],atol=1e-12)
                rng=np.random.default_rng(144+index);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
                values=[score(Q,y[vi],fit) for Q in queries]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[r['metrics']['clean'],r['metrics']['query_noise01']],atol=1e-12)
                counts['outer_models']+=1;counts['prediction_replays']+=4
            counts['splits']+=1
        print(name,'all inner classifier scores and chosen outer models replayed',flush=True)
    assert counts['splits']==44 and counts['inner_splits']==128 and counts['outer_models']==440
    counts['scope']='all_inner_classifier_scores_selection_outer_refits_fingerprints_not_full_selector_rerun'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
