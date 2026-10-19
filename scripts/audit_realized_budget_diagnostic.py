import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from audit_common_fold_recent import independent_fit,verify_trace
from run_common_fold_recent import ROOT,CS,score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/realized-budget-diagnostic'

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    counts={'splits':0,'inner_scores':0,'outer_models':0,'exact_count_matches':0,'prediction_replays':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        common=json.loads((ROOT/'results/common-fold-recent'/f'{name}.json').read_text())
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for index,s in enumerate(p['splits']):
            source=common['splits'][index];assert s['reference_indices']==source['reference_indices'] and s['validation_indices']==source['validation_indices']
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri]
            assert s['target_size']==len(source['proposed_features']) and set(ri).isdisjoint(vi)
            for j,inner in enumerate(source['inner']):
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(raw[a],raw[b]);cache={}
                for method,choices in s['trace'].items():
                    for k,c in enumerate(choices):
                        fold=c['folds'][j];fs=fold['features'];assert len(fs)==fold['target_size']
                        assert fs==inner['paths'][method][k//len(CS)]['ranking'][:fold['target_size']]
                        identity=(tuple(fs),c['C'])
                        if identity not in cache:cache[identity]=score(B,labels[b],independent_fit(A,labels[a],fs,c['C']))
                        np.testing.assert_allclose(cache[identity],fold['clean'],atol=1e-12);counts['inner_scores']+=1
            A,B=preprocess_fit(raw,X[vi])
            for method,choices in s['trace'].items():
                selected=verify_trace(choices);r=s['results'][method];assert r['chosen_index']==selected
                assert len(r['features'])==s['target_size'] and r['features']==r['path']['ranking'][:s['target_size']]
                assert r['C']==choices[selected]['C'] and r['config']==choices[selected]['config']
                fit=independent_fit(A,labels,r['features'],r['C'])
                np.testing.assert_allclose(fit['coef'],r['fit']['coef'],atol=1e-12)
                np.testing.assert_allclose(fit['intercept'],r['fit']['intercept'],atol=1e-12)
                rng=np.random.default_rng(144+index);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
                values=[score(Q,y[vi],fit) for Q in queries]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[r['metrics']['clean'],r['metrics']['query_noise01']],atol=1e-12)
                counts['outer_models']+=1;counts['prediction_replays']+=4;counts['exact_count_matches']+=1
            counts['splits']+=1
        print(name,'exact cardinalities and classifier replays audited',flush=True)
    assert counts['splits']==44 and counts['outer_models']==88 and counts['exact_count_matches']==88
    counts['scope']='exact_count_and_classifier_score_selection_replay_not_full_selector_rerun_or_nested_reselection'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
