"""Recompute all fixed-model predictions from archived outer indices."""
import hashlib
import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from diagnose_capped_classifier_transfer import ROOT,FOLDER,MODELS,model
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def main():
    threadpool_limits(1); counts={'splits':0,'fit_predict_calls':0}
    complete=json.loads((FOLDER/'complete.json').read_text())
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
        prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
        base=json.loads((ROOT/'results/trajectory-recent-comparison'/f'{name}.json').read_text())
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert set(ri).isdisjoint(vi)
            old,b=prior['splits'][index],base['splits'][index]
            assert ri.tolist()==old['reference_indices']==b['reference_indices']
            assert vi.tolist()==old['validation_indices']==b['validation_indices']
            c=old['fit']['compact_selected'];fs=c['features'];native=np.array(c['weights'])[fs]
            weights={'proposed_uniform':np.ones(len(fs)),'proposed_weighted':native/native.mean(),
                     'full':np.ones(X.shape[1])}
            assert s['features']['proposed_uniform']==s['features']['proposed_weighted']==fs
            assert s['features']['full']==list(range(X.shape[1]))
            for m in ('fgs2023','sfrip2025'):
                ids=b['baselines'][m]['selection']['features']; assert s['features'][m]==ids
                weights[m]=np.ones(len(ids))
            A,B=preprocess_fit(X[ri],X[vi]);rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):
                versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            for kind in MODELS:
                for m,ids in s['features'].items():
                    values=[]
                    for ref,query in versions:
                        estimator=model(kind,len(ri)) if ids else DummyClassifier(strategy='prior')
                        R,Q=(ref[:,ids]*weights[m],query[:,ids]*weights[m]) if ids else (np.zeros((len(ref),1)),np.zeros((len(query),1)))
                        values.append(float(balanced_accuracy_score(y[vi],estimator.fit(R,y[ri]).predict(Q))))
                        counts['fit_predict_calls']+=1
                    np.testing.assert_allclose([values[0],np.mean(values[1:])],
                        [s['scores'][kind][m]['clean'],s['scores'][kind][m]['noise01']],atol=1e-12,rtol=1e-12)
            counts['splits']+=1
        print(name,'classifier transfer predictions replayed',flush=True)
    assert counts['splits']==44 and counts['fit_predict_calls']==2640
    counts['scope']='fixed_support_full_prediction_replay_not_classifier_specific_retuning_or_test_accuracy_certificate'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
