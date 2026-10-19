"""Prediction/selection replay; does not replay proposed inner optimizers."""
import hashlib
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from compare_nested_classifier_pipelines import ROOT,FOLDER,KINDS,CS,GRIDS,SELECTORS,predict,representations
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);complete=json.loads((FOLDER/'complete.json').read_text())
    counts={'splits':0,'inner_predictions':0,'external_predictions':0,'inner_certificates':0}
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
        for index,s in enumerate(p['splits']):
            old=prior['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];d=X.shape[1];mass=np.sqrt(d)
            for j,inner in enumerate(s['inner_records']):
                ti,qi=np.array(inner['reference']),np.array(inner['query'])
                assert set(ti).isdisjoint(qi) and len(set(ti).union(qi))==len(ri)
                A,B=preprocess_fit(raw[ti],raw[qi]);reps=inner['representations']
                for m in ('proposed_weighted','dense_weighted'):
                    fs=reps[m]['features'];w=np.array(reps[m]['weights']);native=w*mass/w.sum()
                    assert len(fs)==len(set(fs)) and np.all(native>0) and native.max()<=2*mass/d+1e-8
                    cert=weight_certificate(A,labels[ti],fs,np.ones(d)/mass,native)
                    assert cert['feasible'];counts['inner_certificates']+=1
                cache={}
                for kind in KINDS:
                    for m,choices in s['trace'][kind].items():
                        assert len(choices)==len(CS)*(len(GRIDS[m]) if m in GRIDS else 1)
                        for c in choices:
                            if m in GRIDS:
                                key=(m,json.dumps(c['config'],sort_keys=True))
                                if key not in cache:cache[key]=SELECTORS[m](A,labels[ti],**c['config'])['features']
                                fs=cache[key];w=np.ones(len(fs))
                            else:fs=reps[m]['features'];w=np.array(reps[m]['weights'])
                            assert fs==c['folds'][j]['features']
                            value=balanced_accuracy_score(labels[qi],predict(kind,c['C'],A,labels[ti],B,fs,w))
                            np.testing.assert_allclose(value,c['folds'][j]['clean'],atol=1e-12,rtol=1e-12)
                            counts['inner_predictions']+=1
            A,B=preprocess_fit(raw,X[vi]);reps=representations(old['fit']['compact_selected'],old['certified_trajectory']['selected'],d)
            rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            for kind in KINDS:
                for m,choices in s['trace'][kind].items():
                    for c in choices:
                        np.testing.assert_allclose(c['mean_clean'],np.mean([f['clean'] for f in c['folds']]),atol=1e-12)
                        np.testing.assert_allclose(c['mean_size'],np.mean([len(f['features']) for f in c['folds']]),atol=1e-12)
                    chosen=max(range(len(choices)),key=lambda j:(choices[j]['mean_clean'],-choices[j]['mean_size'],-j))
                    result=s['results'][kind][m];assert result['chosen_index']==chosen;c=choices[chosen]
                    if m in GRIDS:fs=SELECTORS[m](A,labels,**c['config'])['features'];w=np.ones(len(fs))
                    else:fs,w=reps[m]
                    assert fs==result['features'];np.testing.assert_allclose(w,result['weights'],atol=1e-12)
                    values=[float(balanced_accuracy_score(y[vi],predict(kind,c['C'],R,labels,Q,fs,w))) for R,Q in versions]
                    np.testing.assert_allclose([values[0],np.mean(values[1:])],[result['metrics']['clean'],result['metrics']['noise01']],atol=1e-12,rtol=1e-12)
                    counts['external_predictions']+=4
            counts['splits']+=1
        print(name,'nested predictions, configs and inner certificates replayed',flush=True)
    assert counts['splits']==44
    counts['scope']='nested_prediction_and_selected_inner_certificate_replay_not_proposed_optimizer_replay_or_test_risk_proof'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
