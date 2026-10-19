import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_classifier_aware_pool import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.classifier_aware_pool import _score
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);complete=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'inner_scores':0,'external_scores':0}
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            old=prior['splits'][index];assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices']
            raw,labels=X[ri],y[ri];cs=[c for c in old['fit']['candidates'] if c['certificate']['feasible']]
            folds=QueryProbabilityObjective(raw,labels,seed=1701).splits
            tables=[(np.array(f['reference']),np.array(f['query']),*preprocess_fit(raw[np.array(f['reference'])],raw[np.array(f['query'])])) for f in folds]
            for t in s['trace']:
                c=cs[t['candidate_index']];values=[_score(A,labels[ti],B,labels[qi],c['features'],t['C']) for ti,qi,A,B in tables]
                np.testing.assert_allclose(values,t['folds'],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(np.mean(values),t['mean_clean'],atol=1e-12)
                assert t['size']==len(c['features']);counts['inner_scores']+=len(values)
            chosen=max(range(len(s['trace'])),key=lambda j:(s['trace'][j]['mean_clean'],-s['trace'][j]['size'],-j));assert chosen==s['chosen_index']
            t=s['trace'][chosen];c=cs[t['candidate_index']];assert c==s['selected']
            A,B=preprocess_fit(raw,X[vi]);fs=c['features'];w=np.array(c['weights']);d=X.shape[1];mass=np.sqrt(d)
            assert np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-mass)<1e-8 and w.max()<=2*mass/d+1e-8
            cert=weight_certificate(A,labels,fs,np.ones(d)/mass,w[fs]);assert cert['feasible'] and cert==s['certificate']
            rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            values=[_score(R,labels,Q,y[vi],fs,t['C']) for R,Q in versions]
            np.testing.assert_allclose([values[0],np.mean(values[1:])],[s['metrics']['clean'],s['metrics']['noise01']],atol=1e-12)
            counts['external_scores']+=4;counts['splits']+=1
        print(name,'adaptive pool fully replayed',flush=True)
    assert counts['splits']==44
    counts['scope']='archived_pool_selection_and_certificate_replay_not_candidate_generator_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts,flush=True)


if __name__=='__main__':main()
