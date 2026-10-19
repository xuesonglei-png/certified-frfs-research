import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_margin_candidate_pool import ROOT,FOLDER,score
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate

def main():
    threadpool_limits(1);complete=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'certificates':0,'inner_scores':0,'external_scores':0}
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];A,B=preprocess_fit(raw,X[vi]);d=X.shape[1];mass=np.sqrt(d)
            for c in s['candidates']:
                fs=c['features'];w=np.array(c['weights']);assert set(fs)==set(np.flatnonzero(w>0))
                assert np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-mass)<1e-8 and w.max()<=2*mass/d+1e-8
                cert=weight_certificate(A,labels,fs,np.ones(d)/mass,w[fs]);assert cert['feasible'];counts['certificates']+=1
            tables=[]
            for f in QueryProbabilityObjective(raw,labels,seed=1701).splits:
                ti,qi=np.array(f['reference']),np.array(f['query']);tables.append((ti,qi,*preprocess_fit(raw[ti],raw[qi])))
            for t in s['trace']:
                c=s['candidates'][t['candidate_index']];vals=[score(Ai,labels[ti],Bi,labels[qi],c['features'],t['C']) for ti,qi,Ai,Bi in tables]
                np.testing.assert_allclose(vals,t['folds'],atol=1e-12);np.testing.assert_allclose(np.mean(vals),t['mean_clean'],atol=1e-12)
                counts['inner_scores']+=len(vals)
            chosen=max(range(len(s['trace'])),key=lambda j:(s['trace'][j]['mean_clean'],-s['trace'][j]['size'],-j));assert chosen==s['chosen_index']
            t=s['trace'][chosen];c=s['candidates'][t['candidate_index']];assert c==s['selected']
            rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            vals=[score(R,labels,Q,y[vi],c['features'],t['C']) for R,Q in versions]
            np.testing.assert_allclose([vals[0],np.mean(vals[1:])],[s['metrics']['clean'],s['metrics']['noise01']],atol=1e-12)
            counts['splits']+=1;counts['external_scores']+=4
        print(name,'pool constraints and scores replayed',flush=True)
    assert counts['splits']==44
    counts['scope']='candidate_certificates_and_scores_replay_not_margin_generator_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
