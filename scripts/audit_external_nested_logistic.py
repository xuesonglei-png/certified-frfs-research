"""Replay selection, predictions, split provenance and every saved certificate."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_external_nested_logistic import ROOT,FOLDER,SOURCES,METHODS,GRIDS,SELECTORS,dataset,folds,score
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate


def check_fit(A,y,fitted):
    d=A.shape[1];mass=np.sqrt(d)
    for key in ('selected','trajectory_selected'):
        c=fitted[key];fs=c['features'];w=np.array(c['weights'])
        assert fs==np.flatnonzero(w>0).tolist()
        assert np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-mass)<1e-8 and w.max()<=2*mass/d+1e-8
        cert=weight_certificate(A,y,fs,np.ones(d)/mass,w[fs]);assert cert['feasible']
        assert cert['required']==c['certificate']['required'] and cert['retained_counts']==c['certificate']['retained_counts']
        np.testing.assert_allclose(cert['dependency_certificate']['upper_bound'],c['certificate']['dependency_certificate']['upper_bound'],atol=1e-12)


def main():
    threadpool_limits(1);complete=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'certificates':0,'inner_scores':0,'external_scores':0}
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y,groups=dataset(name)
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
        assert p['groups']==(None if groups is None else groups.tolist())
        for index,((tr,te),s) in enumerate(zip(folds(X,y,groups,outer=True),p['splits'])):
            assert tr.tolist()==s['train_indices'] and te.tolist()==s['test_indices'] and set(tr).isdisjoint(te)
            raw,labels=X[tr],y[tr];innergroups=None if groups is None else groups[tr]
            innerfolds=folds(raw,labels,innergroups,outer=False);assert len(innerfolds)==len(s['inner_records'])
            for j,((ti,qi),inner) in enumerate(zip(innerfolds,s['inner_records'])):
                assert ti.tolist()==inner['reference'] and qi.tolist()==inner['query']
                A,B=preprocess_fit(raw[ti],raw[qi]);f=inner['fit'];check_fit(A,labels[ti],f);counts['certificates']+=2
                reps={'proposed':f['selected']['features'],'trajectory':f['trajectory_selected']['features'],'full':list(range(X.shape[1]))};cache={}
                for m,choices in s['trace'].items():
                    for c in choices:
                        if m in GRIDS:
                            key=(m,json.dumps(c['config'],sort_keys=True))
                            if key not in cache:cache[key]=SELECTORS[m](A,labels[ti],**c['config'])['features']
                            fs=cache[key]
                        else:fs=reps[m]
                        assert fs==c['folds'][j]['features']
                        np.testing.assert_allclose(score(A,labels[ti],B,labels[qi],fs,c['C']),c['folds'][j]['clean'],atol=1e-12)
                        counts['inner_scores']+=1
            A,B=preprocess_fit(raw,X[te]);check_fit(A,labels,s['fit']);counts['certificates']+=2
            reps={'proposed':s['fit']['selected']['features'],'trajectory':s['fit']['trajectory_selected']['features'],'full':list(range(X.shape[1]))}
            rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            for m,choices in s['trace'].items():
                for c in choices:
                    np.testing.assert_allclose(c['mean_clean'],np.mean([f['clean'] for f in c['folds']]),atol=1e-12)
                    np.testing.assert_allclose(c['mean_size'],np.mean([len(f['features']) for f in c['folds']]),atol=1e-12)
                chosen=max(range(len(choices)),key=lambda j:(choices[j]['mean_clean'],-choices[j]['mean_size'],-j));r=s['results'][m];assert chosen==r['chosen_index'];c=choices[chosen]
                fs=SELECTORS[m](A,labels,**c['config'])['features'] if m in GRIDS else reps[m];assert fs==r['features']
                values=[score(R,labels,Q,y[te],fs,c['C']) for R,Q in versions]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[r['metrics']['clean'],r['metrics']['noise01']],atol=1e-12)
                counts['external_scores']+=4
            counts['splits']+=1
        print(name,'group/split provenance, nested selection and certificates verified',flush=True)
    assert counts['splits']==24
    counts['scope']='all_saved_certificates_and_nested_predictions_replay_not_full_metric_optimizer_replay_or_test_risk_proof'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts,flush=True)


if __name__=='__main__':main()
