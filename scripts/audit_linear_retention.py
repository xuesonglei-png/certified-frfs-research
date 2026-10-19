import hashlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from diagnose_linear_retention import ROOT,FOLDER,score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.linear_retention import linear_parameters,linear_retention_certificate,projected_parameters

def main():
    threadpool_limits(1);counts={'splits':0,'linear_certificates':0,'fuzzy_certificates':0,'predictions':0,'empty_protected_splits':0}
    complete=json.loads((FOLDER/'complete.json').read_text())
    for row in complete['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        old=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text())
        nested=json.loads((ROOT/'results/nested-classifier-pipelines'/f'{name}.json').read_text())
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            assert ri.tolist()==old['splits'][index]['reference_indices'] and vi.tolist()==old['splits'][index]['validation_indices']
            reference=nested['splits'][index];chosen=reference['results']['logistic']['full']['chosen_index']
            choices=reference['trace']['logistic']['full'];assert chosen==max(range(len(choices)),key=lambda i:(choices[i]['mean_clean'],-choices[i]['mean_size'],-i))
            C=choices[chosen]['C'];assert C==s['teacher_C']
            A,B=preprocess_fit(X[ri],X[vi]);labels=y[ri];model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A,labels)
            W,b,classes=linear_parameters(model);center=A.mean(axis=0)
            np.testing.assert_allclose(W,s['teacher_coef'],atol=1e-12);np.testing.assert_allclose(b,s['teacher_intercept'],atol=1e-12)
            np.testing.assert_allclose(center,s['center'],atol=1e-12);np.testing.assert_array_equal(classes,s['teacher_classes'])
            expected=[c for c in old['splits'][index]['fit']['candidates'] if c['certificate']['feasible']]
            assert [r['candidate'] for r in s['candidates']]==expected
            for record in s['candidates']:
                c=record['candidate'];fs=c['features'];w=np.array(c['weights']);d=X.shape[1];mass=np.sqrt(d)
                assert fs==np.flatnonzero(w>0).tolist() and abs(w.sum()-mass)<1e-8 and w.max()<=2*mass/d+1e-8
                assert weight_certificate(A,labels,fs,np.ones(d)/mass,w[fs])['feasible'];counts['fuzzy_certificates']+=1
                cert=linear_retention_certificate(A,labels,W,b,classes,fs,center)
                assert cert==record['linear_certificate'];counts['linear_certificates']+=1
            feasible=[r for r in s['candidates'] if r['linear_certificate']['feasible']]
            selected=min(feasible,key=lambda r:(len(r['candidate']['features']),r['candidate']['objective'] if r['candidate']['objective'] is not None else 1e9))
            c=selected['candidate'];assert c==s['selected']
            counts['empty_protected_splits']+=selected['linear_certificate']['protected_count']==0
            Rw,Rb=projected_parameters(W,b,c['features'],center);rng=np.random.default_rng(144+index);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            projected=[float(balanced_accuracy_score(y[vi],classes[(Q@Rw.T+Rb).argmax(axis=1)])) for _,Q in versions]
            refit=[score(R,labels,Q,y[vi],c['features'],C) for R,Q in versions]
            for key,values in (('metrics',projected),('refit_metrics',refit)):
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[s[key]['clean'],s[key]['noise01']],atol=1e-12)
            counts['predictions']+=8;counts['splits']+=1
        print(name,'fixed teacher, both certificates and scores replayed',flush=True)
    assert counts['splits']==44
    counts['scope']='fixed_teacher_candidate_and_score_replay_not_historical_generator_replay_or_population_risk'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts,flush=True)

if __name__=='__main__':main()
