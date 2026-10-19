"""Independent final fuzzy score refits and both reference certificates."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from run_fuzzy_teacher_grouped_certified import train,score

FOLDER=ROOT/'results/fuzzy-teacher-grouped-certified'

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'candidate_refits':0,'fuzzy_certificates':0,'selected_compact':0,'predictions':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        source=json.loads((ROOT/'results/common-fuzzy-classifier'/f'{name}.json').read_text())
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            assert s['C']==source['splits'][i]['results']['full']['C']
            A,B=preprocess_fit(X[ri],X[vi]);classes=np.array(s['teacher']['classes']);W,b=np.array(s['teacher']['coef']),np.array(s['teacher']['intercept'])
            original,_=manual_margins(A,y[ri],W,b,classes);protected=original>1e-10
            fullfit=train(A,y[ri],list(range(p['d'])),s['C'])
            np.testing.assert_allclose(fullfit['coef'],W,atol=1e-12);np.testing.assert_allclose(fullfit['intercept'],b,atol=1e-12)
            feasible=[]
            for t in s['trials']:
                c=t['candidate'];fs=c['features'];v=np.array(c['weights']);mass=np.sqrt(p['d'])
                assert abs(v.sum()-mass)<1e-8 and v.max()<=2/mass+1e-8 and np.all(v>=0)
                cert=weight_certificate(A,y[ri],fs,np.ones(p['d'])/mass,v[fs])
                assert cert['feasible']==c['fuzzy_certificate']['feasible'];counts['fuzzy_certificates']+=1
                fit=train(A,y[ri],fs,s['C']);np.testing.assert_allclose(fit['coef'],t['fit']['coef'],atol=1e-12)
                margin,_=manual_margins(A,y[ri],np.array(fit['coef']),np.array(fit['intercept']),classes)
                np.testing.assert_allclose(margin,t['certificate']['margins'],atol=1e-10)
                accepted=bool(cert['feasible'] and np.all(margin[protected]>=.5*original[protected]-1e-8) and np.all(margin[protected]>1e-10))
                assert accepted==t['certificate']['feasible']
                if accepted:feasible.append(t)
                counts['candidate_refits']+=1
            assert s['fallback']==(not bool(feasible))
            if feasible:
                assert min(feasible,key=lambda t:(len(t['candidate']['features']),t['candidate']['reconstruction_error']))==s['selected'];counts['selected_compact']+=1
            else:assert s['selected']['fit']==s['teacher']
            rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            for fit,key in ((s['selected']['fit'],'metrics'),(s['teacher'],'full_metrics')):
                values=[score(Q,y[vi],fit) for Q in queries]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[s[key]['clean'],s[key]['noise']],atol=1e-12)
                counts['predictions']+=4
            counts['splits']+=1
        print(name,'dual certificates and fuzzy refits verified',flush=True)
    assert counts['splits']==44
    counts['scope']='full_classifier_refit_independent_scalar_margins_fuzzy_certificates_prediction_replay_not_support_optimizer_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
