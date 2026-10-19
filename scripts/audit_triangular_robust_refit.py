"""Independent scalar coupled-margin and FRS deployment/prediction replay."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_triangular_robust_refit import ROOT,FOLDER
from audit_triangular_classifier_diagnostic import manual_margins
from run_fuzzy_teacher_grouped_certified import score
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'fuzzy_certificates':0,'successful_LPs':0,'failed_LPs':0,'predictions':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            A,B=preprocess_fit(X[ri],X[vi]);t=s['teacher'];classes=np.array(t['classes']);W,b=np.array(t['coef']),np.array(t['intercept']);original,_=manual_margins(A,y[ri],W,b,classes);protected=original>1e-10
            for trial in s['trials']:
                c=trial['candidate'];fs=c['features'];v=np.array(c['weights']);mass=np.sqrt(p['d'])
                assert abs(v.sum()-mass)<1e-8 and v.max()<=2/mass+1e-8 and np.all(v>=0)
                assert weight_certificate(A,y[ri],fs,np.ones(p['d'])/mass,v[fs])['feasible'];counts['fuzzy_certificates']+=1
                fit=trial['fit']
                if not fit['feasible']:counts['failed_LPs']+=1;continue
                margins,_=manual_margins(A,y[ri],np.array(fit['coef']),np.array(fit['intercept']),classes)
                np.testing.assert_allclose(margins,fit['deployed_margins'],atol=1e-10)
                assert np.all(margins[protected]>=.5*original[protected]-1e-8) and np.all(margins[protected]>1e-10)
                assert np.all(np.array(fit['coef'])[:,np.setdiff1d(np.arange(p['d']),fs)]==0)
                assert np.all(np.array(fit['coef'])[0]==0) and fit['intercept'][0]==0
                counts['successful_LPs']+=1
            feasible=[r for r in s['trials'] if r['fit']['feasible']];assert s['fallback']==(not bool(feasible))
            if feasible:assert min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective']))==s['selected']
            else:np.testing.assert_array_equal(s['selected']['fit']['coef'],W)
            full=next(r for r in s['trials'] if len(r['candidate']['features'])==p['d']);assert s['full_fallback']==(not full['fit']['feasible'])
            if s['full_fallback']:assert s['full_fit']==t
            else:assert s['full_fit']==full['fit']
            rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            for fit,key in ((s['selected']['fit'],'metrics'),(s['full_fit'],'full_metrics')):
                values=[score(Q,y[vi],fit) for Q in queries]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[s[key]['clean'],s[key]['noise']],atol=1e-12);counts['predictions']+=4
            counts['splits']+=1
        print(name,'exact scalar fuzzy margins/FRS/predictions audited',flush=True)
    assert counts['splits']==44
    counts['scope']='scalar_raw_breakpoint_fuzzy_and_deployed_margins_selection_prediction_replay_not_LP_optimizer_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
