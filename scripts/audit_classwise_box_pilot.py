"""Replay classwise reference margins, fuzzy constraints and held-out scores."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_classwise_box_pilot import ROOT,FOLDER,evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.linear_retention import box_class_margins

def deployed_check(A,y,classes,W,b,fit,*,bounded):
    original=box_class_margins(A,y,W,b,classes);coef=np.array(fit['coef']);intercept=np.array(fit['intercept'])
    margins=box_class_margins(A,y,coef,intercept,classes);protected=original>1e-10
    retained=(margins>=.5*original-1e-8)&(margins>1e-10)&protected
    required=[int(np.ceil(.9*np.sum((y==c)&protected)-1e-12)) for c in classes]
    counts=[int(np.sum((y==c)&retained)) for c in classes]
    assert all(c>=r for c,r in zip(counts,required))
    if bounded:
        assert fit['retention']==.9 and fit['required']==required and fit['retained_counts']==counts
        np.testing.assert_allclose(margins,fit['deployed_margins'],atol=1e-12)
        assert abs(coef).max()<=fit['coefficient_bound']+1e-8
        assert abs(intercept).max()<=fit['intercept_bound']+1e-8
        assert np.all(coef[0]==0) and intercept[0]==0
    return margins

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    counts={'splits':0,'fuzzy_certificates':0,'successful_milps':0,'failed_milps':0,'prediction_replays':0,'full_fallbacks':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            A,B=preprocess_fit(X[ri],X[vi]);t=s['teacher'];classes=np.array(t['teacher_classes']);W,b=np.array(t['teacher_coef']),np.array(t['teacher_intercept']);mass=np.sqrt(p['d'])
            for trial in s['trials']:
                c=trial['candidate'];fs=c['features'];v=np.array(c['weights'])
                assert abs(v.sum()-mass)<1e-8 and np.all(v>=0) and v.max()<=2/mass+1e-8
                assert weight_certificate(A,y[ri],fs,np.ones(p['d'])/mass,v[fs])['feasible']
                counts['fuzzy_certificates']+=1
                fit=trial['fit']
                if fit['feasible']:
                    margins=deployed_check(A,y[ri],classes,W,b,fit,bounded=True)
                    assert np.all(np.array(fit['coef'])[:,np.setdiff1d(np.arange(p['d']),fs)]==0)
                    np.testing.assert_allclose(margins,trial['deployed_certificate']['deployed_margins'],atol=1e-12)
                    counts['successful_milps']+=1
                else:assert trial['deployed_certificate'] is None;counts['failed_milps']+=1
            for rejected in s['rejected']:
                # Rejection is recorded against the original weights in the source;
                # it is not a proved violation of another method's objectives.
                assert not rejected['certificate']['feasible']
            feasible=[r for r in s['trials'] if r['fit']['feasible']]
            assert s['fallback']==(not bool(feasible))
            if feasible:assert min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective']))==s['selected']
            else:np.testing.assert_array_equal(s['selected']['fit']['coef'],W)
            deployed_check(A,y[ri],classes,W,b,s['selected']['fit'],bounded=not s['fallback'])
            fulltrial=next(r for r in s['trials'] if len(r['candidate']['features'])==p['d'])
            assert s['full_fallback']==(not fulltrial['fit']['feasible'])
            if s['full_fallback']:
                np.testing.assert_array_equal(s['full_fit']['coef'],W);counts['full_fallbacks']+=1
            else:assert s['full_fit']==fulltrial['fit']
            deployed_check(A,y[ri],classes,W,b,s['full_fit'],bounded=not s['full_fallback'])
            rng=np.random.default_rng(144+index);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            for fit,key in ((s['selected']['fit'],'metrics'),(s['full_fit'],'full_metrics')):
                replay=evaluate(queries,y[vi],classes,fit)
                for field in replay:np.testing.assert_allclose(replay[field],s[key][field],atol=1e-12)
                counts['prediction_replays']+=4
            counts['splits']+=1
        print(name,'classwise deployed coverage and predictions audited',flush=True)
    assert counts['splits']==44
    counts['scope']='fuzzy_and_actual_classwise_reference_coverage_prediction_selection_fingerprint_replay_not_MILP_optimality'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
