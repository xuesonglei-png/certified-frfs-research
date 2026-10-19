import hashlib,json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from diagnose_robust_linear_refit import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.linear_retention import box_class_margins

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'successful_refits':0,'fuzzy_certificates':0,'predictions':0,'fallbacks':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());old=json.loads((ROOT/'results/linear-retention'/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        for index,s in enumerate(p['splits']):
            teacher=old['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==teacher['reference_indices'] and vi.tolist()==teacher['validation_indices'] and set(ri).isdisjoint(vi)
            A,B=preprocess_fit(X[ri],X[vi]);labels=y[ri];classes=np.array(teacher['teacher_classes']);original=box_class_margins(A,labels,np.array(teacher['teacher_coef']),np.array(teacher['teacher_intercept']),classes)
            protected=original>1e-10
            for trial in s['trials']:
                c=trial['candidate'];fs=c['features'];w=np.array(c['weights']);d=X.shape[1];mass=np.sqrt(d)
                assert abs(w.sum()-mass)<1e-8 and w.max()<=2*mass/d+1e-8 and np.all(w>=0)
                assert weight_certificate(A,labels,fs,np.ones(d)/mass,w[fs])['feasible'];counts['fuzzy_certificates']+=1
                r=trial['fit']
                if r['feasible']:
                    W,b=np.array(r['coef']),np.array(r['intercept']);omitted=np.setdiff1d(np.arange(d),fs);assert np.all(W[:,omitted]==0)
                    margin=box_class_margins(A,labels,W,b,classes)
                    np.testing.assert_allclose(margin,r['deployed_margins'],atol=1e-12)
                    assert np.all(margin[protected]>=.5*original[protected]-1e-8) and np.all(margin[protected]>1e-10)
                    counts['successful_refits']+=1
            feasible=[r for r in s['trials'] if r['fit']['feasible']]
            assert s['fallback']==(not bool(feasible))
            if feasible:assert min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective']))==s['selected']
            else:
                counts['fallbacks']+=1
                np.testing.assert_array_equal(s['selected']['fit']['coef'],teacher['teacher_coef'])
                np.testing.assert_array_equal(s['selected']['fit']['intercept'],teacher['teacher_intercept'])
            r=s['selected']['fit'];W,b=np.array(r['coef']),np.array(r['intercept']);rng=np.random.default_rng(144+index)
            queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
            values=[balanced_accuracy_score(y[vi],classes[(Q@W.T+b).argmax(axis=1)]) for Q in queries]
            np.testing.assert_allclose([values[0],np.mean(values[1:])],[s['metrics']['clean'],s['metrics']['query_noise01']],atol=1e-12)
            counts['splits']+=1;counts['predictions']+=4
        print(name,'deployed margins and predictions verified',flush=True)
    assert counts['splits']==44
    counts['scope']='actual_deployed_margins_fuzzy_certificates_prediction_replay_not_LP_optimality_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
