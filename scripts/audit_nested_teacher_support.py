"""Independent nested stage, singleton separation, margin and score audit."""
import hashlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.datasets import load_dataset
from certified_frfs.linear_retention import linear_parameters,box_class_margins
from certified_frfs.feasible_transport import weight_certificate

FOLDER=ROOT/'results/nested-teacher-support'

def independent_score(query,truth,classes,fit):
    pred=classes[(query@np.array(fit['coef']).T+np.array(fit['intercept'])).argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

def audit_stage(raw,y,query,truth,stage,scores,counts,*,seed=None):
    A,B=preprocess_fit(raw,query)
    model=LogisticRegression(C=stage.get('teacher_C',1),max_iter=2000,random_state=0).fit(A,y)
    W,b,classes=linear_parameters(model)
    np.testing.assert_allclose(W,stage['teacher_coef'],atol=1e-12)
    np.testing.assert_allclose(b,stage['teacher_intercept'],atol=1e-12)
    np.testing.assert_array_equal(classes,stage['teacher_classes'])
    original=box_class_margins(A,y,W,b,classes);protected=original>1e-10
    mass=np.sqrt(A.shape[1]);full=np.ones(A.shape[1])/mass;seen=set()
    queries=[B]
    if seed is not None:
        rng=np.random.default_rng(seed);queries += [np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    for key,r in stage['policies'].items():
        fs=r['candidate']['features'];v=np.array(r['candidate']['weights']);fit=r['fit']
        identity=(tuple(fs),json.dumps(fit['coef']),json.dumps(fit['intercept']))
        if identity not in seen:
            assert np.all(v>=0) and abs(v.sum()-mass)<1e-8 and v.max()<=2/mass+1e-8
            cert=weight_certificate(A,y,fs,full,v[fs]);assert cert['feasible']
            coef=np.array(fit['coef']);intercept=np.array(fit['intercept'])
            assert np.all(coef[:,np.setdiff1d(np.arange(A.shape[1]),fs)]==0)
            margins=box_class_margins(A,y,coef,intercept,classes)
            assert np.all(margins[protected]>=.5*original[protected]-1e-8) and np.all(margins[protected]>1e-10)
            if 'deployed_margins' in fit:np.testing.assert_allclose(margins,fit['deployed_margins'],atol=1e-12)
            seen.add(identity);counts['unique_models']+=1
        values=[independent_score(Q,truth,classes,fit) for Q in queries]
        np.testing.assert_allclose(values[0],scores[key]['clean'],atol=1e-12)
        assert scores[key]['retained']==len(fs)/A.shape[1]
        if seed is not None:np.testing.assert_allclose(np.mean(values[1:]),scores[key]['query_noise01'],atol=1e-12)
        counts['predictions']+=len(queries)
        if r['fallback']:
            assert fs==list(range(A.shape[1])) and r['failure'] is not None
            counts['fallback_policies']+=1
    counts['stages']+=1

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text())
    counts={'outer_splits':0,'inner_splits':0,'stages':0,'unique_models':0,'predictions':0,'fallback_policies':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];query_ids=[]
            classes,classcounts=np.unique(labels,return_counts=True);single=np.flatnonzero(np.isin(labels,classes[classcounts==1]))
            for inner in s['inner']:
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices'])
                assert set(a).isdisjoint(b) and set(a)|set(b)==set(range(len(ri)))
                assert set(single).issubset(a) and not set(single)&set(b)
                assert set(ri[a]).isdisjoint(vi) and set(ri[b]).isdisjoint(vi)
                audit_stage(raw[a],labels[a],raw[b],labels[b],inner['stage'],inner['policies'],counts)
                query_ids.extend(b.tolist());counts['inner_splits']+=1
            assert sorted(query_ids)==sorted(set(range(len(ri)))-set(single))
            keys=['0.5','0.65','0.8','1.0']
            scores={k:float(np.mean([r['policies'][k]['clean'] for r in s['inner']])) for k in keys}
            retained={k:float(np.mean([r['policies'][k]['retained'] for r in s['inner']])) for k in keys}
            eligible=[k for k in keys if scores[k]>=max(scores.values())-.005-1e-12]
            chosen=min(eligible,key=lambda k:(retained[k],-scores[k],float(k)));assert chosen==s['chosen']
            assert s['selection']['scores']==scores and s['selection']['retained']==retained and s['selection']['eligible']==eligible
            # The outer record stores only selected and full metrics; recompute others independently.
            _,B=preprocess_fit(raw,X[vi]);rng=np.random.default_rng(144+i)
            queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)];outer={}
            for k,r in s['stage']['policies'].items():
                values=[independent_score(Q,y[vi],np.array(s['stage']['teacher_classes']),r['fit']) for Q in queries]
                outer[k]={'clean':values[0],'query_noise01':float(np.mean(values[1:])),'retained':len(r['candidate']['features'])/p['d']}
            for k,v in ((chosen,s['metrics']),('1.0',s['full_metrics'])):
                for field in v:np.testing.assert_allclose(outer[k][field],v[field],atol=1e-12)
            audit_stage(raw,labels,X[vi],y[vi],s['stage'],outer,counts,seed=144+i)
            counts['outer_splits']+=1
        print(name,'nested reference stages and selection audited',flush=True)
    assert counts['outer_splits']==44
    counts['scope']='full_teacher_refit_certificate_prediction_selection_replay_not_LP_optimality_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
