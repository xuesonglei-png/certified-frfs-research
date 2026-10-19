"""Independent audit of joint inner C/support choice and matched-full C choice."""
import hashlib,json
from run_decision_iteration import ROOT
from sklearn.linear_model import LogisticRegression
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.feasible_transport import weight_certificate
from audit_triangular_classifier_diagnostic import manual_margins
from certified_frfs.triangular_classifier import triangular_memberships
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


FOLDER=ROOT/'results/nested-triangular'


def independent_score(query,truth,classes,fit):
    pred=classes[(np.einsum('ndk,qdk->nq',triangular_memberships(query),np.array(fit['coef']))+np.array(fit['intercept'])).argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

def audit_stage(raw,y,query,truth,stage,scores,counts,*,seed=None):
    A,B=preprocess_fit(raw,query)
    model=LogisticRegression(C=stage['teacher_C'],max_iter=2000,random_state=0).fit(triangular_memberships(A).reshape(len(A),-1),y)
    W,b,classes=linear_parameters(model);W=W.reshape(len(classes),A.shape[1],3)
    np.testing.assert_allclose(W,stage['teacher_coef'],atol=1e-12)
    np.testing.assert_allclose(b,stage['teacher_intercept'],atol=1e-12)
    original,_=manual_margins(A,y,W,b,classes);protected=original>1e-10
    mass=np.sqrt(A.shape[1]);seen=set();queries=[B]
    if seed is not None:
        rng=np.random.default_rng(seed);queries += [np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    for key,r in stage['policies'].items():
        c=r['candidate'];fs=c['features'];fit=r['fit'];v=np.array(c['weights'])
        identity=(tuple(fs),json.dumps(fit['coef']),json.dumps(fit['intercept']))
        if identity not in seen:
            assert abs(v.sum()-mass)<1e-8 and np.all(v>=0) and v.max()<=2/mass+1e-8
            assert weight_certificate(A,y,fs,np.ones(A.shape[1])/mass,v[fs])['feasible']
            coef=np.array(fit['coef']);intercept=np.array(fit['intercept'])
            assert np.all(coef[:,np.setdiff1d(np.arange(A.shape[1]),fs)]==0)
            margin,_=manual_margins(A,y,coef,intercept,classes)
            assert np.all(margin[protected]>=.5*original[protected]-1e-8) and np.all(margin[protected]>1e-10)
            if 'deployed_margins' in fit:np.testing.assert_allclose(margin,fit['deployed_margins'],atol=1e-10)
            seen.add(identity);counts['unique_models']+=1
        values=[independent_score(Q,truth,classes,fit) for Q in queries]
        np.testing.assert_allclose(values[0],scores[key]['clean'],atol=1e-12)
        assert scores[key]['retained']==len(fs)/A.shape[1]
        if seed is not None:np.testing.assert_allclose(np.mean(values[1:]),scores[key]['query_noise01'],atol=1e-12)
        counts['predictions']+=len(queries)
        if r['fallback']:
            assert len(fs)==A.shape[1] and r['failure'] is not None
            counts['fallback_policies']+=1
    counts['stages']+=1

def outer_scores(raw,query,truth,stage,seed):
    _,B=preprocess_fit(raw,query);rng=np.random.default_rng(seed)
    queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    out={}
    for k,r in stage['policies'].items():
        values=[independent_score(Q,truth,np.array(stage['teacher_classes']),r['fit']) for Q in queries]
        out[k]={'clean':values[0],'query_noise01':float(np.mean(values[1:])),'retained':len(r['candidate']['features'])/stage['d']}
    return out

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
                assert set(inner['by_C'])=={'0.1','1.0','10.0'}
                for C,r in inner['by_C'].items():
                    assert float(C)==r['stage']['teacher_C']
                    audit_stage(raw[a],labels[a],raw[b],labels[b],r['stage'],r['policies'],counts)
                query_ids.extend(b.tolist());counts['inner_splits']+=1
            assert sorted(query_ids)==sorted(set(range(len(ri)))-set(single))
            options=[]
            for C in (.1,1.,10.):
                for fraction in ('0.5','0.65','0.8','1.0'):
                    scores=[r['by_C'][str(C)]['policies'][fraction] for r in s['inner']]
                    options.append({'C':C,'fraction':fraction,'score':float(np.mean([r['clean'] for r in scores])),
                                    'retained':float(np.mean([r['retained'] for r in scores]))})
            full=min((r for r in options if r['fraction']=='1.0'),key=lambda r:(-r['score'],r['C']))
            family=[r for r in options if r['C']==full['C']]
            eligible=[r for r in family if r['score']>=max(z['score'] for z in family)-.005-1e-12]
            proposed=min(eligible,key=lambda r:(r['retained'],-r['score'],float(r['fraction'])))
            assert proposed==s['selected_config'] and full==s['full_config']
            assert s['selection']=={'options':options,'eligible':eligible,'tolerance':.005,'hierarchical':True}
            byC={}
            for C,stage in s['outer_stages'].items():
                assert float(C)==stage['teacher_C']
                scores=outer_scores(raw,X[vi],y[vi],stage,144+i)
                audit_stage(raw,labels,X[vi],y[vi],stage,scores,counts,seed=144+i);byC[C]=scores
            expected=[byC[str(proposed['C'])][proposed['fraction']],byC[str(full['C'])]['1.0'],byC[str(proposed['C'])]['1.0']]
            for scores,stored in zip(expected,[s['metrics'],s['full_metrics'],s['same_C_full_metrics']]):
                for field,value in stored.items():np.testing.assert_allclose(scores[field],value,atol=1e-12)
            assert s['stage']==s['outer_stages'][str(proposed['C'])]
            counts['outer_splits']+=1
        print(name,'C/support selection and refitted stages audited',flush=True)
    assert counts['outer_splits']==44
    counts['scope']='all_nested_teacher_refits_certificates_and_selection_prediction_replay_not_LP_optimality_replay'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
