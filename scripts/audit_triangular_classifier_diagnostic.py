"""Replay fuzzy teacher inner/outer fits and independent raw-breakpoint margins."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_triangular_classifier_diagnostic import ROOT,FOLDER,CS,fit,ba
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters

def membership(x):
    return np.column_stack([np.maximum(0,1-2*x),1-abs(2*x-1),np.maximum(0,2*x-1)])

def manual_margins(X,y,W,b,classes):
    exact=[];relaxed=[]
    for raw,label in zip(X,y):
        own=int(np.flatnonzero(classes==label)[0]);pairs=[];loose=[]
        for enemy in range(len(classes)):
            if enemy==own:continue
            value=b[own]-b[enemy];lower=value
            for a,x in enumerate(raw):
                lo=max(0,x-.01);hi=min(1,x+.01);points=np.array([lo,hi,np.clip(.5,lo,hi)])
                m=membership(points);difference=W[own,a]-W[enemy,a]
                value+=min(m@difference)
                lower+=np.where(difference>=0,difference*m.min(axis=0),difference*m.max(axis=0)).sum()
            pairs.append(value);loose.append(lower)
        exact.append(min(pairs));relaxed.append(min(loose))
    return np.array(exact),np.array(relaxed)

def main():
    threadpool_limits(1);summary=json.loads((FOLDER/'complete.json').read_text());counts={'splits':0,'inner_fits':0,'outer_fits':0,'heldout_objects':0}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        source=json.loads((ROOT/'results/nested-verified-classwise'/f'{name}.json').read_text())
        for rel,digest in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        for i,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert set(ri).isdisjoint(vi)
            trace={str(C):[] for C in CS}
            for inner in source['splits'][i]['inner']:
                a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(X[ri[a]],X[ri[b]])
                query=np.stack([membership(B[:,j]) for j in range(B.shape[1])],axis=1).reshape(len(B),-1)
                for C in CS:
                    trace[str(C)].append(ba(y[ri[b]],fit(A,y[ri[a]],C).predict(query)));counts['inner_fits']+=1
            assert trace==s['inner_trace'];chosen=min(CS,key=lambda C:(-np.mean(trace[str(C)]),C));assert chosen==s['C']
            A,B=preprocess_fit(X[ri],X[vi]);teacher=fit(A,y[ri],chosen);W,b,classes=linear_parameters(teacher);W=W.reshape(len(classes),X.shape[1],3)
            np.testing.assert_allclose(W,s['coef'],atol=1e-12);np.testing.assert_allclose(b,s['intercept'],atol=1e-12)
            query=np.stack([membership(B[:,j]) for j in range(B.shape[1])],axis=1).reshape(len(B),-1)
            np.testing.assert_allclose(ba(y[vi],teacher.predict(query)),s['clean'],atol=1e-12)
            exact,relaxed=manual_margins(B,y[vi],W,b,classes)
            np.testing.assert_allclose(exact,s['coupled_margins'],atol=1e-10);np.testing.assert_allclose(relaxed,s['independent_margins'],atol=1e-10)
            assert np.all(relaxed<=exact+1e-10)
            for margins,key in ((exact,'coupled_certified'),(relaxed,'independent_certified')):
                rate=np.mean([np.mean(margins[y[vi]==c]>1e-10) for c in np.unique(y[vi])])
                np.testing.assert_allclose(rate,s[key],atol=1e-12)
            counts['splits']+=1;counts['outer_fits']+=1;counts['heldout_objects']+=len(vi)
        print(name,'fuzzy teacher fits and shared-source bounds verified',flush=True)
    assert counts['splits']==44 and counts['inner_fits']==384
    counts['scope']='full_teacher_refit_selection_raw_breakpoint_margin_replay_capacity_only_not_reduction_or_confirmation'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
