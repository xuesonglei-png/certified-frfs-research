"""Standalone epigraph/integer/Big-M checks plus deployed certificate audit."""
import json
import numpy as np
import audit_classwise_box_pilot as base
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import box_class_margins

FOLDER=base.ROOT/'results/verified-classwise-box-pilot'

def verify_point(A,y,teacher,trial):
    fit=trial['fit'];fs=trial['candidate']['features'];classes=np.array(teacher['teacher_classes'])
    Wt,bt=np.array(teacher['teacher_coef']),np.array(teacher['teacher_intercept']);p=np.array(fit['primal_solution'])
    n,d=A.shape;q=len(classes);k=len(fs);nw=q*k;wo=nw+q;uo=wo+nw;pairlist=[(c,r) for c in range(q) for r in range(c+1,q)]
    so=uo+len(pairlist)*k;do=so+n;zo=do+n*(q-1)
    assert len(p)==zo+n and np.isfinite(p).all()
    W=p[:nw].reshape(q,k);b=p[nw:nw+q];ab=p[wo:uo].reshape(q,k);pair=p[uo:so].reshape(len(pairlist),k)
    slack=p[so:do];dist=p[do:zo].reshape(n,q-1);z=p[zo:]
    bound=max(1.,2*abs(Wt-Wt[0]).max());ibound=max(1.,2*abs(bt-bt[0]).max())
    assert fit['coefficient_bound']==bound and fit['intercept_bound']==ibound
    assert abs(W).max()<=bound+1e-8 and abs(b).max()<=ibound+1e-8
    assert np.all(W[0]==0) and b[0]==0
    assert np.all(ab>=-1e-8) and np.all(ab<=bound+1e-8)
    assert np.all(pair>=-1e-8) and np.all(pair<=2*bound+1e-8)
    assert np.all(slack>=-1e-8) and np.all(dist>=-1e-8)
    assert np.all(z>=-1e-8) and np.all(z<=1+1e-8) and abs(z-np.rint(z)).max()<=1e-6
    residuals=[float(np.maximum(abs(W)-ab,0).max())]
    ids={p:i for i,p in enumerate(pairlist)}
    for j,(c,r) in enumerate(pairlist):residuals.append(float(np.maximum(abs(W[c]-W[r])-pair[j],0).max()))
    original=box_class_margins(A,y,Wt,bt,classes);protected=original>1e-10
    assert np.all(z[~protected]==0)
    lo=np.maximum(0,A[:,fs]-.01);hi=np.minimum(1,A[:,fs]+.01);mid=(hi+lo)/2;rad=(hi-lo)/2
    own=np.argmax(y[:,None]==classes[None,:],axis=1)
    for c in range(q):
        mask=own==c
        for r in range(q):
            if r==c:continue
            pi=ids[tuple(sorted((c,r)))];lower=mid[mask]@(W[c]-W[r])+b[c]-b[r]-rad[mask]@pair[pi]
            residuals.append(float(np.maximum(1-lower-slack[mask],0).max()))
            guarded=protected[mask]
            if guarded.any():
                threshold=np.maximum(.5*original[mask][guarded],1e-7)
                bigM=2*bound*hi[mask][guarded].sum(axis=1)+2*ibound+threshold
                residuals.append(float(np.maximum(threshold-lower[guarded]-bigM*(1-z[mask][guarded]),0).max()))
        group=mask&protected;minimum=int(np.ceil(.9*group.sum()-1e-12))
        assert z[group].sum()>=minimum-1e-7
    target=A@(Wt-Wt[0]).T+(bt-bt[0]);scores=A[:,fs]@W.T+b
    residuals.append(float(np.maximum(abs(scores[:,1:]-scores[:,[0]]-target[:,1:])-dist,0).max()))
    assert max(residuals)<=1e-7
    deployed=np.array(fit['coef']);np.testing.assert_allclose(deployed[:,fs],W,atol=1e-12)
    np.testing.assert_allclose(fit['intercept'],b,atol=1e-12)
    balance=np.array([1/(q*sum(y==c)) for c in y]);margin=box_class_margins(A,y,deployed,b,classes)
    objective=.01*abs(W).sum()+balance@np.maximum(0,1-margin)+balance@np.mean(abs(scores[:,1:]-scores[:,[0]]-target[:,1:]),axis=1)
    np.testing.assert_allclose(objective,fit['objective'],atol=1e-10)
    assert fit['solver_optimal']==(fit['status']==0)
    return max(residuals)

def main():
    base.FOLDER=FOLDER;base.main()
    summary=json.loads((FOLDER/'complete.json').read_text());counts={'verified_points':0,'nonoptimal_incumbents':0,'max_primal_inequality_violation':0.}
    for row in summary['rows']:
        name=row['dataset'];p=json.loads((FOLDER/f'{name}.json').read_text());X,y=load_dataset(name)
        for s in p['splits']:
            ri=np.array(s['reference_indices']);A,_=preprocess_fit(X[ri],X[ri])
            for trial in s['trials']:
                if trial['fit']['feasible']:
                    violation=verify_point(A,y[ri],s['teacher'],trial)
                    counts['verified_points']+=1;counts['nonoptimal_incumbents']+=not trial['fit']['solver_optimal']
                    counts['max_primal_inequality_violation']=max(counts['max_primal_inequality_violation'],violation)
    counts['scope']='explicit_primal_solution_epigraph_BigM_integrality_and_actual_objective_not_global_optimality'
    (FOLDER/'complete-primal-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)

if __name__=='__main__':main()
