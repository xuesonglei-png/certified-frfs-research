"""Independent small-table geometric embedding and solver-contract tests.

True absolute differences and memberships are assigned independently. We
enumerate every selector one-hot group, rather than running the bound solver
to decide whether the embedding is feasible. No classification data is read.
"""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shared_graph_dependency_v1 as source


def scalar_memberships(Z,y,w):
    return np.asarray([min(1.,min(sum(float(w[a])*abs(float(Z[i,a])-float(Z[j,a]))
                                    for a in range(Z.shape[1]))
                                for j in range(len(Z)) if y[i]!=y[j]))
                       for i in range(len(Z))])


def structure(model,z):
    sparse,rhs,cost=model.arrays();matrix=sparse.toarray();nd=z.size
    qindices=[j for j in range(nd,len(cost)) if not model.integer[j] and cost[j]==0]
    pairs={}
    for q in qindices:
        for row in matrix:
            zi=np.flatnonzero(row[:nd])
            if row[q]!=0 and len(zi)==2 and np.isclose(abs(row[zi[0]]),1.) and np.isclose(abs(row[zi[1]]),1.):
                if np.count_nonzero(row)==3:
                    assert zi[0]//z.shape[1]!=zi[1]//z.shape[1]
                    assert zi[0]%z.shape[1]==zi[1]%z.shape[1]
                    pairs[q]=tuple(zi);break
        assert q in pairs
    # A pair coordinate is truly shared instead of duplicated per metric.
    assert len(set(pairs.values()))==len(pairs)
    groups=[]
    for row,b in zip(matrix,rhs,strict=True):
        ids=np.flatnonzero(row)
        if b==1. and len(ids) and all(model.integer[i] and row[i]==1. for i in ids):
            groups.append(tuple(ids))
    groups=list(dict.fromkeys(groups));group_variables={j for group in groups for j in group}
    signs=[j for j,v in enumerate(model.integer) if v and j not in group_variables]
    return matrix,rhs,cost,pairs,groups,signs


def embed(A,y,full,reduced,target,eps,Z,exact_abs=False,object_upper=None,builder=None,enemy_upper=None):
    builder=source.build if builder is None else builder
    kwargs={'eps':eps,'exact_abs':exact_abs,'object_upper':object_upper}
    if enemy_upper is not None:kwargs['enemy_upper']=enemy_upper
    model,z,meta=builder(A,y,full,reduced,target,**kwargs)
    matrix,rhs,cost,pairs,groups,signs=structure(model,z)
    x=np.zeros(len(cost));x[z]=Z
    for q,(a,b) in pairs.items():x[q]=abs(x[a]-x[b])
    own=np.flatnonzero(y==target)
    full_values=scalar_memberships(Z,y,full);red_values=scalar_memberships(Z,y,reduced)
    x[np.flatnonzero(cost<0)]=full_values[own]
    x[np.flatnonzero(cost>0)]=red_values[own]
    # Each exact-absolute sign has its own inequalities and no selector row.
    for sign in signs:
        rows=matrix[:,sign]!=0;x[sign]=0.
        if np.max(matrix[rows]@x-rhs[rows])>1e-10:x[sign]=1.
        assert np.max(matrix[rows]@x-rhs[rows])<=1e-10
    for choices in itertools.product(*groups):
        for group in groups:x[list(group)]=0.
        x[list(choices)]=1.
        if np.max(matrix@x-rhs)<=1e-10:
            assert np.max(np.asarray(model.lower)-x)<=1e-10
            assert np.max(x-np.asarray(model.upper))<=1e-10
            assert all(abs(x[j]-round(x[j]))<=1e-12 for j,v in enumerate(model.integer) if v)
            np.testing.assert_allclose(-cost@x,float((full_values-red_values)[own].mean()),atol=1e-12,rtol=0)
            return meta
    raise AssertionError('Actual shared table did not embed in the model')


CASES=[
    (np.array([[.04],[.12],[.2]]),np.array([0,1,1]),[3.],[1.],0,.06),
    (np.array([[0.,.4],[.05,.95],[.7,.2],[1.,.9]]),np.array([0,0,1,1]),[1.,.2],[1.2,.8],0,.35),
    (np.array([[.1,.2],[.5,.1],[.7,.8]]),np.array([0,0,1]),[1.,.2],[0.,0.],0,.15),
    (np.array([[.1,.2],[.5,.1],[.7,.8]]),np.array([0,0,1]),[0.,0.],[1.,.2],0,.15),
    (np.array([[0.,0.],[1.,1.],[1.,1.]]),np.array([0,1,1]),[50.,0.],[0.,30.],0,.01),
    (np.array([[.5],[.5],[.5],[.5]]),np.array([8,8,-3,-3]),[1.],[.8],8,.1),
    (np.array([[0.],[.2],[.4]]),np.array([0,1,1]),[.5],[1.],0,0.),
    (np.array([[.1,.9],[.6,.4],[.8,.7]]),np.array([0,1,1]),[0.,0.],[0.,0.],0,.4),
]


@pytest.mark.parametrize('A,y,wf,wr,target,eps',CASES)
@pytest.mark.parametrize('exact_abs',[False,True])
def test_every_corner_and_fixed_interior_tables_have_actual_loss_embedding(A,y,wf,wr,target,eps,exact_abs):
    low,high=np.maximum(0.,A-eps),np.minimum(1.,A+eps)
    for corner in itertools.product((0,1),repeat=A.size):
        Z=np.where(np.asarray(corner).reshape(A.shape),high,low)
        embed(A,y,wf,wr,target,eps,Z,exact_abs)
    rng=np.random.default_rng(9817)
    for _ in range(8):embed(A,y,wf,wr,target,eps,rng.uniform(low,high),exact_abs)


def test_timeout_uses_dual_not_relaxation_incumbent_or_fun(monkeypatch):
    monkeypatch.setattr(source,'milp',lambda *args,**kwargs:SimpleNamespace(
        status=1,message='fixed synthetic timeout',mip_dual_bound=-.4,fun=-.05,x=None))
    report=source.class_upper([[.1],[.2]],[0,1],[1.],[.5],0)
    assert report['solver_bound_available'] and report['solver_upper_bound']==pytest.approx(.4)
    assert report['upper_bound']>.4 and report['observed_signed_loss_lower']==pytest.approx(.05)


def test_missing_dual_on_timeout_cannot_promote_primal_fun(monkeypatch):
    monkeypatch.setattr(source,'milp',lambda *args,**kwargs:SimpleNamespace(
        status=1,message='fixed synthetic timeout',fun=-.05,x=None))
    report=source.class_upper([[.1],[.2]],[0,1],[1.],[.5],0)
    assert not report['solver_bound_available'] and report['upper_bound']==1.


def test_inflated_relaxation_primal_is_not_reported_as_a_real_attack(monkeypatch):
    A=np.full((2,2),.5);y=np.array([0,1]);full=np.array([1.,1.]);reduced=np.array([1.,0.])
    model,z,_=source.build(A,y,full,reduced,0,eps=.1)
    matrix,rhs,cost,pairs,groups,signs=structure(model,z);assert not groups and not signs
    x=np.zeros(len(cost));x[z]=A
    for q in pairs:x[q]=.2
    x[cost<0]=.4;x[cost>0]=.2
    assert np.max(matrix@x-rhs)<=1e-12
    monkeypatch.setattr(source,'milp',lambda *args,**kwargs:SimpleNamespace(
        status=0,message='synthetic feasible hull incumbent',mip_dual_bound=-.2,fun=-.2,x=x))
    report=source.class_upper(A,y,full,reduced,0,eps=.1)
    assert report['relaxation_primal_objective']==pytest.approx(.2)
    assert report['primal_max_violation']<=1e-12
    assert report['observed_signed_loss_lower']==pytest.approx(0.)
    assert report['upper_bound']>=.2


def test_dual_contradicting_actual_nominal_table_is_rejected(monkeypatch):
    monkeypatch.setattr(source,'milp',lambda *args,**kwargs:SimpleNamespace(
        status=1,message='synthetic false dual',mip_dual_bound=.2,fun=None,x=None))
    with pytest.raises(ArithmeticError):source.class_upper([[.1],[.2]],[0,1],[1.],[.5],0)


def test_independent_signed_object_cuts_preserve_actual_tables_and_negative_bounds():
    A=np.array([[.1],[.2],[.7]]);y=np.array([0,0,1]);wf=np.array([.2]);wr=np.array([.8]);eps=.01
    lo,hi=np.maximum(0.,A-eps),np.minimum(1.,A+eps);bounds=[]
    for i in range(len(A)):
        enemies=[j for j in range(len(A)) if y[j]!=y[i]]
        full_upper=min(sum(wf[a]*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a]))
                           for a in range(A.shape[1])) for j in enemies)
        reduced_lower=min(sum(wr[a]*max(0.,lo[i,a]-hi[j,a],lo[j,a]-hi[i,a])
                              for a in range(A.shape[1])) for j in enemies)
        bounds.append(min(1.,full_upper)-min(1.,reduced_lower))
    assert max(bounds)<0
    for values in itertools.product(*(np.linspace(a,b,3) for a,b in zip(lo.ravel(),hi.ravel()))):
        embed(A,y,wf,wr,0,eps,np.asarray(values).reshape(A.shape),object_upper=bounds)
    for invalid in ([0.],[0.,float('nan'),0.],[[0.],[0.],[0.]]):
        with pytest.raises(ValueError):source.build(A,y,wf,wr,0,object_upper=invalid)


def rectangle_roof(left,right,x,y):
    """Independent normalized four-corner roof and vertex-mixture value."""
    l1,l2=left;h1,h2=right;a=(x-l1)/(h1-l1);b=(y-l2)/(h2-l2)
    v00,v10,v01,v11=abs(l1-l2),abs(h1-l2),abs(l1-h2),abs(h1-h2)
    delta=v00+v11-v10-v01
    first=v00+(v10-v00)*a+(v01-v00)*b
    second=v10+v01-v11+(v11-v01)*a+(v11-v10)*b
    # Given the two vertex marginals a,b, p11 has this exact feasible interval.
    p11=max(0.,a+b-1.)
    mixture=(1.-a-b+p11)*v00+(a-p11)*v10+(b-p11)*v01+p11*v11
    return min(first,second),mixture,delta


def test_rectangle_roof_is_exact_upper_hull_and_dominates_difference_secant():
    rng=np.random.default_rng(20261015)
    for _ in range(80):
        ends=np.sort(rng.uniform(0.,1.,(2,2)),axis=1);left=ends[:,0];right=ends[:,1]
        L=left[0]-right[1];H=right[0]-left[1]
        for a,b in itertools.product(np.linspace(0.,1.,7),repeat=2):
            x=left[0]+a*(right[0]-left[0]);y=left[1]+b*(right[1]-left[1])
            roof,mixture,delta=rectangle_roof(left,right,x,y)
            assert delta<=1e-12
            assert abs(x-y)<=roof+1e-12
            assert roof==pytest.approx(mixture,abs=1e-12)
            secant=abs(x-y) if L>=0 or H<=0 else (H+L)/(H-L)*(x-y)-2*H*L/(H-L)
            assert roof<=secant+1e-12
    # The lifted scalar secant permits q=1 here, while the rectangle hull forces q=0.
    roof,_,delta=rectangle_roof(np.array([0.,0.]),np.array([1.,1.]),0.,0.)
    assert delta==-2. and roof==0.


@pytest.mark.parametrize('A,y,wf,wr,target,eps',CASES)
@pytest.mark.parametrize('exact_abs',[False,True])
def test_v2_rectangle_roof_conditional_and_saturation_cuts_embed_every_true_table(A,y,wf,wr,target,eps,exact_abs):
    from shared_graph_dependency_v2 import build
    wf,wr=np.asarray(wf),np.asarray(wr);lo,hi=np.maximum(0.,A-eps),np.minimum(1.,A+eps)
    objects=[];conditional={}
    for i in range(len(A)):
        enemies=[j for j in range(len(A)) if y[j]!=y[i]]
        full_cap=min(1.,min(sum(wf[a]*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a]))
                                   for a in range(A.shape[1])) for j in enemies))
        lower={j:sum(wr[a]*max(0.,lo[i,a]-hi[j,a],lo[j,a]-hi[i,a])
                     for a in range(A.shape[1])) for j in enemies}
        objects.append(full_cap-min(1.,min(lower.values())))
        conditional[i]={j:full_cap-v for j,v in lower.items()}
    for corner in itertools.product((0,1),repeat=A.size):
        Z=np.where(np.asarray(corner).reshape(A.shape),hi,lo)
        embed(A,y,wf,wr,target,eps,Z,exact_abs,objects,builder=build,enemy_upper=conditional)
    rng=np.random.default_rng(77301)
    for _ in range(8):
        embed(A,y,wf,wr,target,eps,rng.uniform(lo,hi),exact_abs,objects,builder=build,enemy_upper=conditional)
