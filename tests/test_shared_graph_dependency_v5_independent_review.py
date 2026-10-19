"""Independent V5 embedding checks; no solver calls."""
import itertools
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import independent_positive_graph_embedding_v4 as helper
import shared_graph_dependency_v5 as v5

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


def cuts(A,y,wf,wr,eps):
    lo,hi=np.maximum(0.,A-eps),np.minimum(1.,A+eps);objects=[];enemy={}
    for i in range(len(A)):
        es=[j for j in range(len(A)) if y[j]!=y[i]]
        full_hi=min(1.,min(sum(wf[a]*max(abs(lo[i,a]-hi[j,a]),abs(hi[i,a]-lo[j,a])) for a in range(A.shape[1])) for j in es))
        lower={j:sum(wr[a]*max(0.,lo[i,a]-hi[j,a],lo[j,a]-hi[i,a]) for a in range(A.shape[1])) for j in es}
        objects.append(full_hi-min(1.,min(lower.values())));enemy[i]={j:full_hi-v for j,v in lower.items()}
    return objects,enemy


@pytest.mark.parametrize('A,y,wf,wr,target,eps',CASES)
@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('exact',[False,True])
def test_all_true_corners_and_interior_tables_embed_v5(A,y,wf,wr,target,eps,positive,exact):
    wf,wr=np.asarray(wf),np.asarray(wr);objects,enemy=cuts(A,y,wf,wr,eps)
    original=helper.build;helper.build=v5.build
    try:
        validator=helper.TrueEmbedding(A,y,wf,np.pad(wr,(0,A.shape[1]-len(wr))) if len(wr)<A.shape[1] else wr,
            target,eps=eps,positive=positive,exact_abs=exact,object_upper=objects,enemy_upper=enemy)
        lo,hi=np.maximum(0.,A-eps),np.minimum(1.,A+eps)
        for corner in itertools.product((0,1),repeat=A.size):validator.table(np.where(np.asarray(corner).reshape(A.shape),hi,lo))
        rng=np.random.default_rng(7251)
        for _ in range(8):validator.table(rng.uniform(lo,hi))
    finally:helper.build=original


def test_v5_positive_M_is_interval_valid_and_nonnegative_loss_never_needs_negative_slack():
    rng=np.random.default_rng(825)
    for _ in range(100):
        A=rng.uniform(.05,.95,(4,2));y=np.array([0,0,1,1]);wf=rng.uniform(.1,1.5,2);wr=rng.uniform(.1,1.5,2)
        _,_,meta=v5.build(A,y,wf,wr,0,eps=.08,positive=True,object_upper=np.ones(4))
        lo,hi=np.maximum(0.,A-.08),np.minimum(1.,A+.08)
        for s in meta['selections']:
            assert s['positive_branch_M']>=0
            assert s['positive_branch_M']==pytest.approx(max(0.,s['reduced_upper']-s['full_lower']),abs=1e-12)


def test_v5_seed839_actual_table_is_above_old_scipy_bound_but_model_embedding_is_exact():
    record=__import__('json').loads((Path(__file__).resolve().parents[1]/'results/positive-graph-independent-embedding-1791763926622800200.json').read_text())
    original=helper.build;helper.build=v5.build
    try:
        validator=helper.TrueEmbedding(np.asarray(record['A']),np.asarray(record['y']),record['full'],record['reduced'],0,eps=.08,positive=True)
        actual,residual=validator.table(record['actual_z_table'])
    finally:helper.build=original
    assert actual==pytest.approx(.022875358978938373,abs=1e-12) and residual<=1e-12
