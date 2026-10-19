import itertools
import numpy as np

from anchor_subdivision_dependency_v1 import refine_object


def test_strict_anchor_partition_improves_switching_witness_example():
    A=np.array([[.5,0.],[.49,0.],[.51,0.]])
    y=np.array([0,1,1]);wf=np.array([1.,0.]);wr=np.array([1.])
    report=refine_object(A,y,[1],wf,wr,0,max_leaves=2)
    assert report["refined_upper"]<report["root_upper"]-0.005
    assert np.isclose(report["root_upper"],.03)
    assert np.isclose(report["refined_upper"],.02)
    lo,hi=np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    grids=[np.unique([a,(a+b)/2,b]) for a,b in zip(lo.ravel(),hi.ravel())]
    for values in itertools.product(*grids):
        Z=np.array(values).reshape(A.shape)
        full=min(1.,np.min(abs(Z[1:]-Z[0])@wf))
        reduced=min(1.,np.min(abs(Z[1:,1]-Z[0,1])*wr[0]))
        assert full-reduced<=report["refined_upper"]+1e-12


def test_random_signed_bounds_dominate_samples_and_leaf_budget_monotonic():
    rng=np.random.default_rng(8374)
    for _ in range(5):
        A=rng.random((8,3));y=np.repeat([0,1],4);wf=np.ones(3)/np.sqrt(3);wr=np.array([.8,.9])
        small=refine_object(A,y,[0,2],wf,wr,0,max_leaves=1)
        refined=refine_object(A,y,[0,2],wf,wr,0,max_leaves=16)
        assert refined["refined_upper"]<=small["root_upper"]+1e-12
        assert len(refined["leaves"])<=16
        for _ in range(30):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0.,1.)
            full=min(1.,np.min(abs(Z[y!=y[0]]-Z[0])@wf))
            reduced=min(1.,np.min(abs(Z[y!=y[0]][:,[0,2]]-Z[0,[0,2]])@wr))
            assert full-reduced<=refined["refined_upper"]+1e-10
