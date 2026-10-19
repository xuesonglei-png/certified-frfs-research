import numpy as np
from crossgroup_soft5_terms import crossgroup_terms,assignments
from soft_topk_metric_objective import soft_topk_objective


def test_query_reference_exclusion_robust_bounds_and_rare_queries():
    A=np.random.default_rng(6).uniform(size=(18,4))
    y=np.array([0]*8+[1]*8+[2]+[3])
    (C,same,mask,labels),info=crossgroup_terms(A,y)
    g=np.array(info['group_assignments'])
    assert info['excluded_query_indices']==[16,17]
    for row,i in enumerate(info['query_indices']):
        for j in range(len(A)):
            assert mask[row,j]==(g[i]!=g[j])
            assert same[row,j]==(y[i]==y[j] and mask[row,j])
            lo_i,hi_i=np.maximum(0,A[i]-.01),np.minimum(1,A[i]+.01)
            lo_j,hi_j=np.maximum(0,A[j]-.01),np.minimum(1,A[j]+.01)
            expected=(np.maximum(abs(lo_i-hi_j),abs(hi_i-lo_j)) if y[i]==y[j] and i!=j
                      else np.maximum(0,np.maximum(lo_i-hi_j,lo_j-hi_i)))
            np.testing.assert_allclose(C[row,j],expected)
    assert np.all(mask.sum(axis=1)>=5)
    assert np.array_equal(g,assignments(y))


def test_crossgroup_implicit_soft5_gradient():
    A=np.random.default_rng(9).uniform(size=(21,4))
    y=np.repeat([0,1,2],7)
    terms,_=crossgroup_terms(A,y)
    # The implicit threshold solver stops at 1e-10 mass error. A central
    # 1e-4 step avoids magnifying that error with machine-scale differences.
    w=np.array([.1,.2,.3,.4]);h=1e-4
    numeric=np.array([(soft_topk_objective(w+np.eye(4)[i]*h,terms)[0]
        -soft_topk_objective(w-np.eye(4)[i]*h,terms)[0])/(2*h) for i in range(4)])
    np.testing.assert_allclose(numeric,soft_topk_objective(w,terms)[1],atol=1e-6,rtol=0)


def test_neighbor_groups_do_not_depend_on_query_table_values():
    # Group formation is train-label only; geometry does not alter membership.
    A=np.random.default_rng(9).uniform(size=(18,4));y=np.repeat([0,1,2],6)
    a,ia=crossgroup_terms(A,y);b,ib=crossgroup_terms(1-A,y)
    assert ia==ib
    assert np.array_equal(a[2],b[2])


def test_independent_scalar_loss_and_grouping():
    from independent_crossgroup_soft5 import independent_group_info,independent_objective
    for n in (8,18,21):
        A=np.random.default_rng(n).uniform(size=(n,4));y=np.arange(n)%3
        terms,info=crossgroup_terms(A,y)
        assert info==independent_group_info(y)
        for w in (np.ones(4)/4,np.array([.1,.2,.3,.4]),np.array([0.,.4,0.,.6])):
            np.testing.assert_allclose(soft_topk_objective(w,terms)[0],
                independent_objective(A,y,w),atol=1e-9,rtol=0)
