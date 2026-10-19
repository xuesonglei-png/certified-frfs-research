import numpy as np
from robust_majority_hinge import RobustMajorityHinge
from fit_robust_majority_hinge import fit_majority_hinge_target
from independent_majority_hinge import independent_oracle,independent_full_lp
from label_equivariant_vote import predict
from audit_forest_compact_transfer import equivalent


def test_scalar_box_oracle_and_supporting_hyperplanes_match():
    rng=np.random.default_rng(871);A=rng.uniform(0,1,(18,3));y=np.repeat([0,1],9);anchor=np.array([.2,.3,.5])
    oracle=RobustMajorityHinge(A,y,anchor);w=np.array([.7,.2,.1]);loss,faces=oracle.oracle(w)
    fresh,scalar=independent_oracle(A,w,oracle.export());equivalent(loss.tolist(),fresh.tolist());equivalent(faces,scalar)
    for _ in range(12):
        other=rng.dirichlet(np.ones(3));values,_=oracle.oracle(other)
        linear=np.array([np.array(f['gradient'])@other+f['intercept'] for f in faces])
        assert np.all(values>=linear-1e-10)
    left=rng.dirichlet(np.ones(3));right=rng.dirichlet(np.ones(3));t=.37
    assert oracle.objective((1-t)*left+t*right)<=((1-t)*oracle.objective(left)+t*oracle.objective(right))+1e-10


def test_majority_hinge_upper_bounds_crossgroup_errors_for_joint_perturbations():
    rng=np.random.default_rng(931);A=rng.uniform(0,1,(27,2));y=np.repeat([9,0,7],9);anchor=np.array([.4,.6])
    oracle=RobustMajorityHinge(A,y,anchor)
    for _ in range(24):
        w=rng.dirichlet(np.ones(2));loss,_=oracle.oracle(w)
        for draw in range(4):
            table=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);errors=[]
            for row in oracle.rows:
                i=row['query'];ref=np.array(row['references'])
                if len(ref)<5:errors.append(1.);continue
                distance=(abs(table[ref]-table[i])@w)[None,:]
                errors.append(float(predict(distance,y[ref])['predictions'][0]!=y[i]))
            assert np.all(np.array(errors)<=loss+1e-10)
            assert oracle.alpha@errors<=oracle.alpha@loss+1e-10


def test_cutting_plane_optimum_matches_independent_full_epigraph_lp():
    rng=np.random.default_rng(51);A=rng.uniform(0,1,(18,3));y=np.repeat([0,1],9)
    anchor=np.ones(3)/np.sqrt(3)
    weights,fit=fit_majority_hinge_target(A,y,[0,1,2],anchor,max_iter=200)
    assert fit['optimizer_success'],fit['status'];fresh=independent_full_lp(A,anchor/np.sqrt(3),fit['training_witnesses'])
    np.testing.assert_allclose(fit['training_objective_final'],fresh['objective'],atol=1e-7,rtol=0)
    assert fit['trace'][-1]['gap']<=1e-7 and fit['training_objective_final']<=fit['training_objective_initial']+1e-7
    assert abs(sum(weights)-np.sqrt(3))<1e-10 and min(weights)>=0


def test_small_classes_have_explicit_constant_error_bound():
    A=np.linspace(0,1,14).reshape(7,2);y=np.array([0,0,0,0,0,1,1]);anchor=np.array([.5,.5])
    oracle=RobustMajorityHinge(A,y,anchor);loss,_=oracle.oracle(anchor)
    assert all(row['constant_loss']==1 for row in oracle.rows if y[row['query']]==1)
    np.testing.assert_allclose(oracle.alpha.sum(),1)
