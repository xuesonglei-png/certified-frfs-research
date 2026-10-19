import numpy as np
import pytest
import certified_frfs.verified_classwise_box_linear_refit as model

@pytest.mark.parametrize('failure',[None,'missing','fractional','bounds'])
def test_timeout_incumbent_requires_real_primal_and_margin_checks(monkeypatch,failure):
    original=model.milp
    def timeout(*args,**kwargs):
        result=original(*args,**kwargs)
        assert result.success
        result.success=False;result.status=1;result.message='test timeout';result.mip_gap=.2
        if failure=='missing':result.x=None
        elif failure=='fractional':result.x[np.flatnonzero(kwargs['integrality'])[0]]=.5
        elif failure=='bounds':result.x[0]=10000
        return result
    monkeypatch.setattr(model,'milp',timeout)
    X=np.array([[0.,0.],[.1,.1],[.9,.9],[1.,1.]])
    y=np.array([0,0,1,1]);W=np.array([[0.,0.],[2.,2.]]);b=np.array([0.,-2.])
    result=model.fit_verified_classwise_box_linear(X,y,W,b,[0,1],[0])
    assert result['feasible']==(failure is None)
    if failure is None:
        assert not result['solver_optimal'] and result['mip_gap']==.2
        assert result['retained_counts']==[2,2]
        assert abs(result['objective']-result['solver_objective'])<1e-8
