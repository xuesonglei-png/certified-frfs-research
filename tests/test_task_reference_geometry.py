import numpy as np
import pytest
from certified_frfs.task_reference_geometry import TaskReferenceGeometryContext
from certified_frfs.feasible_transport import weight_certificate


@pytest.mark.parametrize('family',['task_unit','uniform_unit'])
def test_new_reference_is_unit_mass_and_every_accepted_support_is_certified(family):
    rng = np.random.default_rng(291); raw = rng.uniform(size=(24,5)); y = np.array([0,1]*12)
    context = TaskReferenceGeometryContext(raw,y,family=family)
    assert not context.geometry
    stage = context.fit_policy(fraction=.5)
    np.testing.assert_allclose(context.original_weights.sum(),1.,atol=1e-12)
    assert context.cap >= max(context.original_weights) and context.cap >= 2/5
    for point in stage['path']['trace']:
        fs = point['features']; w = np.array(point['weights'])[fs]
        np.testing.assert_allclose(w.sum(),1.,atol=1e-12)
        assert max(w) <= context.cap+1e-12
        certificate = weight_certificate(context.A,y,fs,context.original_weights,w,protected=context.protected)
        assert certificate == point['certificate'] and certificate['feasible']
    assert stage['reference_teacher']['features'] == list(range(5))
    assert stage['fit']['features'] == stage['candidate']['features']
    if family == 'uniform_unit': np.testing.assert_allclose(context.original_weights,np.ones(5)/5)
    else: np.testing.assert_allclose(context.original_weights,stage['sensitivity']['unit_mass_weights'])


def test_train_only_context_exact_budget_and_validation():
    rng = np.random.default_rng(97); raw = rng.uniform(size=(20,4)); y = np.array([0,1]*10)
    context = TaskReferenceGeometryContext(raw,y)
    stage = context.fit_policy(target_count=4)
    assert stage['fit']['features'] == list(range(4)) and not stage['path']['attempts']
    for target in (0,1,2.5,5):
        with pytest.raises(ValueError): context.fit_policy(target_count=target)
    with pytest.raises(ValueError): context.candidate([0])
    with pytest.raises(ValueError): TaskReferenceGeometryContext(raw,y,family='old')
