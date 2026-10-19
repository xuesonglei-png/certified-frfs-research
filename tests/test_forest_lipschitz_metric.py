import itertools
import numpy as np
import pytest
from certified_frfs.forest_lipschitz_metric import forest_contrast_lipschitz
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores


def model(q,d,seed):
    rng = np.random.default_rng(seed); edges = [[0,a] for a in range(1,d)]
    return {'coef':rng.normal(size=(q,d,3)).tolist(),'intercept':rng.normal(size=q).tolist(),
        'edges':edges,'pair_coef':rng.normal(size=(q,len(edges),3,3)).tolist(),'classes':list(range(q))}


@pytest.mark.parametrize('q',[2,3])
def test_coordinate_constants_equal_exhaustive_grid_directional_slopes(q):
    fit = model(q,4,234); result = forest_contrast_lipschitz(fit); maximum = np.zeros(4)
    for a in range(4):
        others = [b for b in range(4) if b != a]
        for values in itertools.product((0.,.5,1.),repeat=3):
            for lo,hi in ((0.,.5),(.5,1.)):
                X = np.zeros((2,4)); X[:,others] = values; X[:,a] = [lo,hi]
                slopes = np.diff(orthogonal_scores(X,fit),axis=0)[0]/(hi-lo)
                maximum[a] = max(maximum[a],max(abs(slopes[c]-slopes[r]) for c in range(q) for r in range(c+1,q)))
    np.testing.assert_allclose(result['coordinate_constants'],maximum,atol=1e-12)


def test_task_metric_bounds_all_class_contrast_changes_and_has_unit_mass():
    fit = model(3,5,45); result = forest_contrast_lipschitz(fit); rng = np.random.default_rng(15)
    X,Z = rng.uniform(size=(200,5)),rng.uniform(size=(200,5)); change = orthogonal_scores(X,fit)-orthogonal_scores(Z,fit)
    weights = np.array(result['unit_mass_weights']); np.testing.assert_allclose(weights.sum(),1.,atol=1e-12)
    bound = result['total_constant']*(abs(X-Z)@weights)
    for c,r in result['class_index_pairs']: assert np.all(abs(change[:,c]-change[:,r]) <= bound+1e-12)


def test_class_common_shifts_leave_metric_unchanged_and_zero_model_fallback_is_explicit():
    fit = model(3,4,544); baseline = forest_contrast_lipschitz(fit); rng = np.random.default_rng(73)
    shifted = {**fit,'coef':(np.array(fit['coef'])+rng.normal(size=(1,4,3))).tolist(),
        'pair_coef':(np.array(fit['pair_coef'])+rng.normal(size=(1,3,3,3))).tolist()}
    np.testing.assert_allclose(forest_contrast_lipschitz(shifted)['unit_mass_weights'],baseline['unit_mass_weights'],atol=1e-12)
    constant = {**fit,'coef':np.zeros((3,4,3)).tolist(),'pair_coef':np.zeros((3,3,3,3)).tolist()}
    result = forest_contrast_lipschitz(constant)
    assert result['uniform_fallback'] and result['total_constant'] == 0
    np.testing.assert_allclose(result['unit_mass_weights'],np.ones(4)/4)
