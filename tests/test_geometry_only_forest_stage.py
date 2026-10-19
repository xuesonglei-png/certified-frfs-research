import numpy as np
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores


def test_geometry_stage_preserves_joint_signal_and_certifies_every_policy():
    X = np.array([[u, v, .5, .5, .5] for u in (.1, .9) for v in (.1, .9)]*3)
    y = ((X[:, 0] > .5) == (X[:, 1] > .5)).astype(int)
    context = GeometryStageContext(X, y)
    stage = context.fit_stage(C=1., pair_penalty=3.)
    assert stage['teacher_guards_enforced'] is False
    for record in stage['policies'].values():
        candidate = record['candidate']; fs = candidate['features']
        assert 0 in fs and 1 in fs
        certificate = weight_certificate(context.A, y, fs, np.ones(5)/np.sqrt(5), np.array(candidate['weights'])[fs])
        assert certificate['feasible'] and candidate['certificate']['feasible']
        assert record['fit']['teacher_guards_enforced'] is False
    assert len(stage['policies']['0.8']['candidate']['features']) == 4
    assert len(stage['policies']['0.5']['candidate']['features']) == 3
    # A mechanism test on its training examples, not accuracy evidence.
    fit = stage['policies']['0.5']['fit']
    np.testing.assert_array_equal(np.array(fit['classes'])[orthogonal_scores(context.A, fit).argmax(axis=1)], y)
    empty = context.classifier([], C=1., pair_penalty=3.)
    assert empty['features'] == [] and empty['edges'] == []
    np.testing.assert_array_equal(orthogonal_scores(context.A, empty), np.zeros((len(X), 2)))
