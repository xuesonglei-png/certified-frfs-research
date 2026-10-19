import numpy as np
from certified_frfs.learned_reference_geometry import LearnedReferenceGeometryContext
from certified_frfs.feasible_transport import weight_certificate


def test_full_reference_weights_and_reduction_certificate_are_explicit():
    rng = np.random.default_rng(343)
    raw = rng.uniform(size=(24, 5)); y = (raw[:, 0] > .5).astype(int)
    context = LearnedReferenceGeometryContext(raw, y)
    assert np.all(context.original_weights > 0)
    np.testing.assert_allclose(np.linalg.norm(context.original_weights), 1., atol=1e-12)
    assert context.cap >= max(context.original_weights)
    assert 3*context.cap >= context.mass
    stage = context.fit_stage(C=1., pair_penalty=None)
    np.testing.assert_allclose(stage['reference_metric']['weights'], context.original_weights)
    assert 'not_preserved_uniform_relation' in stage['reference_metric']['scope']
    full = context.candidate(range(5)); assert full['certificate']['feasible']
    for candidate in stage['path']['trace']:
        fs = candidate['features']; weights = np.array(candidate['weights'])[fs]
        np.testing.assert_allclose(sum(weights), context.mass, atol=1e-10)
        assert max(weights) <= context.cap+1e-10
        independent = weight_certificate(context.A, y, fs, context.original_weights, weights)
        assert independent['feasible'] and independent == candidate['certificate']
