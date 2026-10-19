import numpy as np
from certified_frfs.adaptive_multiknot_stage import fit_adaptive_certified_stage, MODEL_GRID
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.multiknot_forest import scores
from certified_frfs.multiknot_projection import MultiknotProjectionLoss


def test_stage_training_partitions_deletion_losses_and_original_certificate():
    rng = np.random.default_rng(736)
    signal = rng.uniform(size=24)
    X = np.column_stack([signal, signal, rng.uniform(size=24)])
    y = (signal > .5).astype(int)
    result = fit_adaptive_certified_stage(X, y, fraction=.65)
    assert result['candidate']['certificate']['feasible']
    choice = result['teacher_selection']
    expected = min(range(len(MODEL_GRID)), key=lambda i: (round(choice['options'][i]['mean_balanced_logloss'], 12), i))
    assert choice['chosen_index'] == expected
    for fold in choice['folds']:
        a, b = fold['reference_indices'], fold['validation_indices']
        assert set(a).isdisjoint(b) and set(a)|set(b) == set(range(len(X)))
    A, _ = preprocess_fit(X, X)
    projection = MultiknotProjectionLoss(A, y, result['teacher'])
    for attempt in result['path']['attempts']:
        np.testing.assert_allclose(attempt['projection_loss'], projection(attempt['features']), atol=1e-12)
    fs = result['candidate']['features']
    weights = np.array(result['candidate']['weights'])[fs]
    cert = weight_certificate(A, y, fs, np.ones(3)/np.sqrt(3), weights)
    assert cert == result['candidate']['certificate']
    assert scores(A, result['teacher']).shape == (24, 2)
    assert not result['teacher_guards_enforced']
