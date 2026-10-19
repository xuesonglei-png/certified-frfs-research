from collections import Counter
import numpy as np
from capped_third_enemy_hinge import CappedThirdEnemyHinge
from audit_third_enemy_fit import ScalarRankValidator
from fit_capped_frs_cell_hinge import fit_capped_frs_cell_target
from audit_capped_frs_cell_fit import check_capped_frs_fit


def test_capped_DC_identity_and_minor_support_are_valid_everywhere():
    rng = np.random.default_rng(345); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    model = CappedThirdEnemyHinge(A, y, np.ones(3)/3)
    validator = ScalarRankValidator(A, model.export()); seen = set()
    for _ in range(20):
        point = rng.dirichlet(np.ones(3)); elsewhere = rng.dirichlet(np.ones(3))
        two, three = validator.top_sum(point, 2), validator.top_sum(point, 3)
        H, faces, shifts = model.minorant_at(point); seen.update(shifts)
        np.testing.assert_allclose(model.losses(point), three-H, atol=1e-10)
        np.testing.assert_allclose(model.losses(point), np.minimum(1.,validator.losses(point)), atol=1e-10)
        H_elsewhere = np.maximum(validator.top_sum(elsewhere, 2), validator.top_sum(elsewhere, 3)-1.)
        for i, (face, shift) in enumerate(zip(faces, shifts)):
            validator.validate(point, face)
            assert face['k'] == (3 if shift else 2)
            tangent = np.asarray(face['gradient'])@point+face['intercept']-shift
            np.testing.assert_allclose(tangent, H[i], atol=1e-10)
            plane = np.asarray(face['gradient'])@elsewhere+face['intercept']-shift
            assert plane <= H_elsewhere[i]+1e-10
    assert seen == {0, 1}


def test_capped_training_objective_and_duals_replay_independently():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    target, fit = fit_capped_frs_cell_target(A, y, [0, 1, 2], anchor,
        full_reference=anchor, reference_scope='original_full')
    assert fit['optimizer_success'], fit['status']
    assert fit['class_cell_final']['feasible'] and fit['training_objective_final'] <= fit['training_objective_initial']
    counts = Counter()
    check_capped_frs_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit}, counts)
    assert counts['independent_master_primal_dual_replays'] >= 1
    assert 0 <= fit['training_error_upper_bound_final'] <= 1


def test_capped_loss_retains_explicit_small_class_constants():
    A = np.linspace(0, 1, 18)[:, None]; y = np.r_[np.zeros(16), [1, 2]]
    model = CappedThirdEnemyHinge(A, y, np.ones(1)); H, faces, shifts = model.minorant_at(np.ones(1))
    for i in [16, 17]:
        assert model.rows[i]['constant_loss'] == 1.
        assert model.losses(np.ones(1))[i] == 1. and H[i] == 0.
        assert shifts[i] == 0 and faces[i]['k'] == 2
