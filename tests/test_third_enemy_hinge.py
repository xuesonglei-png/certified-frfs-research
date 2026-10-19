from collections import Counter
from itertools import product
import numpy as np
from third_enemy_hinge import ThirdEnemyHinge
from robust_majority_hinge import RobustMajorityHinge
from fit_third_enemy_hinge import fit_third_enemy_target
from label_equivariant_vote import predict


def test_rank_loss_ignores_two_close_enemies_that_do_not_break_majority():
    A = np.r_[0., np.repeat(.1, 8), .8, .01, .02, np.repeat(.8, 6)][:, None]
    y = np.repeat([0, 1], 9); w = np.ones(1)
    rank = ThirdEnemyHinge(A, y, w); convex = RobustMajorityHinge(A, y, w)
    assert rank.rows[0]['constant_loss'] is None
    assert rank.losses(w)[0] == 0 and convex.oracle(w)[0][0] > 1
    row = rank.rows[0]; refs = np.array(row['references'])
    assert predict(abs(A[refs]-A[0]).T, y[refs])['predictions'][0] == 0


def test_rank_third_decomposition_and_support_planes_independently_replay():
    rng = np.random.default_rng(341); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    model = ThirdEnemyHinge(A, y, np.ones(3)/3)
    from audit_third_enemy_fit import ScalarRankValidator
    validator = ScalarRankValidator(A, model.export())
    for _ in range(8):
        w = rng.dirichlet(np.ones(3)); other = rng.dirichlet(np.ones(3))
        third = validator.losses(w)
        np.testing.assert_allclose(model.losses(w), third, atol=1e-10, rtol=1e-9)
        for k in (2, 3):
            value, faces = model.top_sum(w, k); elsewhere, _ = model.top_sum(other, k)
            for i, face in enumerate(faces):
                validator.validate(w, face)
                assert elsewhere[i] >= np.array(face['gradient'])@other+face['intercept']-1e-9


def test_pointwise_error_bound_on_exact_one_query_corner_tables():
    # One query, three fixed friends and four enemies; enumerate ALL256
    # interval-corner tables, independently computing contrasts and 5NN votes.
    nominal = np.array([.5, .43, .45, .48, .46, .52, .58, .72]); labels = np.r_[0, np.zeros(3), np.ones(4)]
    lo, hi = nominal-.01, nominal+.01; margin = .02
    hinges = []
    for enemy in range(4, 8):
        lower = []
        for friend in range(1, 4):
            lower.append(min(max(0., lo[enemy]-t, t-hi[enemy])-max(abs(t-lo[friend]), abs(t-hi[friend]))
                             for t in (lo[0], hi[0])))
        hinges.append(max(0., 1-min(lower)/margin))
    bound = min(1., sorted(hinges, reverse=True)[2])
    for bits in product([0, 1], repeat=8):
        Z = np.where(bits, hi, lo)
        error = float(predict(abs(Z[1:]-Z[0])[None, :], labels[1:])['predictions'][0] != labels[0])
        assert error <= bound+1e-12


def test_cccp_all_steps_decrease_actual_objective_and_lp_audit_passes():
    rng = np.random.default_rng(781); A = rng.uniform(0, 1, (18, 3)); y = np.repeat([0, 1], 9)
    anchor = np.ones(3)/np.sqrt(3)
    from audit_third_enemy_fit import check_fit
    for constrained in (False, True):
        target, fit = fit_third_enemy_target(A, y, [0, 1, 2], anchor, constrained=constrained)
        assert fit['optimizer_success'], fit['status']
        assert fit['training_objective_final'] <= fit['training_objective_initial']+1e-10
        counts = Counter(); check_fit(A, y, {'features': [0, 1, 2], 'anchor': anchor.tolist(),
                                           'target': target.tolist(), 'fit': fit}, counts)
        assert counts['independent_dc_fit_replays'] == 1 and counts['independent_master_primal_dual_replays'] >= 1


def test_invalid_dc_face_is_rejected_even_if_it_has_a_plausible_value():
    rng = np.random.default_rng(911); A = rng.uniform(0, 1, (18, 2)); y = np.repeat([0, 1], 9); w = np.array([.4, .6])
    model = ThirdEnemyHinge(A, y, w); face = model.top_sum(w, 2)[1][0]
    assert model.rows[0]['constant_loss'] is None
    from audit_third_enemy_fit import ScalarRankValidator
    import pytest
    altered = {**face, 'gradient': (np.array(face['gradient'])+.1).tolist()}
    with pytest.raises(AssertionError):
        ScalarRankValidator(A, model.export()).validate(w, altered)
