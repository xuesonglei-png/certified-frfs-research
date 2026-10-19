import numpy as np
from certified_frfs.adaptive_multiknot_stage import fit_adaptive_certified_stage
from certified_frfs.label_loss_certified_stage import fit_label_loss_certified_stage
from certified_frfs.projected_forest_logloss import MultiknotProjectedLogloss
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate


def test_only_deletion_objective_changes_teacher_and_FRS_contract_preserved():
    rng = np.random.default_rng(756)
    signal = rng.uniform(size=24)
    X, y = np.column_stack([signal, signal, rng.uniform(size=24)]), (signal > .5).astype(int)
    teacher = fit_adaptive_certified_stage(X, y, fraction=1.)
    changed = fit_label_loss_certified_stage(X, y, fraction=.65)
    assert changed['teacher'] == teacher['teacher'] and changed['teacher_selection'] == teacher['teacher_selection']
    A, _ = preprocess_fit(X, X)
    objective = MultiknotProjectedLogloss(A, y, changed['teacher'])
    for row in changed['path']['attempts']:
        np.testing.assert_allclose(row['projected_balanced_logloss'], objective(row['features']), atol=1e-12)
    candidate = changed['candidate']
    fs = candidate['features']
    cert = weight_certificate(A, y, fs, np.ones(3)/np.sqrt(3), np.array(candidate['weights'])[fs])
    assert cert == candidate['certificate'] and cert['feasible']
    assert not changed['teacher_guards_enforced']
