"""OOF imputation identity, singleton handling and independent replay."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from certified_frfs.heldout_projected_logloss import HeldoutProjectedLogloss
from certified_frfs.multiknot_forest import fit_multiknot_forest, scores
from certified_frfs.adaptive_multiknot_stage import balanced_logloss
from certified_frfs.crossfit_label_loss_stage import fit_crossfit_label_loss_stage
from certified_frfs.validated_pool import preprocess_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_external_center_exact_imputation_with_missing_heldout_class():
    rng = np.random.default_rng(921)
    A = rng.uniform(size=(24, 3))
    y = np.repeat([0, 1, 2], 8)
    fit = fit_multiknot_forest(A, y, [0, 1, 2], [[0, 1], [1, 2]], knot_count=5, C=.1)
    B = rng.uniform(size=(7, 3))
    truth = np.array([0, 1, 0, 1, 0, 1, 0])
    center = A.mean(axis=0)
    objective = HeldoutProjectedLogloss(B, truth, fit, center)
    for fs in ([], [0], [1, 2], [0, 1, 2]):
        raw = B.copy()
        omitted = [a for a in range(3) if a not in fs]
        raw[:, omitted] = center[omitted]
        value = scores(raw, fit)
        np.testing.assert_allclose(objective.projected_scores(fs), value, atol=1e-12)
        assert objective(fs) == pytest.approx(balanced_logloss(value, np.array(fit['classes']), truth), abs=1e-12)
    np.testing.assert_array_equal(objective.center, center)
    for invalid in ([np.nan, .1, .2], [-.1, .1, .2], [.1, .2]):
        with pytest.raises(ValueError):
            HeldoutProjectedLogloss(B, truth, fit, invalid)


def test_crossfit_stage_matches_cv_and_independent_auditor_rejects_corrupt_center():
    import copy
    rng = np.random.default_rng(377)
    signal = rng.uniform(size=26)
    X = np.column_stack([signal]*4+[rng.uniform(size=26)])
    y = (signal > .5).astype(int)
    y[-1] = 2  # never appears in a heldout fold
    stage = fit_crossfit_label_loss_stage(X, y)
    selection = stage['teacher_selection']
    assert stage['path']['trace'][0]['crossfit_balanced_logloss'] == pytest.approx(selection['options'][selection['chosen_index']]['mean_balanced_logloss'])
    for block, center in zip(selection['folds'], stage['path']['fold_centers']):
        a, b = np.array(block['reference_indices']), np.array(block['validation_indices'])
        assert 25 in a and 25 not in b and set(a).isdisjoint(b)
        A, _ = preprocess_fit(X[a], X[b])
        weights = np.array([1/(len(np.unique(y[a]))*sum(y[a] == c)) for c in y[a]])
        np.testing.assert_allclose(center, weights@A, atol=1e-12)
    spec = importlib.util.spec_from_file_location('oof_auditor_test', ROOT/'scripts/audit_crossfit_label_loss_recent_pilot.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    counts = Counter()
    auditor.check_stage(X, y, stage, counts)
    assert counts['stages'] == 1 and counts['FRS_certificate_replays'] >= 2
    corrupted = copy.deepcopy(stage)
    corrupted['path']['fold_centers'][0][0] += .01
    with pytest.raises(AssertionError):
        auditor.check_stage(X, y, corrupted, Counter())


def test_generated_sources_are_exact():
    from prepare_crossfit_label_loss_pilot import generated
    for name, source in generated().items():
        assert (ROOT/'scripts'/name).read_text(encoding='utf-8') == source
        compile(source, name, 'exec')
