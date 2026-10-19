import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_decoupled_deployment_C import refit, select, CS, FRACTIONS
from certified_frfs.balanced_teacher_backward_stage import fit_stage
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.triangular_classifier import triangular_box_margins


def test_changed_C_keeps_original_teacher_anchor_requirement():
    rng = np.random.default_rng(92)
    X = rng.random((24, 4))
    y = (X[:, 0] > .6).astype(int)
    source = fit_stage(X, y, C=1)
    fitted = refit(X, y, source, .1)
    assert fitted['teacher_C'] == 1 and fitted['deployment_C'] == .1
    A, _ = preprocess_fit(X, X)
    original = triangular_box_margins(A, y, source['teacher_coef'], source['teacher_intercept'], source['teacher_classes'])
    mask = original > 1e-10
    for record in fitted['policies'].values():
        model = record['fit']
        actual = triangular_box_margins(A, y, model['coef'], model['intercept'], source['teacher_classes'])
        assert np.all(actual[mask] >= .5*original[mask]-1e-8)
        assert np.all(actual[mask] > 1e-10)


def test_selector_allows_different_full_and_reduced_C_from_inner_only():
    inner = [{'by_C': {str(C): {'metrics': {f: {'clean': .8 if f == '1.0' else .81,
        'retained': 1 if f == '1.0' else .5} for f in FRACTIONS}} for C in CS}}]
    proposed, full, _ = select(inner)
    assert proposed['fraction'] == '0.5' and full['fraction'] == '1.0'
    assert proposed['C'] == .1 and full['C'] == .1
