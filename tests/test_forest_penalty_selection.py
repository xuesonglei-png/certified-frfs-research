import numpy as np
import pytest
from certified_frfs.forest_penalty_selection import choose_option, OPTIONS
from certified_frfs.nested_teacher_regularized import inner_splits


def test_ties_prefer_additive_then_stronger_pair_penalty():
    rows = [{'scores': {k: .8 for k in OPTIONS}}]
    assert choose_option(rows)[0] == 'additive'
    rows[0]['scores']['additive'] = .7
    rows[0]['scores']['1.0'] += 5e-15
    assert choose_option(rows)[0] == '10.0'
    rows[0]['scores']['3.0'] = .81
    assert choose_option(rows)[0] == '3.0'


def test_singletons_never_enter_validation_or_leave_training():
    y = np.array([0]*4+[1]*4+[2])
    folds = inner_splits(y, seed=20261012)
    seen = []
    for a, b in folds:
        assert 8 in a and 8 not in b and set(a).isdisjoint(b)
        seen.extend(b)
    assert sorted(seen) == list(range(8))


def test_selection_rejects_missing_or_nonfinite_evidence():
    with pytest.raises(ValueError):
        choose_option([])
    with pytest.raises(ValueError):
        choose_option([{'scores': {k: np.nan for k in OPTIONS}}])
