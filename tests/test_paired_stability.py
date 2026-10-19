import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_paired_stability import paired_score


def test_constant_improvement_and_self_comparison():
    base = np.array([.6, .7, .8])
    scores = paired_score(base+.02, base)
    assert scores['penalized_gain'] == pytest.approx(.02)
    assert paired_score(base, base)['penalized_gain'] == 0.


def test_variable_gain_is_penalized_without_claiming_confidence():
    scores = paired_score([.8, .6, .7], [.7, .7, .7])
    assert scores['mean_gain'] == pytest.approx(0.)
    assert scores['variability_penalty'] == pytest.approx(.1/np.sqrt(3))
    assert scores['penalized_gain'] < 0
    with pytest.raises(ValueError):
        paired_score([.5], [.5])
