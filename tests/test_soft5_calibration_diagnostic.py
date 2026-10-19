"""High neighbor leakage need not imply class error; LOO excludes self."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_soft5_calibration import profile


def test_identical_same_class_neighbors_have_leakage_but_negligible_class_error():
    y = np.repeat([0, 1], 12); A = (.1+.8*y)[:, None]
    result = profile(A, y, [np.sqrt(12)])
    assert result['clean_hard5_LOO_BA'] == 1
    assert result['mean_robust_soft_to_hard_leak_fraction'] > .5
    assert result['mean_robust_class_fraction_error'] < 1e-5
    assert result['max_mass_residual'] <= 1e-10
