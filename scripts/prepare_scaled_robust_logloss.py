"""Separate mathematically identical objective-scaled numerical diagnostic.

Frozen28/29 source/results are preserved. Positive scale n preserves minimizers;
reported relaxation values are normalized back to the original objective.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    original = ROOT/'src/certified_frfs/forest_robust_logloss.py'
    source = original.read_text(encoding='utf-8')
    old = "objective = cp.Minimize(alpha@cp.log_sum_exp(upper,axis=1)+cp.sum_squares(penalized)/(2*C*n))"
    assert source.count(old) == 1
    source = source.replace(old,"objective = cp.Minimize(n*(alpha@cp.log_sum_exp(upper,axis=1)+cp.sum_squares(penalized)/(2*C*n)))")
    assert source.count('float(problem.value)') == 2
    source = source.replace('float(problem.value)','float(problem.value/n)')
    source = source.replace("'class_balanced':True,'teacher_guards_enforced':False,","'class_balanced':True,'teacher_guards_enforced':False,'objective_scale':n,")
    source = source.replace('Label-targeted robust log-loss envelope via shared-coordinate forest cuts.','Numerical diagnostic: n-scaled equivalent robust log-loss objective.')
    (ROOT/'src/certified_frfs/forest_robust_logloss_scaled.py').write_text(source,encoding='utf-8')
