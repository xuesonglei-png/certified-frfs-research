"""Repair convex class bounds in actual coordinates, preserving thresholds."""
import numpy as np
from canonical_hinge_metric_repair import repair_actual_metric
from frs_one_sided_cell import evaluate_class_cell


def repair_cell_metric(C, rhs, info, actual, anchor):
    actual, linear_trace, linear_blend = repair_actual_metric(C, rhs, actual, anchor)
    trace = [{'kind': 'linear', **r} for r in linear_trace]
    anchor_report = evaluate_class_cell(info, anchor)
    anchor_raw = np.asarray(anchor_report['raw_upper_bounds_with_inflation'])
    assert info['anchor_strictly_feasible'] and anchor_report['feasible']
    total = linear_blend
    for attempt in range(8):
        report = evaluate_class_cell(info, actual)
        if report['feasible'] and (not len(C) or np.all(C@actual >= rhs)):
            return actual, trace, total
        raw = np.asarray(report['raw_upper_bounds_with_inflation'])
        bad = raw > info['eta']
        # A convex bound at the mixture is at most the mixture of bounds.
        fraction = float(min(1., max(10.**(attempt-12),
            1.01*np.max((raw[bad]-info['eta'])/(raw[bad]-anchor_raw[bad]))))) if np.any(bad) else 1e-10
        actual = (1-fraction)*actual+fraction*anchor
        total = 1-(1-total)*(1-fraction)
        actual, retrace, reblend = repair_actual_metric(C, rhs, actual, anchor)
        total = 1-(1-total)*(1-reblend)
        trace.append({'kind': 'convex_class', 'attempt': attempt, 'anchor_fraction': fraction,
                      'class_raw_before': raw.tolist(),
                      'class_raw_after': evaluate_class_cell(info, actual)['raw_upper_bounds_with_inflation']})
        trace.extend({'kind': 'linear', **r} for r in retrace)
    trace.append({'kind': 'explicit_anchor', 'anchor_fraction': 1.})
    return np.array(anchor, float, copy=True), trace, 1.
