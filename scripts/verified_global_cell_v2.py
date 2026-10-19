"""Conservative conjunction of scalar and vector global-cell evaluations."""
import numpy as np
from frs_global_saturation_transition import evaluate_class_cell as vector_evaluation
from independent_global_frs_transition import scalar_cell_evaluation
from canonical_hinge_metric_repair import repair_actual_metric


def evaluate_class_cell(info,weights):
    vector=vector_evaluation(info,weights);scalar=scalar_cell_evaluation(info,weights)
    raw=np.maximum(vector['raw_upper_bounds_with_inflation'],scalar['raw_upper_bounds_with_inflation'])
    linear=bool(vector['cell_constraints_verified'] and scalar['cell_constraints_verified'])
    return {**scalar,'raw_upper_bounds_with_inflation':raw.tolist(),
        'class_upper_bounds':np.clip(raw,0,1).tolist(),'cell_constraints_verified':linear,
        'feasible':bool(linear and vector['feasible'] and scalar['feasible'] and np.all(raw<=info['eta']))}


def repair_cell_metric(C,rhs,info,actual,anchor):
    actual,linear_trace,total=repair_actual_metric(C,rhs,actual,anchor)
    trace=[{'kind':'linear',**r} for r in linear_trace]
    anchor_report=evaluate_class_cell(info,anchor)
    assert info['anchor_strictly_feasible'] and anchor_report['feasible']
    anchor_raw=np.asarray(anchor_report['raw_upper_bounds_with_inflation'])
    for attempt in range(12):
        report=evaluate_class_cell(info,actual)
        if report['feasible'] and (not len(C) or np.all(C@actual>=rhs)):
            return actual,trace,total
        raw=np.asarray(report['raw_upper_bounds_with_inflation']);bad=raw>info['eta']
        needed=1.01*np.max((raw[bad]-info['eta'])/(raw[bad]-anchor_raw[bad])) if np.any(bad) else 0.
        fraction=float(min(1.,max(10.**(attempt-12),needed)))
        actual=(1-fraction)*actual+fraction*anchor;total=1-(1-total)*(1-fraction)
        actual,retrace,blend=repair_actual_metric(C,rhs,actual,anchor);total=1-(1-total)*(1-blend)
        trace.append({'kind':'dual_path_convex_class','attempt':attempt,'anchor_fraction':fraction,
            'class_raw_before':raw.tolist(),
            'class_raw_after':evaluate_class_cell(info,actual)['raw_upper_bounds_with_inflation']})
        trace.extend({'kind':'linear',**r} for r in retrace)
    trace.append({'kind':'explicit_anchor','anchor_fraction':1.})
    return np.array(anchor,float,copy=True),trace,1.
