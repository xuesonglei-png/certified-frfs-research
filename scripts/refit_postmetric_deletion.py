"""Relearn remaining metric coordinates after a fixed candidate deletion.

This training-only proposal keeps ORIGINAL full guards and metric mass. A
rejected renormalized deletion may still have a feasible relearned metric.
No claim of minimum reduct, optimizer globality, or population accuracy.
"""
import numpy as np
from simplex_soft_topk_targets import fit_simplex_target
from hardaware_guarded_soft5_descent import hard5_LOO_BA
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

GRID = (0., .25, .5, 1.)


def fit_refitted_deletion(A, y, features, rescaled, parent_ba, *, max_iter=1000, protected=None):
    A, y, anchor = np.asarray(A,float), np.asarray(y), np.asarray(rescaled,float)
    if not np.isfinite(parent_ba) or not 0<=parent_ba<=1:
        raise ValueError('invalid parent training accuracy')
    fs = list(features)
    # Source routine validates support, training table and total metric mass.
    target, fit = fit_simplex_target(A,y,fs,anchor,constrained=False,max_iter=max_iter)
    mass = np.sqrt(A.shape[1]); full = np.ones(A.shape[1])/mass
    checks = []
    for t in GRID:
        w = (1-t)*anchor+t*target
        certificate = weight_certificate(A,y,fs,full,w,protected=protected)
        classes = classwise_dependency_certificate(y,certificate)
        actual = hard5_LOO_BA(A[:,fs],y,w)
        admitted = bool(certificate['feasible'] and classes['feasible_all_classes'])
        eligible = bool(admitted and actual>=parent_ba-1e-12)
        checks.append({'interpolation':t,'weights':w.tolist(),'certificate':certificate,
            'class_certificate':classes,'accepted':admitted,'eligible':eligible,
            'hard_training_LOO_BA':actual,'nonworsening_parent_BA':bool(actual>=parent_ba-1e-12)})
    candidates = [i for i,r in enumerate(checks) if r['eligible']]
    selected = min(candidates,key=lambda i:(-round(checks[i]['hard_training_LOO_BA'],12),checks[i]['interpolation'])) if candidates else None
    return {'features':fs,'rescaled_weights':anchor.tolist(),'target_weights':target.tolist(),
        'parent_hard_training_LOO_BA':float(parent_ba),'refit':fit,'grid':list(GRID),
        'checks':checks,'selected_id':selected,'guard_reference':'original_full',
        'scope':'fixed_candidate_training_refit_not_new_deletion_path_or_predictive_superiority'}
