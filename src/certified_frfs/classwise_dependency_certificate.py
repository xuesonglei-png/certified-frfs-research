"""Class-conditional aggregation of reference objectwise dependency bounds.

If every class bound is <=eta, every prior mixture of these SAME finite
reference classes has loss <=eta. This is not a population/test-risk bound.
"""
import numpy as np


def classwise_dependency_certificate(y, original_certificate, *, eta=.1):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('invalid classes/threshold')
    dependency = original_certificate['dependency_certificate']
    if 'redistributed_object_bounds' in dependency:
        values = np.asarray(dependency['redistributed_object_bounds'], float)
        if values.shape != y.shape or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError('invalid objectwise bounds')
    elif dependency['upper_bound'] == 0 and dependency['raw_redistributed_bound'] == 0:
        values = np.zeros(len(y))
    else:
        raise ValueError('objectwise certificate required')
    classes, counts = np.unique(y, return_counts=True)
    raw = np.array([np.mean(values[y == c]) for c in classes])
    bounds = np.minimum(1., raw+1e-10*(1+abs(raw)))
    if not np.any(values):
        bounds[:] = 0.
    return {'classes': classes.tolist(), 'counts': counts.tolist(), 'upper_bounds': bounds.tolist(),
            'max_upper_bound': float(bounds.max()), 'eta': float(eta),
            'feasible_all_classes': bool(np.all(bounds <= eta)),
            'scope': 'same_finite_reference_shared_box_class_prior_mixtures_not_population_or_classifier_risk'}
