"""Exact closure of a fuzzy interaction forest under measurement omission.

Unmeasured raw attributes are replaced by declared fixed reference centers.
Incident pair terms become unary terms or constants, preserving exact box
certification on the remaining attributes. This is a fixed-model certificate,
not a test-accuracy or fuzzy-rough dependency guarantee.
"""
import numpy as np
from .triangular_classifier import triangular_memberships
from .forest_fuzzy_classifier import forest_box_margins


def project_forest_parameters(W, bias, edges, V, features, center):
    W, bias, V, center = (np.asarray(z, float) for z in (W, bias, V, center))
    edges = [tuple(e) for e in edges]; fs = list(features)
    if W.ndim != 3 or W.shape[2] != 3:
        raise ValueError('triangular unary coefficients required')
    q, d, _ = W.shape
    if bias.shape != (q,) or center.shape != (d,) or V.shape != (q, len(edges), 3, 3):
        raise ValueError('incompatible forest coefficients or centers')
    if len(fs) != len(set(fs)) or any(int(a) != a or not 0 <= a < d for a in fs):
        raise ValueError('invalid support')
    if not all(np.isfinite(z).all() for z in (W, bias, V, center)) or np.any((center < 0)|(center > 1)):
        raise ValueError('finite normalized reference center required')
    # Existing exact oracle validates canonical forest structure, including
    # components that are subsequently omitted.
    forest_box_margins(center[None, :], np.array([0]), W, bias, np.arange(q), edges, V, eps=0.)
    keep = set(fs); Z = triangular_memberships(center[None, :])[0]
    out = W.copy(); b = bias.copy(); retained_edges = []; retained_pairs = []
    omitted = [a for a in range(d) if a not in keep]
    if omitted:
        b += np.einsum('qdk,dk->q', W[:, omitted], Z[omitted]); out[:, omitted] = 0.
    for e, (u, v) in enumerate(edges):
        if u in keep and v in keep:
            retained_edges.append([u, v]); retained_pairs.append(V[:, e])
        elif u in keep:
            out[:, u] += np.einsum('qkl,l->qk', V[:, e], Z[v])
        elif v in keep:
            out[:, v] += np.einsum('k,qkl->ql', Z[u], V[:, e])
        else:
            b += np.einsum('k,qkl,l->q', Z[u], V[:, e], Z[v])
    pairs = np.stack(retained_pairs, axis=1) if retained_pairs else np.zeros((q, 0, 3, 3))
    # Retain ndarray shape for zero edges: JSON [[]] loses its final axes.
    # A serialized consumer must reconstruct (q,len(edges),3,3) explicitly.
    return {'coef': out.tolist(), 'intercept': b.tolist(), 'edges': retained_edges,
            'pair_coef': pairs, 'features': fs, 'center': center.tolist(),
            'scope': 'exact_fixed_model_missing_measurement_substitution'}


def forest_projection_certificate(X, y, W, bias, classes, edges, V, features, center,
                                  *, eps=.01, ratio=.5):
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('margin ratio must be in (0,1]')
    projected = project_forest_parameters(W, bias, edges, V, features, center)
    original = forest_box_margins(X, y, W, bias, classes, edges, V, eps=eps)
    reduced = forest_box_margins(X, y, projected['coef'], projected['intercept'], classes,
                               projected['edges'], projected['pair_coef'], eps=eps)
    protected = original > 1e-10
    retained = protected & (reduced > 1e-10) & (reduced >= ratio*original-1e-10)
    return {'feasible': bool(np.all(retained[protected])), 'ratio': float(ratio),
            'protected_count': int(protected.sum()), 'retained_count': int(retained.sum()),
            'original_margins': original.tolist(), 'reduced_margins': reduced.tolist(),
            'projected': projected, 'scope': 'fixed_reference_model_box_guarantee_no_population_accuracy_claim'}
