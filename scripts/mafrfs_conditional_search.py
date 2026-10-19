"""Published dynamic-pool search with an explicitly supplied fuzzy measure.

Xu et al., IEEE TFS34(1),138-151 (2026), printed p143 Algorithm2.
This engine does not supply author-specific relations, S/N or hyperparameters.
It is NOT evidence of faithful FD+/FCE+ reproduction or new baseline scores.
"""
import numpy as np
from certified_frfs.baselines import mafrfs_margin_ratio


def select_conditional_mafrfs(A, y, k, *, subset_measure, maximize, sop,
                             margin_type, pool_boundary, margin_evaluator=None):
    """Rank pool members dynamically on selected UNION current pool.

    Compare absolute uncertainty/WBMR values: subtracting a common base
    cannot change a finite argmin/argmax. This explicitly extends the empty
    and zero-between-class boundary where Eq23's subtraction is undefined.
    Ties use the smallest original feature index. Boundary conventions and
    pool size must be declared, not passed off as author defaults.
    """
    A, y = np.asarray(A, float), np.asarray(y)
    if A.ndim != 2 or not len(A) or not np.isfinite(A).all() or y.shape != (len(A),):
        raise ValueError('finite training table and one label per object required')
    if isinstance(k, (bool, np.bool_)) or int(k) != k or not 0 <= k <= A.shape[1]:
        raise ValueError('integer k in [0,d] required')
    if isinstance(sop, (bool, np.bool_)) or int(sop) != sop or sop < 1:
        raise ValueError('positive integer pool size required')
    if not isinstance(maximize, (bool, np.bool_)) or not callable(subset_measure):
        raise ValueError('explicit uncertainty orientation and callable measure required')
    if margin_type not in ('local', 'global') or pool_boundary not in ('truncate', 'error'):
        raise ValueError('explicit margin type and pool boundary convention required')
    if margin_evaluator is None:
        margin_evaluator = lambda fs: mafrfs_margin_ratio(A, y, fs, margin_type)
    selected, remaining, trace, cache = [], list(range(A.shape[1])), [], {}

    def measure(fs):
        key = tuple(sorted(fs))
        if key not in cache:
            value = float(subset_measure(key))
            if not np.isfinite(value): raise ValueError('subset uncertainty must be finite')
            cache[key] = value
        return cache[key]

    while len(selected) < k:
        if len(remaining) < sop and pool_boundary == 'error':
            raise ValueError('remaining features smaller than declared pool size')
        pool, pool_trace = [], []
        for _ in range(min(int(sop), len(remaining))):
            base = selected+pool
            candidates = [{'feature': a, 'subset': sorted(base+[a]), 'measure': measure(base+[a])}
                          for a in remaining if a not in pool]
            chosen = min(candidates, key=lambda q: ((-1 if maximize else 1)*q['measure'], q['feature']))['feature']
            pool_trace.append({'base': sorted(base), 'candidates': candidates, 'chosen': chosen})
            pool.append(chosen)
        margins = [{'feature': a, 'WBMR': float(margin_evaluator(tuple(sorted(selected+[a]))))} for a in pool]
        if any(np.isnan(q['WBMR']) or q['WBMR'] < 0 for q in margins):
            raise ValueError('nonnegative WBMR (including explicit +infinity) required')
        chosen = min(margins, key=lambda q: (q['WBMR'], q['feature']))['feature']
        trace.append({'selected_before': selected.copy(), 'pool': pool, 'pool_trace': pool_trace,
                      'candidate_margins': margins, 'selected_feature': chosen})
        selected.append(chosen); remaining.remove(chosen)
    return selected, {'trace': trace, 'sop': int(sop), 'maximize': bool(maximize),
                      'margin_type': margin_type, 'pool_boundary': pool_boundary,
                      'tie_policy': 'smallest_original_feature_index',
                      'WBMR_zero_between_convention': 'positive_infinity',
                      'empty_base_comparison': 'absolute_scores_without_undefined_subtraction',
                      'scope': 'published_dynamic_search_given_explicit_measure_not_author_reproduction'}
