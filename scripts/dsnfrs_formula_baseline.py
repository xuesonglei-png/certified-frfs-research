"""DSNFRS (FSS527,109704,2026), Eqs8/13-15/30-31 and Algorithm1.

Formula-level implementation with explicit unresolved numerical conventions.
This is not author-code or a reproduction of the original experiment tables.
No symmetric-distance or nearest-enemy gamma proxy is substituted.
"""
import numpy as np


class DirectedSoftNeighborhood:
    def __init__(self, A, y, *, k, beta, variance_ddof, zero_variance_lambda,
                 singleton_variance):
        self.A, self.y = np.asarray(A, float), np.asarray(y)
        if (self.A.ndim != 2 or not len(self.A) or not np.isfinite(self.A).all()
                or self.y.shape != (len(self.A),) or np.any(self.y != self.y)):
            raise ValueError('finite training table and valid labels required')
        classes, inverse, sizes = np.unique(self.y, return_inverse=True, return_counts=True)
        if len(classes) < 2: raise ValueError('two or more training classes required')
        if isinstance(k, (bool, np.bool_)) or int(k) != k or k < 1:
            raise ValueError('positive integer heterogeneous-neighbor count required')
        if not np.isfinite(beta) or beta < 1: raise ValueError('penalty beta>=1 required')
        if variance_ddof not in (0, 1) or singleton_variance not in ('zero', 'error'):
            raise ValueError('declared variance and singleton conventions required')
        if not np.isfinite(zero_variance_lambda) or zero_variance_lambda <= 0:
            raise ValueError('strictly positive zero-variance lambda required')
        variances = []
        for c, size in zip(classes, sizes):
            if size <= variance_ddof:
                if singleton_variance == 'error': raise ValueError('sample variance undefined for singleton class')
                variances.append(np.zeros(self.A.shape[1]))
            else: variances.append(np.var(self.A[self.y == c], axis=0, ddof=variance_ddof))
        variances = np.array(variances)
        self.denominator = np.where(variances == 0, zero_variance_lambda, variances)[inverse]
        self.same = self.y[:, None] == self.y[None, :]
        self.enemy_counts = len(self.A)-sizes[inverse]
        self.k, self.beta = int(k), float(beta)
        self.conventions = {'k': int(k), 'beta': float(beta), 'variance_ddof': variance_ddof,
            'zero_variance_lambda': float(zero_variance_lambda), 'singleton_variance': singleton_variance,
            'zero_variance_lambda_applied': 'only_exactly_zero_class_variances',
            'distance_denominator': 'target_object_class_variance',
            'support': 'strict_same_class_distance_less_than_enemy_distance_including_self',
            'distance_ties': 'stable_original_object_index', 'argmax_ties': 'first_enemy_rank',
            'insufficient_enemies': 'truncate_k_per_object', 'distance_comparison': 'log_relation_domain',
            'feature_distance_accumulation': 'canonical_ascending_original_feature_index',
            'dependency': 'object_mean_not_class_balanced',
            'scope': 'published_formulas_with_declared_numerical_conventions_not_author_tables'}

    def feature_distance(self, a):
        if isinstance(a, (bool, np.bool_)) or int(a) != a or not 0 <= a < self.A.shape[1]:
            raise ValueError('valid feature index required')
        values = self.A[:, int(a)]
        D = (values[:, None]-values[None, :])**2/self.denominator[None, :, int(a)]
        if not np.isfinite(D).all(): raise ValueError('directed distances overflowed')
        return D

    def distances(self, features):
        fs = list(features)
        if len(fs) != len(set(fs)): raise ValueError('distinct features required')
        D = np.zeros((len(self.A), len(self.A)))
        for a in sorted(fs): D += self.feature_distance(a)
        return D

    def profile(self, D):
        D = np.asarray(D, float)
        if D.shape != self.same.shape or not np.isfinite(D).all() or D.min() < 0:
            raise ValueError('finite nonnegative directed distance matrix required')
        width = min(self.k, int(self.enemy_counts.max()))
        ordered = np.argsort(np.where(self.same, np.inf, D), axis=1, kind='stable')[:, :width]
        values = D[np.arange(len(D))[:, None], ordered]
        valid = np.arange(width)[None, :] < self.enemy_counts[:, None]
        supports = np.sum(self.same[:, :, None] & (D[:, :, None] < values[:, None, :]), axis=1)
        scores = np.where(valid, supports-self.beta*np.arange(width)[None, :], -np.inf)
        ranks = scores.argmax(axis=1)
        witness = ordered[np.arange(len(D)), ranks]
        lower = -np.expm1(-D[np.arange(len(D)), witness])
        return {'dependency': float(lower.mean()), 'own_class_lower': lower.tolist(),
                'witness_indices': witness.tolist(), 'witness_enemy_ranks': (ranks+1).tolist(),
                'candidate_support_counts': supports.tolist(), 'candidate_valid': valid.tolist()}

    def dependency(self, features):
        return self.profile(self.distances(features))['dependency']

    def forward(self, *, delta, requested_count=None):
        """Native gain stopping, or explicitly labelled fixed-count adaptation."""
        d = self.A.shape[1]
        if not np.isfinite(delta) or delta < 0: raise ValueError('nonnegative delta required')
        if requested_count is not None and (isinstance(requested_count, (bool, np.bool_))
                or int(requested_count) != requested_count or not 0 <= requested_count <= d):
            raise ValueError('requested_count must be an integer in [0,d]')
        selected, remaining, trace = [], list(range(d)), []
        value = 0.
        limit = d if requested_count is None else int(requested_count)
        status = 'attributes_exhausted' if requested_count is None else 'requested_count_reached'
        while remaining and len(selected) < limit:
            candidates = []
            for a in remaining:
                dependency = self.profile(self.distances(selected+[a]))['dependency']
                candidates.append({'feature': a, 'dependency': dependency, 'gain': dependency-value})
            chosen = min(candidates, key=lambda r: (-r['gain'], r['feature']))
            accepted = requested_count is not None or chosen['gain'] > delta
            trace.append({'selected_before': selected.copy(), 'base_dependency': value,
                          'candidates': candidates, 'chosen_feature': chosen['feature'], 'accepted': accepted})
            if not accepted:
                status = 'gain_not_above_delta'; break
            a = chosen['feature']; selected.append(a); remaining.remove(a)
            value = chosen['dependency']
        return selected, {'conventions': self.conventions, 'delta': float(delta), 'requested_count': requested_count,
            'mode': 'native_gain_stopping' if requested_count is None else 'fixed_count_adaptation',
            'candidate_evaluation': 'evaluate_all_then_choose_best', 'feature_ties': 'smallest_original_index',
            'empty_subset_dependency': 0., 'status': status, 'trace': trace, 'final_dependency': value}
