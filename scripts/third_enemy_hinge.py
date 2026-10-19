"""Rank-three robust hinge, DC decomposition, and finite-table error bound.

Uses the SAME fixed crossgroup friends/boxes as the previous convex loss.
Two enemy violations may be ignored, as a strict 5NN majority permits them.
No population bound or verified originality is asserted.
"""
import numpy as np
from robust_majority_hinge import RobustMajorityHinge


class ThirdEnemyHinge(RobustMajorityHinge):
    def top_sum(self, w, k):
        if k not in (2, 3):
            raise ValueError('only the declared top2/top3 decomposition is supported')
        w = np.asarray(w, float)
        values, faces = [], []
        for i, row in enumerate(self.rows):
            if row['constant_loss'] is not None:
                value = row['constant_loss'] if k == 3 else 0.
                gradient, intercept, selected, worst, active = np.zeros(len(w)), value, [], [], []
            else:
                C = self.coefficients(i)
                contrast = C @ w
                friend = contrast.argmin(axis=1)
                hinge = np.maximum(0., 1. - contrast[np.arange(len(contrast)), friend] / self.margin)
                selected = np.lexsort((np.asarray(row['enemies']), -hinge))[:k].tolist()
                worst = [int(friend[j]) for j in selected]
                active = [j for j in selected if hinge[j] > 0]
                gradient = -sum((C[j, friend[j]] for j in active), start=np.zeros(len(w))) / self.margin
                intercept, value = float(len(active)), float(hinge[selected].sum())
            values.append(value)
            faces.append({'query': i, 'k': k, 'gradient': gradient.tolist(), 'intercept': intercept,
                          'enemy_indices': selected, 'worst_friend_indices': worst, 'active_enemy_indices': active})
        return np.asarray(values), faces

    def losses(self, w):
        three, _ = self.top_sum(w, 3)
        two, _ = self.top_sum(w, 2)
        difference = three - two
        assert difference.min() >= -1e-10
        return np.maximum(0., difference)

    def objective(self, w, *, reg=.01):
        return float(self.alpha @ self.losses(w) + reg * abs(np.asarray(w) - self.anchor).sum())

    def export(self):
        return {**super().export(), 'bound': 'finite_crossgroup_error_le_clipped_third_largest_enemy_hinge',
                'decomposition': 'sum_top3_hinges_minus_sum_top2_hinges',
                'scope': 'DC_finite_training_box_error_bound_not_population_risk_or_verified_novelty'}
