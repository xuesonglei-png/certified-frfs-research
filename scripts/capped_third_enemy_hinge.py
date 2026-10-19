"""Optimize the actual bounded finite-box error surrogate, not its excess.

min(1,S3-S2) = S3-max(S2,S3-1), a difference of convex functions.
CCCP remains local and flat capped regions can prevent strict descent.
"""
import numpy as np
from third_enemy_hinge import ThirdEnemyHinge


class CappedThirdEnemyHinge(ThirdEnemyHinge):
    def losses(self, w):
        return np.minimum(1., super().losses(w))

    def minorant_at(self, w):
        two, f2 = self.top_sum(w, 2); three, f3 = self.top_sum(w, 3)
        cap_active = three-1. > two  # Exact ties keep the top2 branch.
        faces = [f3[i] if flag else f2[i] for i, flag in enumerate(cap_active)]
        shifts = cap_active.astype(int).tolist()
        return np.maximum(two, three-1.), faces, shifts

    def export(self):
        return {**super().export(), 'decomposition': 'S3_minus_max_of_S2_and_S3_minus1',
            'optimized_loss': 'clipped_third_enemy_hinge_equals_finite_training_error_upper_surrogate',
            'scope': 'bounded_DC_training_surrogate_not_population_risk_global_optimum_or_test_advantage'}
