"""Certified support paths relative to a learned full reference relation.

This changes the reference fuzzy relation explicitly. It does NOT preserve
the previous uniform relation. The existing general nonnegative-weight FRS
certificate is applied without relaxing its radius/retention/loss thresholds.
"""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext
from .majority_metric import MajorityTerms, learn_majority_weights
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate


class LearnedReferenceGeometryContext(GeometryStageContext):
    def __init__(self, raw, y):
        super().__init__(raw, y)
        self.original_weights = learn_majority_weights(self.A, self.y, eps=.01, reg=.01)
        self.mass = float(self.original_weights.sum())
        self.cap = max(float(self.original_weights.max()), 2*self.mass/self.d)
        self.protected = MajorityTerms(self.A, self.y, weights=self.original_weights).bounds(range(self.d), allowed_enemies=2) > 1e-10

    def candidate(self, features):
        fs = tuple(sorted(features))
        if fs not in self.geometry:
            weights = self.original_weights.copy() if len(fs) == self.d else capped_sparse_projection(
                transfer_weights(self.A, fs, self.original_weights)/self.mass, len(fs), self.cap/self.mass)*self.mass
            certificate = weight_certificate(self.A, self.y, fs, self.original_weights, weights, protected=self.protected)
            embedded = np.zeros(self.d); embedded[list(fs)] = weights
            self.geometry[fs] = {'features': list(fs), 'weights': embedded.tolist(), 'certificate': certificate}
        return self.geometry[fs]

    def fit_stage(self, *, C=1., pair_penalty=1.):
        stage = super().fit_stage(C=C, pair_penalty=pair_penalty)
        return {**stage, 'reference_metric': {'weights': self.original_weights.tolist(), 'mass': self.mass, 'cap': self.cap,
            'eps': .01, 'reg': .01,
            'scope': 'learned_robust_majority_full_reference_relation_with_optimizer_uniform_fallback_not_preserved_uniform_relation'},
            'scope': 'learned_reference_relation_general_weighted_FRS_certificate_common_classifier_no_population_guarantee'}
