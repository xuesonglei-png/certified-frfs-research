"""FRS-certified support generation with a separate downstream classifier.

No teacher-margin guarantee is imposed or claimed. Every support still passes
the original FRS dependency/majority certificate. This stage has no query API.
"""
import numpy as np
from .validated_pool import preprocess_fit
from .fuzzy_interaction_tree import interaction_tree
from .orthogonal_forest_logistic import fit_orthogonal_forest
from .fast_forest_projection import ForestProjectionLoss
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate

FRACTIONS = (.5, .65, .8, 1.)


class GeometryStageContext:
    def __init__(self, raw, y):
        self.A, _ = preprocess_fit(np.asarray(raw, float), np.asarray(raw, float))
        self.y = np.asarray(y); self.classes = np.unique(self.y); self.d = self.A.shape[1]
        self.mass = np.sqrt(self.d); self.original_weights = np.ones(self.d)/self.mass
        self.protected = MajorityTerms(self.A, self.y, weights=self.original_weights).bounds(range(self.d), allowed_enemies=2) > 1e-10
        self.geometry = {}; self.trees = {}; self.classifiers = {}

    def candidate(self, features):
        fs = tuple(sorted(features))
        if fs not in self.geometry:
            weights = self.original_weights.copy() if len(fs) == self.d else capped_sparse_projection(
                transfer_weights(self.A, fs, self.original_weights)/self.mass, len(fs), 2/self.d)*self.mass
            certificate = weight_certificate(self.A, self.y, fs, self.original_weights, weights, protected=self.protected)
            embedded = np.zeros(self.d); embedded[list(fs)] = weights
            self.geometry[fs] = {'features': list(fs), 'weights': embedded.tolist(), 'certificate': certificate}
        return self.geometry[fs]

    def tree(self, features):
        fs = tuple(sorted(features))
        if fs not in self.trees:
            local = interaction_tree(self.A[:, fs], self.y)
            edges = [sorted([fs[u], fs[v]]) for u, v in local['edges']]
            self.trees[fs] = {'features': list(fs), 'edges': edges, 'local_tree': local}
        return self.trees[fs]

    def classifier(self, features, C, pair_penalty):
        fs = tuple(sorted(features)); key = (fs, float(C), pair_penalty)
        if key not in self.classifiers:
            edges = [] if not fs or pair_penalty is None else self.tree(fs)['edges']
            self.classifiers[key] = fit_orthogonal_forest(self.A, self.y, list(fs), edges,
                C=C, pair_penalty=1. if pair_penalty is None else pair_penalty)
        return self.classifiers[key]

    def fit_stage(self, *, C=1., pair_penalty=1.):
        full = list(range(self.d)); full_fit = self.classifier(full, C, pair_penalty)
        projection = ForestProjectionLoss(self.A, self.y, self.classes, full_fit)
        initial = self.candidate(full)
        if not initial['certificate']['feasible']: raise RuntimeError('original full geometry failed')
        trace = [{**initial, 'projection_loss': projection(full)}]; attempts = []; rankings = []; active = full
        while len(active) > int(np.ceil(self.d/2)):
            ranked = [(projection([a for a in active if a != removed]), removed) for removed in active]
            ranked.sort(key=lambda r: (round(r[0], 12), r[1]))
            rankings.append({'features': active.copy(), 'choices': [{'removed': int(j), 'loss': v} for v, j in ranked]})
            accepted = None
            for cost, removed in ranked:
                fs = [a for a in active if a != removed]; candidate = self.candidate(fs)
                attempts.append({**candidate, 'removed': int(removed), 'projection_loss': cost,
                                 'accepted': candidate['certificate']['feasible']})
                if candidate['certificate']['feasible']:
                    accepted = {**candidate, 'projection_loss': cost}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        policies = {}
        for fraction in FRACTIONS:
            minimum = int(np.ceil(fraction*self.d))
            candidate = next(r for r in reversed(trace) if len(r['features']) >= minimum)
            fit = self.classifier(candidate['features'], C, pair_penalty)
            policies[str(fraction)] = {'candidate': candidate, 'fit': fit,
                'classifier_tree': None if pair_penalty is None else self.tree(candidate['features']),
                'teacher_guards_enforced': False}
        return {'d': self.d, 'C': float(C), 'pair_penalty': pair_penalty, 'policies': policies,
            'path': {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'center': projection.center.tolist(), 'tie_decimals': 12},
            'original_majority_protected_counts': {str(c): int(np.sum(self.protected & (self.y == c))) for c in self.classes},
            'teacher_guards_enforced': False,
            'scope': 'FRS_geometry_certified_supports_with_separate_uncertified_downstream_classifier_no_test_accuracy_guarantee'}
