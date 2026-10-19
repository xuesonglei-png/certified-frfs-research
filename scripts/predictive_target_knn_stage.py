"""Reference-only predictive target and per-mixture original FRS guard."""
import numpy as np
from capped_predictive_metric import fit_predictive_target
from run_certified_metric_knn_pool import evaluate
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate

GRID = (0., .05, .1, .25, .5, .75, 1.)


class PredictiveMetricStage:
    def __init__(self, A, y):
        self.A, self.y = np.asarray(A, float), np.asarray(y)
        self.d = self.A.shape[1]; self.full = np.ones(self.d)/np.sqrt(self.d)
        self.protected = MajorityTerms(self.A, self.y, weights=self.full).bounds(range(self.d), allowed_enemies=2) > 1e-10
        self.learners, self.lookup, self.certificates = [], {}, {}
        self.metrics = []

    def learner(self, base):
        fs, anchor = base['features'], base['weights']; key = (tuple(fs), tuple(anchor))
        if key not in self.lookup:
            target, fit = fit_predictive_target(self.A, self.y, fs, anchor)
            self.lookup[key] = len(self.learners)
            self.learners.append({'features': fs, 'anchor': anchor, 'target': target.tolist(), 'fit': fit})
        return self.lookup[key]

    def guard(self, fs, weights):
        key = (tuple(fs), tuple(weights))
        if key not in self.certificates:
            cert = weight_certificate(self.A, self.y, fs, self.full, weights, protected=self.protected)
            classes = classwise_dependency_certificate(self.y, cert)
            self.certificates[key] = {'id': len(self.certificates), 'certificate': cert, 'class_certificate': classes,
                                      'accepted': cert['feasible'] and classes['feasible_all_classes']}
        return self.certificates[key]

    def metric(self, base, target_index, *, certified=False, weighted=True):
        fs = base['features']; anchor = np.array(base['weights'])
        if not weighted:
            assert target_index == 0
            return {'features': fs, 'weights': anchor.tolist(), 'base_origin': base['origin'],
                    'learner_id': None, 'target_index': 0, 'interpolation': 0., 'guard_id': None, 'deployed_guard_id': None, 'guard_fallback': False}
        lid = self.learner(base); t = GRID[target_index]
        target = np.array(self.learners[lid]['target']); proposed = (1-t)*anchor+t*target
        guard = self.guard(fs, proposed) if certified else None
        fallback = certified and not guard['accepted']
        deployed = self.guard(fs, anchor) if fallback else guard
        if fallback: assert deployed['accepted']
        weights = anchor if fallback else proposed
        return {'features': fs, 'weights': weights.tolist(), 'base_origin': base['origin'],
                'learner_id': lid, 'target_index': target_index, 'interpolation': t,
                'requested_weights': proposed.tolist(), 'guard_id': None if guard is None else guard['id'],
                'deployed_guard_id': None if deployed is None else deployed['id'], 'guard_fallback': bool(fallback)}

    def export(self):
        return {'learners': self.learners, 'certificate_checks': [{'features': list(fs), 'weights': list(w), **cert}
                for (fs, w), cert in self.certificates.items()]}


class PredictionRegistry:
    def __init__(self, stage, B, truth):
        self.stage, self.B, self.truth = stage, B, truth
        self.lookup, self.predictions = {}, []

    def score(self, metric):
        key = (tuple(metric['features']), tuple(metric['weights']))
        if key not in self.lookup:
            self.lookup[key] = len(self.predictions)
            self.predictions.append(evaluate(self.stage.A, self.stage.y, self.B, self.truth, metric['features'], np.array(metric['weights'])))
        return self.lookup[key]
