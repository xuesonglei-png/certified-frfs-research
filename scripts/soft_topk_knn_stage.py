"""Same two target families and interpolation choices for all weighted arms."""
import numpy as np

from predictive_target_knn_stage import PredictiveMetricStage as ParentStage, PredictionRegistry
from capped_soft_topk_metric import fit_predictive_target
from soft_topk_predictive_metric import fit_majority_constrained_target

GRID = tuple((family, t) for family in ('soft_topk_unconstrained', 'soft_topk_constrained') for t in (0., .05, .1, .25, .5, .75, 1.))


class PredictiveMetricStage(ParentStage):
    def learner(self, base, family):
        fs, anchor = base['features'], base['weights']; key = (tuple(fs), tuple(anchor), family)
        if key not in self.lookup:
            fit_fn = fit_predictive_target if family == 'soft_topk_unconstrained' else fit_majority_constrained_target
            target, fit = fit_fn(self.A, self.y, fs, anchor)
            fit = {**fit, 'neighborhood_k': 5}
            self.lookup[key] = len(self.learners)
            self.learners.append({'features': fs, 'anchor': anchor, 'target': target.tolist(), 'fit': fit, 'family': family})
        return self.lookup[key]

    def metric(self, base, target_index, *, certified=False, weighted=True):
        fs, anchor = base['features'], np.array(base['weights'])
        if not weighted:
            assert target_index == 0
            return {'features': fs, 'weights': anchor.tolist(), 'base_origin': base['origin'],
                    'learner_id': None, 'target_index': 0, 'target_family': None, 'interpolation': 0.,
                    'guard_id': None, 'deployed_guard_id': None, 'guard_fallback': False}
        family, t = GRID[target_index]; lid = self.learner(base, family)
        proposed = (1-t)*anchor+t*np.array(self.learners[lid]['target'])
        guard = self.guard(fs, proposed) if certified else None
        fallback = certified and not guard['accepted']
        deployed = self.guard(fs, anchor) if fallback else guard
        if fallback: assert deployed['accepted']
        weights = anchor if fallback else proposed
        return {'features': fs, 'weights': weights.tolist(), 'base_origin': base['origin'],
                'learner_id': lid, 'target_index': target_index, 'target_family': family, 'interpolation': t,
                'requested_weights': proposed.tolist(), 'guard_id': None if guard is None else guard['id'],
                'deployed_guard_id': None if deployed is None else deployed['id'], 'guard_fallback': bool(fallback)}
