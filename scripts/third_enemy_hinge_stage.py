"""Same DC learner and14choices for every weighted comparison arm."""
import numpy as np
from predictive_target_knn_stage import PredictiveMetricStage as ParentStage
from fit_third_enemy_hinge import fit_third_enemy_target

GRID = tuple((family, t) for family in ('third_enemy_unconstrained', 'third_enemy_constrained')
             for t in (0., .05, .1, .25, .5, .75, 1.))


class PredictiveMetricStage(ParentStage):
    def learner(self, base, family):
        fs, anchor = base['features'], base['weights']; key = (tuple(fs), tuple(anchor), family)
        if key not in self.lookup:
            target, fit = fit_third_enemy_target(self.A, self.y, fs, anchor, constrained=family == 'third_enemy_constrained')
            self.lookup[key] = len(self.learners)
            self.learners.append({'features': fs, 'anchor': anchor, 'target': target.tolist(), 'fit': fit, 'family': family})
        return self.lookup[key]

    def metric(self, base, target_index, *, certified=False, weighted=True):
        fs, anchor = base['features'], np.asarray(base['weights'])
        if not weighted:
            assert target_index == 0
            return {'features': fs, 'weights': anchor.tolist(), 'base_origin': base['origin'],
                    'learner_id': None, 'target_index': 0, 'target_family': None, 'interpolation': 0.,
                    'guard_id': None, 'deployed_guard_id': None, 'guard_fallback': False}
        family, t = GRID[target_index]; lid = self.learner(base, family)
        request = (1-t)*anchor+t*np.asarray(self.learners[lid]['target'])
        guard = self.guard(fs, request) if certified else None
        fallback = bool(certified and not guard['accepted'])
        deployed = self.guard(fs, anchor) if fallback else guard
        if fallback:
            assert deployed['accepted']
        return {'features': fs, 'weights': (anchor if fallback else request).tolist(), 'base_origin': base['origin'],
                'learner_id': lid, 'target_index': target_index, 'target_family': family, 'interpolation': t,
                'requested_weights': request.tolist(), 'guard_id': None if guard is None else guard['id'],
                'deployed_guard_id': None if deployed is None else deployed['id'], 'guard_fallback': fallback}
