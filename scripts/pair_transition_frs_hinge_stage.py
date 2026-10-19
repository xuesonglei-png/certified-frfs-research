"""Append seven residual-pair choices; old audited metrics are reused verbatim.

Only new indices can be computed here. Inherited learners, guards and every
prediction candidate stay immutable, avoiding accidental old-family refits.
"""
import copy
import numpy as np
from predictive_target_knn_stage import PredictiveMetricStage as GuardStage
from monotone_frs_hinge_stage import GRID as OLD_GRID
from fit_pair_transition_frs_hinge import fit_pair_transition_target
from frs_pair_transition_cell import build_class_cell, evaluate_class_cell

GRID = OLD_GRID+tuple(('pair_transition_constrained', t) for t in (0., .05, .1, .25, .5, .75, 1.))


class PredictiveMetricStage(GuardStage):
    def __init__(self, A, y, previous_stage):
        super().__init__(A, y)
        self.previous = copy.deepcopy(previous_stage)
        self.learners = copy.deepcopy(previous_stage['learners'])
        self.pair_checks, self.pair_check_lookup, self.pair_learner_lookup = [], {}, {}
        for index, check in enumerate(previous_stage['certificate_checks']):
            assert check['id'] == index
            key = (tuple(check['features']), tuple(check['weights']))
            assert key not in self.certificates
            self.certificates[key] = copy.deepcopy({k:v for k,v in check.items() if k not in ('features','weights')})

    def pair_learner(self, base, certified):
        fs, anchor = base['features'], np.asarray(base['weights'])
        key = (tuple(fs), tuple(anchor), bool(certified))
        if key not in self.pair_learner_lookup:
            _, _, probe = build_class_cell(self.A, self.y, fs, anchor, self.full)
            reference, scope = self.full.copy(), 'original_full'
            if not probe['anchor_strictly_feasible'] and not certified:
                reference = np.zeros(self.d); reference[fs] = anchor; scope = 'own_anchor'
            if not probe['anchor_strictly_feasible'] and certified:
                target = anchor.copy()
                fit = {'status':'original_full_cell_unavailable_anchor_retained', 'optimizer_success':None,
                    'subproblem_failures':0, 'cell_unavailable':True, 'reference_scope':'original_full',
                    'class_cell':probe, 'weights':anchor.tolist(), 'scope':'no_search_not_solver_success_evidence'}
            else:
                target, fit = fit_pair_transition_target(self.A, self.y, fs, anchor,
                    full_reference=reference, reference_scope=scope)
                fit['cell_unavailable'] = False
            self.pair_learner_lookup[key] = len(self.learners)
            self.learners.append({'features':fs, 'anchor':anchor.tolist(), 'target':target.tolist(), 'fit':fit,
                'family':'pair_transition_constrained', 'certified_arm':bool(certified), 'original_cell_probe':probe})
        return self.pair_learner_lookup[key]

    def pair_guard(self, fs, request, lid):
        key = (tuple(fs), tuple(request), lid)
        if key not in self.pair_check_lookup:
            old = self.guard(fs, request); fit = self.learners[lid]['fit']
            assert fit['reference_scope'] == 'original_full'
            value = evaluate_class_cell(fit['class_cell'], request)
            cell_accept = bool(not fit['cell_unavailable'] and value['feasible'] and old['certificate']['majority_feasible'])
            report = {'id':len(self.pair_checks), 'features':fs, 'weights':request.tolist(), 'learner_id':lid,
                'old_guard_id':old['id'], 'cell_evaluation':value, 'cell_accepted':cell_accept,
                'accepted':bool(old['accepted'] or cell_accept), 'old_accepted':old['accepted']}
            self.pair_check_lookup[key] = len(self.pair_checks); self.pair_checks.append(report)
        return self.pair_checks[self.pair_check_lookup[key]]

    def metric(self, base, target_index, *, certified=False, weighted=True):
        if not weighted or not len(OLD_GRID) <= target_index < len(GRID):
            raise ValueError('Only new weighted indices can be computed; reuse all old metrics verbatim')
        family, t = GRID[target_index]; fs, anchor = base['features'], np.asarray(base['weights'])
        lid = self.pair_learner(base, certified); request = (1-t)*anchor+t*np.asarray(self.learners[lid]['target'])
        guard = self.pair_guard(fs, request, lid) if certified else None
        fallback = bool(certified and not guard['accepted'])
        deployed = self.pair_guard(fs, anchor, lid) if fallback else guard
        if certified:
            assert deployed['accepted']
        return {'features':fs, 'weights':(anchor if fallback else request).tolist(), 'base_origin':base['origin'],
            'learner_id':lid, 'target_index':target_index, 'target_family':family, 'interpolation':t,
            'requested_weights':request.tolist(), 'guard_id':None if guard is None else guard['old_guard_id'],
            'deployed_guard_id':None if deployed is None else deployed['old_guard_id'], 'guard_fallback':fallback,
            'pair_guard_id':None if guard is None else guard['id'],
            'pair_deployed_guard_id':None if deployed is None else deployed['id']}

    def export(self):
        inherited = copy.deepcopy(self.previous)
        return {**inherited, **super().export(), 'pair_certificate_checks':self.pair_checks}
