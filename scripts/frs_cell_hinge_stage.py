"""Add one FRS-constrained family equally; preserve all fourteen old choices."""
import numpy as np
from third_enemy_hinge_stage import PredictiveMetricStage as OldStage, GRID as OLD_GRID
from fit_frs_cell_hinge import fit_frs_cell_target
from frs_class_cell import build_class_cell, evaluate_class_cell

GRID = OLD_GRID+tuple(('frs_cell_constrained', t) for t in (0., .05, .1, .25, .5, .75, 1.))


class PredictiveMetricStage(OldStage):
    def __init__(self, A, y, previous_stage=None):
        super().__init__(A, y)
        self.cell_checks, self.cell_lookup = [], {}
        if previous_stage is not None:
            self.learners = list(previous_stage['learners'])
            self.lookup = {(tuple(r['features']), tuple(r['anchor']), r['family']): i
                           for i, r in enumerate(self.learners)}
            for check in previous_stage['certificate_checks']:
                key = (tuple(check['features']), tuple(check['weights']))
                self.certificates[key] = {k: v for k, v in check.items() if k not in ('features', 'weights')}

    def cell_learner(self, base, certified):
        fs, anchor = base['features'], np.asarray(base['weights'])
        key = (tuple(fs), tuple(anchor), 'frs_cell_constrained', bool(certified))
        if key not in self.lookup:
            _, _, probe = build_class_cell(self.A, self.y, fs, anchor, self.full)
            reference = self.full.copy(); scope = 'original_full'
            if not probe['anchor_strictly_feasible'] and not certified:
                reference = np.zeros(self.d); reference[fs] = anchor; scope = 'own_anchor'
            if not probe['anchor_strictly_feasible'] and certified:
                target = anchor.copy()
                fit = {'status': 'original_full_cell_unavailable_anchor_retained', 'optimizer_success': True,
                    'subproblem_failures': 0, 'cell_unavailable': True, 'reference_scope': 'original_full',
                    'class_cell': probe, 'weights': anchor.tolist(), 'scope': 'no_cell_search_not_optimizer_success_evidence'}
            else:
                target, fit = fit_frs_cell_target(self.A, self.y, fs, anchor,
                        full_reference=reference, reference_scope=scope)
                fit['cell_unavailable'] = False
            self.lookup[key] = len(self.learners)
            self.learners.append({'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit,
                'family': 'frs_cell_constrained', 'certified_arm': bool(certified), 'original_cell_probe': probe})
        return self.lookup[key]

    def cell_guard(self, fs, request, lid):
        key = (tuple(fs), tuple(request), lid)
        if key not in self.cell_lookup:
            old = self.guard(fs, request); fit = self.learners[lid]['fit']
            assert fit['reference_scope'] == 'original_full'
            evaluation = evaluate_class_cell(fit['class_cell'], request)
            cell_accept = bool(not fit['cell_unavailable'] and evaluation['feasible'] and old['certificate']['majority_feasible'])
            record = {'id': len(self.cell_checks), 'features': fs, 'weights': request.tolist(), 'learner_id': lid,
                'old_guard_id': old['id'], 'cell_evaluation': evaluation, 'cell_accepted': cell_accept,
                'accepted': bool(old['accepted'] or cell_accept), 'old_accepted': bool(old['accepted'])}
            self.cell_lookup[key] = len(self.cell_checks); self.cell_checks.append(record)
        return self.cell_checks[self.cell_lookup[key]]

    def metric(self, base, target_index, *, certified=False, weighted=True):
        if target_index < len(OLD_GRID):
            return {**super().metric(base, target_index, certified=certified, weighted=weighted),
                    'cell_guard_id': None, 'cell_deployed_guard_id': None}
        assert weighted
        family, t = GRID[target_index]; fs, anchor = base['features'], np.asarray(base['weights'])
        lid = self.cell_learner(base, certified); request = (1-t)*anchor+t*np.asarray(self.learners[lid]['target'])
        guard = self.cell_guard(fs, request, lid) if certified else None
        fallback = bool(certified and not guard['accepted'])
        deployed = self.cell_guard(fs, anchor, lid) if fallback else guard
        if certified:
            assert deployed['accepted']
        return {'features': fs, 'weights': (anchor if fallback else request).tolist(), 'base_origin': base['origin'],
            'learner_id': lid, 'target_index': target_index, 'target_family': family, 'interpolation': t,
            'requested_weights': request.tolist(), 'guard_id': None if guard is None else guard['old_guard_id'],
            'deployed_guard_id': None if deployed is None else deployed['old_guard_id'], 'guard_fallback': fallback,
            'cell_guard_id': None if guard is None else guard['id'],
            'cell_deployed_guard_id': None if deployed is None else deployed['id']}

    def export(self):
        return {**super().export(), 'cell_certificate_checks': self.cell_checks}
