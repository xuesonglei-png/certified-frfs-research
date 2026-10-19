"""Group-shrunk teacher projection ranking under unchanged FRS guards."""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext, FRACTIONS
from .grouped_orthogonal_forest import fit_grouped_forest
from .fast_forest_projection import ForestProjectionLoss


class GroupedGeometryStageContext(GeometryStageContext):
    def fit_stage(self, *, C=1., pair_penalty=1., relative_strength=.1):
        full = list(range(self.d))
        edges = [] if pair_penalty is None else self.tree(full)['edges']
        teacher = fit_grouped_forest(self.A, self.y, edges, C=C,
            pair_penalty=1. if pair_penalty is None else pair_penalty, relative_strength=relative_strength)
        if not teacher['usable']:
            fallback = super().fit_stage(C=C, pair_penalty=pair_penalty)
            return {**fallback, 'group_teacher': teacher, 'group_selector_fallback': True}
        projection = ForestProjectionLoss(self.A, self.y, self.classes, teacher)
        initial = self.candidate(full)
        if not initial['certificate']['feasible']: raise RuntimeError('original full geometry failed')
        trace = [{**initial, 'projection_loss': projection(full)}]
        attempts = []; rankings = []; active = full
        while len(active) > int(np.ceil(self.d/2)):
            ranked = sorted([(projection([a for a in active if a != j]), j) for j in active], key=lambda r: (round(r[0], 12), r[1]))
            rankings.append({'features': active.copy(), 'choices': [{'removed': int(j), 'loss': value} for value, j in ranked]})
            accepted = None
            for value, removed in ranked:
                candidate = self.candidate([a for a in active if a != removed])
                attempts.append({**candidate, 'removed': int(removed), 'projection_loss': value, 'accepted': candidate['certificate']['feasible']})
                if candidate['certificate']['feasible']:
                    accepted = {**candidate, 'projection_loss': value}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        policies = {}
        for fraction in FRACTIONS:
            candidate = next(r for r in reversed(trace) if len(r['features']) >= int(np.ceil(fraction*self.d)))
            policies[str(fraction)] = {'candidate': candidate, 'fit': self.classifier(candidate['features'], C, pair_penalty), 'teacher_guards_enforced': False}
        return {'d': self.d, 'C': float(C), 'pair_penalty': pair_penalty, 'policies': policies,
            'path': {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'center': projection.center.tolist(), 'tie_decimals': 12},
            'group_teacher': teacher, 'group_selector_fallback': False, 'teacher_guards_enforced': False,
            'scope': 'group_shrunk_teacher_projection_ranking_FRS_support_certificate_only'}
