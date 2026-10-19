"""Unchanged FRS guards with out-of-fold log-loss deletion ranking."""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext, FRACTIONS
from .crossfit_forest_risk import CrossfitForestRisk


class CrossfitGeometryStageContext(GeometryStageContext):
    def __init__(self, raw, y):
        super().__init__(raw, y)
        self.raw = np.asarray(raw, float); self.risks = {}

    def fit_stage(self, *, C=1., pair_penalty=1.):
        key = (float(C), pair_penalty)
        if key not in self.risks:
            self.risks[key] = CrossfitForestRisk(self.raw, self.y, C=C, pair_penalty=pair_penalty)
        risk = self.risks[key]; full = list(range(self.d)); initial = self.candidate(full)
        if not initial['certificate']['feasible']: raise RuntimeError('original full geometry failed')
        trace = [{**initial, 'crossfit_balanced_logloss': risk(full)}]
        attempts = []; rankings = []; active = full
        while len(active) > int(np.ceil(self.d/2)):
            ranked = sorted([(risk([a for a in active if a != j]), j) for j in active], key=lambda r: (round(r[0], 12), r[1]))
            rankings.append({'features': active.copy(), 'choices': [{'removed': int(j), 'loss': v} for v, j in ranked]})
            accepted = None
            for value, removed in ranked:
                candidate = self.candidate([a for a in active if a != removed])
                attempts.append({**candidate, 'removed': int(removed), 'crossfit_balanced_logloss': value,
                    'accepted': candidate['certificate']['feasible']})
                if candidate['certificate']['feasible']:
                    accepted = {**candidate, 'crossfit_balanced_logloss': value}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        policies = {}
        for fraction in FRACTIONS:
            candidate = next(r for r in reversed(trace) if len(r['features']) >= int(np.ceil(fraction*self.d)))
            policies[str(fraction)] = {'candidate': candidate, 'fit': self.classifier(candidate['features'], C, pair_penalty),
                'teacher_guards_enforced': False}
        return {'d': self.d, 'C': float(C), 'pair_penalty': pair_penalty, 'policies': policies,
            'path': {'trace': trace, 'attempts': attempts, 'rankings': rankings, 'tie_decimals': 12,
                'crossfit_splits': risk.records, 'singleton_training_only_indices': risk.singleton_indices},
            'teacher_guards_enforced': False,
            'scope': 'reference_internal_crossfit_projection_risk_ranking_FRS_support_certificate_only'}
