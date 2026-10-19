"""Certified candidates ranked by actual refitted crossfit classification loss."""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext
from .refitted_crossfit_forest_risk import RefittedCrossfitForestRisk


class RefittedRiskGeometryContext(GeometryStageContext):
    def __init__(self,raw,y):
        super().__init__(raw,y); self.raw = np.asarray(raw,float)

    def fit_policy(self,*,C=1.,pair_penalty=1.,fraction=.8):
        if not np.isfinite(fraction) or not .5 <= fraction <= 1: raise ValueError('fraction outside [.5,1]')
        risk = RefittedCrossfitForestRisk(self.raw,self.y,C=C,pair_penalty=pair_penalty)
        full = list(range(self.d)); initial = self.candidate(full); assert initial['certificate']['feasible']
        trace = [{**initial,'risk_score_id':len(risk.scores)}]; risk(full)
        active = full; attempts = []; rankings = []; minimum = int(np.ceil(fraction*self.d))
        while len(active) > minimum:
            feasible = []
            for removed in active:
                candidate = self.candidate([a for a in active if a != removed])
                attempt = {**candidate,'removed':int(removed),'risk_score_id':None,'accepted':False}
                if candidate['certificate']['feasible']:
                    score = risk(candidate['features']); attempt['risk_score_id'] = risk.cache[tuple(candidate['features'])]
                    feasible.append((score['mean_balanced_logloss'],removed,len(attempts)))
                attempts.append(attempt)
            ranked = sorted(feasible,key=lambda r:(round(r[0],12),r[1]))
            rankings.append({'features':active.copy(),'choices':[{'removed':int(j),'loss':value,'attempt_id':ai} for value,j,ai in ranked]})
            if not ranked: break
            _,_,ai = ranked[0]; attempts[ai]['accepted'] = True; accepted = attempts[ai]
            active = accepted['features']
            trace.append({k:v for k,v in accepted.items() if k not in ('removed','accepted')})
        selected = trace[-1]; fit = self.classifier(selected['features'],C,pair_penalty)
        return {'C':float(C),'pair_penalty':pair_penalty,'fraction':float(fraction),'candidate':selected,'fit':fit,
            'path':{'trace':trace,'attempts':attempts,'rankings':rankings,'tie_decimals':12},
            'risk':{'folds':risk.records,'models':risk.models,'scores':risk.scores,'seed':20261014},
            'teacher_guards_enforced':False,
            'scope':'adaptive_training_only_actual_classifier_refit_risk_ranking_with_full_reference_FRS_screen_not_unbiased_risk_or_population_bound'}
