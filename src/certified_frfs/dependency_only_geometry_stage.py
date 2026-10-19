"""Explicit dependency-only certificate search; majority guarantee removed."""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext
from .fast_forest_projection import ForestProjectionLoss


class DependencyOnlyGeometryContext(GeometryStageContext):
    def fit_policy(self,*,C=1.,pair_penalty=1.,target_count=None,fraction=.8):
        if target_count is None: target_count = int(np.ceil(fraction*self.d))
        if int(target_count) != target_count or not int(np.ceil(self.d/2)) <= target_count <= self.d:
            raise ValueError('target outside certified search budget [.5d,d]')
        full = list(range(self.d)); teacher = self.classifier(full,C,pair_penalty)
        projection = ForestProjectionLoss(self.A,self.y,self.classes,teacher)
        initial = self.candidate(full); assert initial['certificate']['dependency_certificate']['upper_bound'] <= .1
        trace = [{**initial,'projection_loss':projection(full),'dependency_only_admissible':True}]; attempts = []; rankings = []; active = full
        while len(active) > target_count:
            ordered = sorted([(projection([a for a in active if a != removed]),removed) for removed in active],key=lambda v:(round(v[0],12),v[1]))
            rankings.append({'features':active.copy(),'choices':[{'removed':int(j),'loss':value} for value,j in ordered]})
            accepted = None
            for value,removed in ordered:
                candidate = self.candidate([a for a in active if a != removed])
                admissible = candidate['certificate']['dependency_certificate']['upper_bound'] <= .1
                attempts.append({**candidate,'removed':int(removed),'projection_loss':value,
                    'dependency_only_admissible':bool(admissible),'accepted':bool(admissible)})
                if admissible:
                    accepted = {**candidate,'projection_loss':value,'dependency_only_admissible':True}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        selected = trace[-1]
        return {'C':float(C),'pair_penalty':pair_penalty,'target_count':int(target_count),'candidate':selected,
            'fit':self.classifier(selected['features'],C,pair_penalty),'path':{'trace':trace,'attempts':attempts,
                'rankings':rankings,'center':projection.center.tolist(),'tie_decimals':12},
            'majority_guard_enforced':False,'teacher_guards_enforced':False,
            'scope':'same_full_reference_FRS_dependency_loss_certificate_only_no_majority_retention_or_classifier_guarantee'}
