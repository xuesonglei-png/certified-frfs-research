"""Certified two-coordinate jumps when all single-coordinate moves fail.

The same full uniform reference relation and certificate thresholds remain.
The bounded pair pool is heuristic; global reduct optimality is not claimed.
"""
from itertools import combinations
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext, FRACTIONS
from .fast_forest_projection import ForestProjectionLoss


class JointDeletionGeometryContext(GeometryStageContext):
    def fit_stage(self, *, C=1., pair_penalty=1., pair_pool_size=8):
        if int(pair_pool_size) != pair_pool_size or pair_pool_size < 2:
            raise ValueError('pair pool must contain at least two attributes')
        full = list(range(self.d)); full_fit = self.classifier(full,C,pair_penalty)
        projection = ForestProjectionLoss(self.A,self.y,self.classes,full_fit)
        initial = self.candidate(full); assert initial['certificate']['feasible']
        trace = [{**initial,'projection_loss':projection(full)}]; attempts = []; rankings = []
        active = full; minimum = int(np.ceil(self.d/2)); jumps = 0
        while len(active) > minimum:
            singles = sorted([(projection([a for a in active if a != j]),(j,)) for j in active],key=lambda r:(round(r[0],12),r[1]))
            phases = [(1,singles)]
            accepted = None
            for size, ranked in phases:
                rankings.append({'features':active.copy(),'deletion_size':size,
                    'choices':[{'removed':list(r),'loss':value} for value,r in ranked]})
                for value, removed in ranked:
                    candidate = self.candidate([a for a in active if a not in removed])
                    attempts.append({**candidate,'removed':list(removed),'projection_loss':value,'accepted':candidate['certificate']['feasible']})
                    if candidate['certificate']['feasible']:
                        accepted = {**candidate,'projection_loss':value}; jumps += size == 2; break
                if accepted is not None: break
                if size == 1 and len(active)-2 >= minimum:
                    pool = sorted(r[0] for _,r in singles[:int(pair_pool_size)])
                    pairs = sorted([(projection([a for a in active if a not in removed]),removed) for removed in combinations(pool,2)],key=lambda r:(round(r[0],12),r[1]))
                    # The appended phase is evaluated only after every single
                    # has actually failed the unchanged certificate.
                    phases.append((2,pairs))
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        policies = {}
        for f in FRACTIONS:
            candidate = next(r for r in reversed(trace) if len(r['features']) >= int(np.ceil(f*self.d)))
            policies[str(f)] = {'candidate':candidate,'fit':self.classifier(candidate['features'],C,pair_penalty),'teacher_guards_enforced':False}
        return {'d':self.d,'C':float(C),'pair_penalty':pair_penalty,'policies':policies,
            'path':{'trace':trace,'attempts':attempts,'rankings':rankings,'center':projection.center.tolist(),'tie_decimals':12,
                'pair_pool_size':int(pair_pool_size),'accepted_pair_jumps':jumps},'teacher_guards_enforced':False,
            'original_majority_protected_counts':{str(c):int(sum(self.protected & (self.y == c))) for c in self.classes},
            'scope':'same_uniform_reference_FRS_certified_bounded_joint_deletion_search_no_global_optimality_or_population_guarantee'}
