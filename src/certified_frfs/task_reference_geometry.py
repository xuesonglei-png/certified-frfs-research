"""Train-only unit-mass reference relations with separate FRS certification.

The reference teacher is fixed at C=1/pair penalty=1. Changing its metric
changes the full reference relation; certificates from uniform sqrt(d) mass
are never valid cache entries for this context.
"""
import numpy as np
from .geometry_only_forest_stage import GeometryStageContext
from .forest_lipschitz_metric import forest_contrast_lipschitz
from .majority_metric import MajorityTerms
from .redistributed_loss import transfer_weights
from .capped_sparse import capped_sparse_projection
from .feasible_transport import weight_certificate
from .fast_forest_projection import ForestProjectionLoss


class TaskReferenceGeometryContext(GeometryStageContext):
    def __init__(self, raw, y, *, family='task_unit'):
        if family not in ('task_unit', 'uniform_unit'):
            raise ValueError('unknown reference family')
        super().__init__(raw, y)
        self.family = family
        self.reference_teacher = self.classifier(list(range(self.d)), 1., 1.)
        self.sensitivity = forest_contrast_lipschitz(self.reference_teacher)
        self.mass = 1.
        self.original_weights = (np.array(self.sensitivity['unit_mass_weights'])
            if family == 'task_unit' else np.ones(self.d)/self.d)
        self.cap = max(float(self.original_weights.max()), 2/self.d)
        self.protected = MajorityTerms(self.A, self.y, weights=self.original_weights).bounds(
            range(self.d), allowed_enemies=2) > 1e-10
        # No old-reference geometry is seeded or inherited.
        self.geometry = {}

    def candidate(self, features):
        fs = tuple(sorted(features))
        if len(fs) < int(np.ceil(self.d/2)) or len(set(fs)) != len(fs) or any(a not in range(self.d) for a in fs):
            raise ValueError('support outside declared half-dimensional search')
        if fs not in self.geometry:
            weights = self.original_weights.copy() if len(fs) == self.d else capped_sparse_projection(
                transfer_weights(self.A, fs, self.original_weights), len(fs), self.cap)
            certificate = weight_certificate(self.A, self.y, fs, self.original_weights, weights,
                protected=self.protected)
            embedded = np.zeros(self.d); embedded[list(fs)] = weights
            self.geometry[fs] = {'features':list(fs), 'weights':embedded.tolist(), 'certificate':certificate}
        return self.geometry[fs]

    def fit_policy(self, *, fraction=.8, target_count=None):
        if not np.isfinite(fraction) or not .5 <= fraction <= 1:
            raise ValueError('fraction outside [.5,1]')
        minimum = int(np.ceil(fraction*self.d)) if target_count is None else target_count
        if int(minimum) != minimum or not int(np.ceil(self.d/2)) <= minimum <= self.d:
            raise ValueError('target count outside [.5d,d]')
        minimum = int(minimum); full = list(range(self.d))
        projection = ForestProjectionLoss(self.A, self.y, self.classes, self.reference_teacher)
        initial = self.candidate(full)
        if not initial['certificate']['feasible']:
            raise RuntimeError('new full reference failed its own certificate')
        trace = [{**initial,'projection_loss':projection(full)}]; attempts = []; rankings = []; active = full
        while len(active) > minimum:
            ranked = sorted([(projection([a for a in active if a != removed]),removed) for removed in active],
                key=lambda row:(round(row[0],12),row[1]))
            rankings.append({'features':active.copy(), 'choices':[{'removed':int(j),'loss':v} for v,j in ranked]})
            accepted = None
            for cost,removed in ranked:
                candidate = self.candidate([a for a in active if a != removed])
                admissible = candidate['certificate']['feasible']
                attempts.append({**candidate,'removed':int(removed),'projection_loss':cost,'accepted':admissible})
                if admissible:
                    accepted = {**candidate,'projection_loss':cost}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        selected = trace[-1]
        return {'family':self.family, 'C':1., 'pair_penalty':1., 'fraction':float(fraction),
            'target_count':minimum, 'reference_weights':self.original_weights.tolist(), 'cap':self.cap,
            'sensitivity':self.sensitivity, 'reference_teacher':self.reference_teacher,
            'protected_mask':self.protected.tolist(), 'candidate':selected,
            'fit':self.classifier(selected['features'],1.,1.),
            'path':{'trace':trace,'attempts':attempts,'rankings':rankings,
                'center':projection.center.tolist(),'tie_decimals':12},
            'scope':'changed_train_only_unit_mass_full_reference_FRS_certification_separate_refitted_classifier_no_population_guarantee'}
