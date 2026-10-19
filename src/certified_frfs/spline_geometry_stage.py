"""Capacity-aware support search under the unchanged uniform FRS reference."""
import numpy as np
from sklearn.metrics import mutual_info_score
from .geometry_only_forest_stage import GeometryStageContext
from .spline_forest import fit_spline_forest,spline_memberships,parameters

KNOTS = (0.,.25,.5,.75,1.)


def spline_interaction_tree(X,y,knots=KNOTS):
    X,y = np.asarray(X,float),np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] < 1: raise ValueError('invalid training table')
    bins = spline_memberships(X,knots).argmax(axis=-1); k = len(knots); d = X.shape[1]
    single = [float(mutual_info_score(bins[:,a],y)) for a in range(d)]; candidates = []
    for u in range(d):
        for v in range(u+1,d):
            joint = float(mutual_info_score(k*bins[:,u]+bins[:,v],y))
            candidates.append({'edge':[u,v],'joint_MI':joint,'complementarity':max(0.,joint-max(single[u],single[v]))})
    candidates.sort(key=lambda row:(-row['complementarity'],row['edge'])); parents = list(range(d)); chosen = []
    def root(a):
        while parents[a] != a: a = parents[a]
        return a
    for row in candidates:
        u,v = row['edge']; a,b = root(u),root(v)
        if a != b: parents[a] = b; chosen.append(row)
        if len(chosen) == d-1: break
    return {'edges':[r['edge'] for r in chosen],'selected':chosen,'single_MI':single,
        'scope':'train_only_five_state_complementarity_tree_no_causal_or_prediction_guarantee'}


class SplineProjectionLoss:
    """Exact class-centered raw-substitution MSE from a small Gram matrix."""
    def __init__(self,X,y,fit):
        X,knots,Z,W,b,edges,V = parameters(X,fit); y = np.asarray(y); classes = np.array(fit['classes'])
        if y.shape != (len(X),) or not np.all(np.isin(y,classes)) or any(not np.any(y == c) for c in classes):
            raise ValueError('training reference must contain every class')
        q,d = len(classes),X.shape[1]; self.d = d; self.edges = edges
        weights = np.array([1/(q*sum(y == label)) for label in y]); self.center = weights@X
        centerZ = spline_memberships(self.center[None],knots)[0]; D = Z-centerZ[None]
        unary = np.einsum('ndk,qdk->ndq',D,W); interactions = np.zeros((len(X),len(edges),q))
        for e,(u,v) in enumerate(edges):
            unary[:,u] += np.einsum('nk,qkl,l->nq',D[:,u],V[:,e],centerZ[v])
            unary[:,v] += np.einsum('k,qkl,nl->nq',centerZ[u],V[:,e],D[:,v])
            interactions[:,e] = np.einsum('nk,qkl,nl->nq',D[:,u],V[:,e],D[:,v])
        terms = np.concatenate([unary,interactions],axis=1); terms -= terms.mean(axis=2,keepdims=True)
        design = terms.transpose(0,2,1).reshape(len(X)*q,-1)
        self.gram = design.T@(np.repeat(weights/q,q)[:,None]*design); self.cache = {}

    def __call__(self,features):
        fs = tuple(sorted(features))
        if len(set(fs)) != len(fs) or any(int(a) != a or not 0 <= a < self.d for a in fs): raise ValueError('invalid support')
        if fs not in self.cache:
            keep = set(fs); omitted = np.array([a not in keep for a in range(self.d)]+[u not in keep or v not in keep for u,v in self.edges],float)
            self.cache[fs] = max(0.,float(omitted@self.gram@omitted))
        return self.cache[fs]


class SplineGeometryStageContext(GeometryStageContext):
    def tree(self,features):
        fs = tuple(sorted(features))
        if fs not in self.trees:
            local = spline_interaction_tree(self.A[:,fs],self.y)
            self.trees[fs] = {'features':list(fs),'edges':[sorted([fs[u],fs[v]]) for u,v in local['edges']],'local_tree':local}
        return self.trees[fs]

    def classifier(self,features,C=1.,pair_penalty=1.):
        fs = tuple(sorted(features)); key = (fs,float(C),pair_penalty)
        if key not in self.classifiers:
            edges = [] if not fs or pair_penalty is None else self.tree(fs)['edges']
            self.classifiers[key] = fit_spline_forest(self.A,self.y,list(fs),edges,C=C,
                pair_penalty=1. if pair_penalty is None else pair_penalty)
        return self.classifiers[key]

    def fit_policy(self,*,fraction=.8):
        if not np.isfinite(fraction) or not .5 <= fraction <= 1: raise ValueError('fraction outside [.5,1]')
        full = list(range(self.d)); teacher = self.classifier(full)
        projection = SplineProjectionLoss(self.A,self.y,teacher); initial = self.candidate(full)
        if not initial['certificate']['feasible']: raise RuntimeError('original full reference certificate failed')
        trace = [{**initial,'projection_loss':projection(full)}]; active = full; attempts = []; rankings = []
        minimum = int(np.ceil(fraction*self.d))
        while len(active) > minimum:
            choices = sorted([(projection([a for a in active if a != removed]),removed) for removed in active],key=lambda row:(round(row[0],12),row[1]))
            rankings.append({'features':active.copy(),'choices':[{'removed':int(j),'loss':v} for v,j in choices]})
            accepted = None
            for value,removed in choices:
                point = self.candidate([a for a in active if a != removed]); feasible = point['certificate']['feasible']
                attempts.append({**point,'removed':int(removed),'projection_loss':value,'accepted':feasible})
                if feasible: accepted = {**point,'projection_loss':value}; break
            if accepted is None: break
            trace.append(accepted); active = accepted['features']
        point = trace[-1]
        return {'fraction':float(fraction),'candidate':point,'fit':self.classifier(point['features']),
            'teacher':teacher,'path':{'trace':trace,'attempts':attempts,'rankings':rankings,
                'center':projection.center.tolist(),'tie_decimals':12},'teacher_guards_enforced':False,
            'scope':'changed_five_knot_teacher_and_tree_support_search_original_uniform_FRS_certification_no_test_accuracy_guarantee'}
