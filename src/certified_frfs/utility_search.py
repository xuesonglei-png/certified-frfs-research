"""Training utility-directed deletion with optional majority retention.

Each move evaluates all one-attribute deletions using true-label LOO5NN
balanced accuracy, averaged over nominal and one shared perturbed table.
Candidates are tried in utility order until a certified feasible move exists.
All feature fractions come from the same path; no test labels enter search.
"""
import numpy as np
from .core import _inputs,certificate
from .majority_metric import MajorityTerms
from .adaptive_witness import adaptive_bounds


class UtilityWorkspace:
    def __init__(self,X,y,weights,*,eps=.01,seed=101):
        self.X,self.y,self.full,self.w=_inputs(X,y,None,weights)
        if len(self.y)<2 or np.any(self.X<0) or np.any(self.X>1) or not 0<=eps<=1:
            raise ValueError('scaled table with at least two observations required')
        self.labels,self.codes=np.unique(self.y,return_inverse=True)
        self.balance=1/(len(self.labels)*np.bincount(self.codes)[self.codes])
        rng=np.random.default_rng(seed)
        tables=[self.X,np.clip(self.X+rng.uniform(-eps,eps,self.X.shape),0,1)]
        self.components=[np.moveaxis(abs(Z[:,None,:]-Z[None,:,:])*self.w,2,0) for Z in tables]
        self.cache={};self.k=min(5,len(self.y)-1)

    def evaluate(self,features):
        fs=tuple(features)
        if not fs:raise ValueError('empty distance subset')
        if fs not in self.cache:
            scores=[]
            for components in self.components:
                D=components[list(fs)].sum(axis=0);np.fill_diagonal(D,np.inf)
                nn=np.argsort(D,axis=1,kind='stable')[:,:self.k]
                votes=np.stack([(self.codes[nn]==label).sum(axis=1) for label in range(len(self.labels))],axis=1)
                prediction=votes.argmax(axis=1)
                scores.append(float(self.balance@(prediction==self.codes)))
            self.cache[fs]={'clean_LOO':scores[0],'shared_noise_LOO':scores[1],'utility':float(np.mean(scores))}
        return self.cache[fs]


def utility_path(X,y,weights,*,certified=True,retention=.9,eps=.01,seed=101,dependency_eta=None,shared_loss=False,ranking_workspace=None):
    utility=UtilityWorkspace(X,y,weights,eps=eps,seed=seed)
    X,y,w=utility.X,utility.y,utility.w;d=X.shape[1]
    if not 0<retention<=1:raise ValueError('invalid retention')
    if dependency_eta is not None and (not certified or not 0<=dependency_eta<=1):
        raise ValueError('dependency constraint requires certified mode and tolerance in [0,1]')
    current=list(range(d));candidates=[current.copy()];trace=[];bound_cache={}
    fixed=MajorityTerms(X,y,weights=w,eps=eps) if certified else None
    protected=fixed.bounds(current,allowed_enemies=2)>1e-10 if certified else np.zeros(len(y),bool)
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups]
    def certify(fs):
        key=tuple(fs)
        if key not in bound_cache:
            bounds=fixed.bounds(fs,allowed_enemies=2)
            counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
            # A passing fixed witness is sufficient; refresh only when needed.
            refreshed=False
            if any(c<r for c,r in zip(counts,required)):
                bounds=np.maximum(bounds,adaptive_bounds(X,y,fs,w,eps=eps));refreshed=True
                counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
            majority_feasible=all(c>=r for c,r in zip(counts,required))
            detail={'retained':counts,'refreshed':refreshed,'feasible':majority_feasible}
            if dependency_eta is not None:
                detail['majority_feasible']=majority_feasible
                detail['dependency_certificate']=certificate(X,y,fs,weights=w,eps=eps,method='hybrid') if majority_feasible else None
                if shared_loss and majority_feasible and detail['dependency_certificate']['upper_bound']>dependency_eta:
                    from .shared_loss import shared_loss_bound
                    detail['dependency_certificate']=shared_loss_bound(X,y,fs,w,eps=eps)
                detail['feasible']=majority_feasible and detail['dependency_certificate']['upper_bound']<=dependency_eta
            bound_cache[key]=detail
        return bound_cache[key]
    while len(current)>1:
        moves=[]
        for a in current:
            fs=[b for b in current if b!=a];scores=(ranking_workspace or utility).evaluate(fs)
            moves.append({'feature':a,'features':fs,**scores})
        moves.sort(key=lambda r:(-r['utility'],-r['clean_LOO'],r['feature']))
        accepted=None;attempts=[]
        for move in moves:
            detail=certify(move['features']) if certified else {'feasible':True,'retained':[],'refreshed':False}
            attempts.append({**move,**detail})
            if detail['feasible']:accepted=move;break
        trace.append({'current':current.copy(),'ranked_moves':moves,'certification_attempts':attempts,
                      'accepted':accepted['features'] if accepted else None})
        if accepted is None:break
        current=accepted['features'];candidates.append(current.copy())
    result={'candidates':candidates,'trace':trace,'required':required,
        'protected_counts':[int(g.sum()) for g in groups],
        'training_utility_evaluations':len((ranking_workspace or utility).cache),'certificate_evaluations':len(bound_cache),
        'scope':'training_LOO_utility_and_finite_majority_retention_not_test_risk',
        'certified':certified,'eps':eps,'seed':seed}
    if dependency_eta is not None:
        result['dependency_eta']=dependency_eta
        result['scope']='joint_finite_training_dependency_loss_and_majority_retention_not_test_risk'
        if shared_loss:result['shared_loss']=True
    if ranking_workspace is not None:
        result['ranking_scope']='adaptive_training_crossfit_not_validation_or_test'
        result['ranking_splits']=ranking_workspace.splits
    return result


def utility_candidates(X,y,w,*,certified=True,dependency_eta=None,shared_loss=False,ranking_workspace=None,beam_width=1):
    if int(beam_width)!=beam_width or beam_width<1:raise ValueError('positive integer beam width required')
    if beam_width>1:
        from .beam_utility import beam_utility_candidates
        return beam_utility_candidates(X,y,w,width=beam_width,certified=certified,dependency_eta=dependency_eta,shared_loss=shared_loss,ranking_workspace=ranking_workspace)
    from .crossfit_majority import FRACTIONS
    path=utility_path(X,y,w,certified=certified,dependency_eta=dependency_eta,shared_loss=shared_loss,ranking_workspace=ranking_workspace);d=X.shape[1]
    rows=[]
    for fraction in FRACTIONS:
        target=max(1,int(np.ceil(fraction*d)))
        fs=min(path['candidates'],key=lambda f:(abs(len(f)-target),-len(f),tuple(f)))
        rows.append({'config':{'fraction':fraction},'features':fs})
    return rows,path
