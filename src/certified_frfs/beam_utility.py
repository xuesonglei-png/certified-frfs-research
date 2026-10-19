"""Fixed-width backward beam in the same finite certified feasible set.

Beam search explores alternative deletion orders, not global optimality.
All scores are training-only; validation still selects the size/metric.
"""
import numpy as np
from .core import _inputs,certificate
from .majority_metric import MajorityTerms
from .adaptive_witness import adaptive_bounds
from .shared_loss import shared_loss_bound


def beam_utility_path(X,y,weights,*,width=3,certified=True,retention=.9,
                      eps=.01,seed=101,dependency_eta=None,shared_loss=False,ranking_workspace=None):
    from .utility_search import UtilityWorkspace
    X,y,current,w=_inputs(X,y,None,weights)
    if len(y)<2 or np.any(X<0) or np.any(X>1):raise ValueError('scaled table with at least two observations required')
    if int(width)!=width or width<1:raise ValueError('positive integer width required')
    if not 0<retention<=1 or not 0<=eps<=1:raise ValueError('invalid retention or radius')
    if dependency_eta is not None and (not certified or not 0<=dependency_eta<=1):
        raise ValueError('dependency constraint requires certified mode')
    utility=ranking_workspace or UtilityWorkspace(X,y,w,eps=eps,seed=seed)
    full=current.tolist();fixed=MajorityTerms(X,y,weights=w,eps=eps) if certified else None
    protected=fixed.bounds(full,allowed_enemies=2)>1e-10 if certified else np.zeros(len(y),bool)
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(retention*g.sum()-1e-12)) for g in groups];cache={}
    def certify(fs):
        key=tuple(fs)
        if key not in cache:
            if not certified:
                cache[key]={'feasible':True,'retained':[],'refreshed':False}
            else:
                bounds=fixed.bounds(fs,allowed_enemies=2)
                counts=[int(((bounds>1e-10)&g).sum()) for g in groups];refreshed=False
                if any(c<r for c,r in zip(counts,required)):
                    bounds=np.maximum(bounds,adaptive_bounds(X,y,fs,w,eps=eps));refreshed=True
                    counts=[int(((bounds>1e-10)&g).sum()) for g in groups]
                majority=all(c>=r for c,r in zip(counts,required))
                detail={'retained':counts,'refreshed':refreshed,'feasible':majority}
                if dependency_eta is not None:
                    dep=certificate(X,y,fs,weights=w,eps=eps,method='hybrid') if majority else None
                    if shared_loss and majority and dep['upper_bound']>dependency_eta:
                        dep=shared_loss_bound(X,y,fs,w,eps=eps)
                    detail.update(majority_feasible=majority,dependency_certificate=dep,
                                  feasible=majority and dep['upper_bound']<=dependency_eta)
                cache[key]=detail
        return cache[key]
    beams=[full];candidates=[full];scores={tuple(full):utility.evaluate(full)};trace=[]
    while len(beams[0])>1:
        generated={}
        for parent in beams:
            for a in parent:
                fs=[b for b in parent if b!=a];key=tuple(fs)
                if key not in generated:
                    generated[key]={'features':fs,'parents':[],**utility.evaluate(fs)}
                generated[key]['parents'].append(parent.copy())
        moves=sorted(generated.values(),key=lambda r:(-r['utility'],-r['clean_LOO'],tuple(r['features'])))
        accepted=[];attempts=[]
        for move in moves:
            detail=certify(move['features']);attempts.append({**move,**detail})
            if detail['feasible']:
                accepted.append(move['features']);scores[tuple(move['features'])]=utility.evaluate(move['features'])
                if len(accepted)==width:break
        trace.append({'current_beam':[p.copy() for p in beams],'ranked_moves':moves,
                      'certification_attempts':attempts,'accepted_beam':accepted})
        if not accepted:break
        beams=accepted;candidates.extend(accepted)
    result={'candidates':candidates,'trace':trace,'required':required,
        'protected_counts':[int(g.sum()) for g in groups],'beam_width':int(width),
        'candidate_utilities':[{'features':list(fs),**score} for fs,score in scores.items()],
        'training_utility_evaluations':len(utility.cache),'certificate_evaluations':len(cache),
        'scope':'training_backward_beam_finite_certificates_not_global_optimality_or_test_risk',
        'certified':certified,'eps':eps,'seed':seed}
    if dependency_eta is not None:result['dependency_eta']=dependency_eta
    if shared_loss:result['shared_loss']=True
    if ranking_workspace is not None:
        result['ranking_scope']='adaptive_training_crossfit_not_validation_or_test'
        result['ranking_splits']=ranking_workspace.splits
    return result


def beam_utility_candidates(X,y,w,**kwargs):
    from .crossfit_majority import FRACTIONS
    path=beam_utility_path(X,y,w,**kwargs);d=np.asarray(X).shape[1]
    scores={tuple(r['features']):r for r in path['candidate_utilities']};rows=[]
    for fraction in FRACTIONS:
        target=max(1,int(np.ceil(fraction*d)))
        fs=min(path['candidates'],key=lambda f:(abs(len(f)-target),-len(f),
            -scores[tuple(f)]['utility'],-scores[tuple(f)]['clean_LOO'],tuple(f)))
        rows.append({'config':{'fraction':fraction},'features':fs})
    return rows,path
