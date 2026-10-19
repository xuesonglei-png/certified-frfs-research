"""Fold-local certified restoration with locally calibrated soft-neighbor loss.

Experimental code is versioned outside frozen production modules.  Restoration
uses only training ranks and certificates.  It never inspects validation labels.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.relation_support_selector_v1 import _candidate, _ranking, _fit_relation
from certified_frfs.selective_support_selector_v3 import query_scores
from certified_frfs.softvote_metric import softvote_terms, softvote_objective
from certified_frfs.validated_pool import preprocess_fit

FAMILIES = ('uniform_sqrt', 'half_majority', 'learned_majority',
            'half_local_soft', 'local_soft')
FRACTIONS = (.5, .7, .85, 1.)


def local_soft_relation(A, y):
    """Training-only adaptive scale; capped unit-simplex nonconvex local fit.

    Uniform leave-one-out fifth-neighbor radius calibrates the temperature.
    This avoids the dimension-dependent scale mismatch of the former fixed
    temperature.  No global optimality or hard-vote surrogate bound is claimed.
    """
    d = A.shape[1]; start = np.ones(d)/d
    D = np.zeros((len(A), len(A)))
    for a in range(d):
        D += np.abs(A[:, a, None]-A[None, :, a])/d
    np.fill_diagonal(D, np.inf)
    radius = np.partition(D, min(4, len(A)-2), axis=1)[:, min(4, len(A)-2)]
    temperature = max(.005, float(np.median(radius)))
    C, same, mask, labels = softvote_terms(A, y, eps=.01)
    fun = lambda w: softvote_objective(w, C, same, mask, labels,
                                      temperature=temperature, reg=.02)
    before = fun(start)[0]
    fit = minimize(fun, start, jac=True, method='SLSQP',
                   bounds=[(1e-8, 2/d)]*d,
                   constraints={'type': 'eq', 'fun': lambda w:w.sum()-1,
                                'jac':lambda w:np.ones(d)},
                   options={'maxiter':200, 'ftol':1e-9})
    chosen = start.copy(); status = 'explicit_uniform_solver_fallback'
    if fit.success and np.isfinite(fit.x).all():
        v = capped_sparse_projection(fit.x, d, 2/d)
        if v.min()>0 and fun(v)[0] <= before+1e-10:
            chosen = v; status = 'local_descent'
    return chosen*np.sqrt(d), {'temperature':temperature,'reg':.02,
        'status':status,'solver_success':bool(fit.success),
        'solver_message':str(fit.message),'objective_before':before,
        'objective_after':fun(chosen)[0], 'unit_weights':chosen.tolist()}


def relations(A, y):
    uniform = _fit_relation(A,y,'uniform_sqrt')
    majority = _fit_relation(A,y,'learned_majority')
    soft, record = local_soft_relation(A,y)
    return {'uniform_sqrt':uniform,'half_majority':(uniform+majority)/2,
            'learned_majority':majority,'half_local_soft':(uniform+soft)/2,
            'local_soft':soft}, record


def repaired(A,y,reference,ranking,requested,cache):
    """Validate every proposed support and restore in a fixed rank order."""
    trace=[]; d=A.shape[1]
    for count in range(requested,d+1):
        fs=sorted(ranking[:count]); key=tuple(fs)
        if key not in cache:cache[key]=_candidate(A,y,reference,fs,len(fs))
        c=cache[key];trace.append(c)
        if c['joint_feasible']:
            return {**c,'requested_count':requested,'restored_count':count-requested,
                    'uniform_fallback':False}, trace
    # Full learned relation itself can fail the original uniform guard.
    uniform=np.ones(d)/np.sqrt(d)
    c=_candidate(A,y,uniform,list(range(d)),d)
    if not c['joint_feasible']:raise RuntimeError('uniform full certificate failed')
    return {**c,'requested_count':requested,'restored_count':d-requested,
            'uniform_fallback':True}, trace+[c]


def block(A,y,B,truth,seed):
    fitted,learner=relations(A,y);families={}
    for family,w in fitted.items():
        ranking=_ranking(A,y,w,family,seed);cache={};scores={};candidates=[]
        for fraction in FRACTIONS:
            requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(fraction*A.shape[1])))
            model,trace=repaired(A,y,w,ranking,requested,cache)
            key=(tuple(model['features']),tuple(model['weights']))
            if key not in scores:scores[key]=query_scores(A,y,B,truth,model)
            candidates.append({'fraction':fraction,'model':model,'trace':trace,'score':scores[key]})
        families[family]={'reference_weights':w.tolist(),'ranking':ranking,'candidates':candidates}
    return {'families':families,'learner':learner}


def choose(blocks):
    options=[]
    for family in FAMILIES:
        for index,fraction in enumerate(FRACTIONS):
            entries=[b['families'][family]['candidates'][index] for b in blocks]
            options.append({'family':family,'recipe_index':index,'fraction':fraction,
                'mean_BA':float(np.mean([v['score']['clean_BA'] for v in entries])),
                'mean_certified_BA':float(np.mean([v['score']['query_certified_BA'] for v in entries])),
                'mean_count':float(np.mean([len(v['model']['features']) for v in entries])),
                'mean_logloss':float(np.mean([v['score']['balanced_logloss'] for v in entries]))})
    best=max(r['mean_BA'] for r in options)
    eligible=[r for r in options if r['mean_BA']>=best-.005-1e-12]
    selected=max(eligible,key=lambda r:(r['mean_certified_BA'],r['mean_BA'],
        -r['mean_count'],-r['mean_logloss'],-FAMILIES.index(r['family']),-r['recipe_index']))
    return options,selected


def fit_outer(X,y,seed=20261011,inner_splits=3):
    blocks=[]
    for fold,(ii,jj) in enumerate(StratifiedKFold(inner_splits,shuffle=True,random_state=seed).split(X,y)):
        A,B=preprocess_fit(X[ii],X[jj])
        b=block(A,y[ii],B,y[jj],seed+fold)
        blocks.append({'fit_indices':ii.tolist(),'validation_indices':jj.tolist(),**b})
    options,chosen=choose(blocks)
    A,_=preprocess_fit(X,X);fitted,learner=relations(A,y)
    family=chosen['family'];w=fitted[family];rank=_ranking(A,y,w,family,seed+100)
    requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(chosen['fraction']*A.shape[1])))
    selected,trace=repaired(A,y,w,rank,requested,{})
    return {'selected':selected,'chosen':chosen,'options':options,'inner':blocks,
            'reference_weights':w.tolist(),'ranking':rank,'learner':learner,
            'outer_trace':trace,'seed':seed,'inner_splits':inner_splits}
