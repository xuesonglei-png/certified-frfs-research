"""Prospective clean-primary selector with a paired training-fold anchor gate.

The one-standard-error gate is a selection heuristic on three correlated score
blocks, not a statistical noninferiority guarantee. All certificates stay fixed.
"""
import numpy as np
from repair_relation_selector_v4 import FAMILIES,FRACTIONS,relations,repaired
from certified_frfs.relation_support_selector_v1 import _ranking
from certified_frfs.validated_pool import preprocess_fit

RULES=('paired_anchor','clean_primary')

def options(fit):
    entries=[]
    anchor=np.array([b['families']['uniform_sqrt']['candidates'][3]['score']['clean_BA'] for b in fit['inner']])
    for f in FAMILIES:
        for i,fraction in enumerate(FRACTIONS):
            rows=[b['families'][f]['candidates'][i] for b in fit['inner']]
            scores=np.array([c['score']['clean_BA'] for c in rows]);delta=scores-anchor
            stderr=float(np.std(delta,ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 0.
            entries.append({'family':f,'recipe_index':i,'fraction':fraction,
                'mean_BA':float(scores.mean()),'mean_certified_BA':float(np.mean([c['score']['query_certified_BA'] for c in rows])),
                'mean_count':float(np.mean([len(c['model']['features']) for c in rows])),
                'mean_logloss':float(np.mean([c['score']['balanced_logloss'] for c in rows])),
                'paired_delta':delta.tolist(),'paired_standard_error':stderr,
                'anchor_gate_passed':float(delta.mean())-stderr>=-1e-12})
    return entries

def choose(fit,rule):
    pool=options(fit)
    eligible=[r for r in pool if r['anchor_gate_passed']] if rule=='paired_anchor' else pool
    if rule not in RULES:raise ValueError(rule)
    chosen=max(eligible,key=lambda r:(round(r['mean_BA'],12),round(r['mean_certified_BA'],12),
        -r['mean_count'],-round(r['mean_logloss'],12),-FAMILIES.index(r['family']),-r['recipe_index']))
    return pool,chosen

def refit(X,y,chosen,seed,*,fitted=None):
    A,_=preprocess_fit(X,X)
    fitted,learner=relations(A,y) if fitted is None else fitted
    w=fitted[chosen['family']];rank=_ranking(A,y,w,chosen['family'],seed+100)
    requested=max(int(np.ceil(A.shape[1]/2)),int(np.ceil(A.shape[1]*chosen['fraction'])))
    selected,trace=repaired(A,y,w,rank,requested,{})
    return {'selected':selected,'outer_trace':trace,'reference_weights':w.tolist(),
            'ranking':rank,'learner':learner,'chosen':chosen,'seed':seed}
