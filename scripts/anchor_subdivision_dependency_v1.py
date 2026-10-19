"""Bounded anchor-box subdivision of signed nearest-enemy dependency loss.

This reuses exact scalar contrasts, but partitions the anchor box so nearest
enemy supersets and full-witness choices can adapt to subregions. It does not
share coordinates across different anchor objects and claims no new theorem.
"""
from __future__ import annotations

import numpy as np


def leaf_upper(lo, hi, i, anchor_lo, anchor_hi, enemies, witnesses, wf, wide):
    """Valid signed membership-loss upper on one anchor subbox."""
    el, eh = lo[enemies], hi[enemies]
    near = np.maximum(0., np.maximum(anchor_lo-eh, el-anchor_hi))@wide
    far = np.maximum(abs(anchor_lo-eh), abs(anchor_hi-el))@wide
    cap = float(far.min())
    possible = enemies[near <= cap+1e-12]
    lows = near[near <= cap+1e-12]
    jl, jh = lo[possible][:,None,:], hi[possible][:,None,:]
    kl, kh = lo[witnesses][None,:,:], hi[witnesses][None,:,:]
    def at(t):
        return wf*np.maximum(abs(t-kl),abs(t-kh))-wide*np.maximum(0.,np.maximum(jl-t,t-jh))
    values = np.maximum(at(anchor_lo), at(anchor_hi))
    for point in (jl,jh,kl,kh,(kl+kh)/2):
        values = np.maximum(values,at(np.clip(point,anchor_lo,anchor_hi)))
    coefficient = wf-wide
    same_near = np.maximum(0.,np.maximum(anchor_lo-jh,jl-anchor_hi))
    same_far = np.maximum(abs(anchor_lo-jh),abs(anchor_hi-jl))
    values = np.where((possible[:,None] == witnesses[None,:])[:,:,None],
                      coefficient*np.where(coefficient >= 0,same_far,same_near), values)
    signed = values.sum(axis=2).min(axis=1)
    raw = float(np.max(np.minimum(1.-lows,signed)))
    unsaturated = cap <= 1.-1e-12
    return {"upper": raw if unsaturated else max(0.,raw), "raw_upper": raw,
            "unsaturated": bool(unsaturated), "reduced_upper_cap": cap,
            "possible_enemies": possible.tolist()}


def refine_object(A,y,fs,wf,wr,i,*,max_leaves=16):
    A,y,wf,wr = np.asarray(A,float),np.asarray(y),np.asarray(wf,float),np.asarray(wr,float)
    fs = np.asarray(fs,int)
    if (max_leaves < 1 or int(max_leaves) != max_leaves or A.ndim != 2 or wf.shape != (A.shape[1],)
        or wr.shape != fs.shape or np.any(wf<0) or np.any(wr<0) or not np.isfinite(A).all()
        or not np.isfinite(wf).all() or not np.isfinite(wr).all()):
        raise ValueError("invalid subdivision inputs")
    lo,hi = np.maximum(0.,A-.01),np.minimum(1.,A+.01)
    wide = np.zeros(A.shape[1]);wide[fs]=wr
    enemies = np.flatnonzero(y != y[i])
    if not len(enemies):
        raise ValueError("requires at least two classes")
    nominal = np.zeros(len(enemies))
    for a in range(A.shape[1]):
        nominal += abs(A[i,a]*wf[a]-A[enemies,a]*wf[a])
    witnesses = enemies[np.lexsort((enemies,nominal))][:8]
    def node(left,right,parent=None):
        result = leaf_upper(lo,hi,i,left,right,enemies,witnesses,wf,wide)
        if parent is not None:
            result["upper"] = min(result["upper"],parent["upper"])
        return {"lower":left.tolist(),"upper_coordinates":right.tolist(), **result}
    nodes = [node(lo[i],hi[i])]
    leaves = [0];splits=[]
    while len(leaves)<max_leaves:
        chosen = max(leaves,key=lambda idx:(nodes[idx]["upper"],-idx))
        parent = nodes[chosen]
        left,right=np.array(parent["lower"]),np.array(parent["upper_coordinates"])
        axis=int(np.argmax((right-left)*(wf+wide)))
        if right[axis]-left[axis] <= 1e-15:
            break
        midpoint=float((left[axis]+right[axis])/2)
        middle_left=right.copy();middle_left[axis]=midpoint
        middle_right=left.copy();middle_right[axis]=midpoint
        children=[len(nodes),len(nodes)+1]
        nodes += [node(left,middle_left,parent),node(middle_right,right,parent)]
        leaves.remove(chosen);leaves.extend(children)
        splits.append({"parent":chosen,"axis":axis,"midpoint":midpoint,"children":children})
    value=min(nodes[0]["upper"],max(nodes[idx]["upper"] for idx in leaves))
    return {"object":int(i),"root_upper":nodes[0]["upper"],"refined_upper":float(value),
            "witnesses":witnesses.tolist(),"nodes":nodes,"leaves":leaves,"splits":splits,
            "max_leaves":max_leaves,"scope":"signed_anchor_subbox_witness_envelope_not_population_risk"}


def refine_signed_guard(A,y,fs,wf,wr,signed_report,baseline_class_bounds,*,max_leaves=16,max_objects=24):
    values=np.array(signed_report["signed_object_report"]["signed_object_upper_bounds"],float)
    labels=np.asarray(y);classes,counts=np.unique(labels,return_counts=True)
    failing=[c for c,b in zip(classes,baseline_class_bounds) if b>.1]
    eligible=np.flatnonzero(np.isin(labels,failing))
    eligible=eligible[np.lexsort((eligible,-values[eligible]))][:max_objects]
    objects=[]
    for i in eligible:
        refined=refine_object(A,y,fs,wf,wr,int(i),max_leaves=max_leaves)
        np.testing.assert_allclose(refined["root_upper"],values[i],atol=1e-9,rtol=1e-8)
        values[i]=min(values[i],refined["refined_upper"])
        objects.append(refined)
    raw=[float(values[labels==c].mean()) for c in classes]
    padded=[float(np.clip(b+1e-10*(1+abs(b)),0.,1.)) for b in raw]
    bounds=[min(float(old),new) for old,new in zip(baseline_class_bounds,padded)]
    global_upper=float(np.asarray(counts)@np.asarray(bounds)/len(labels))
    return {"class_bounds":bounds,"signed_object_values":values.tolist(),"raw_class_means":raw,
            "global_upper":global_upper,"objects":objects,"classes":classes.tolist(),
            "max_leaves":max_leaves,"max_refined_objects":max_objects}
