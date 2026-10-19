"""Expose accepted prefixes of an audited fixed-reference deletion path.

This helper validates provenance/continuity fields, not the underlying bound.
Callers must independently verify the path artifact before using its guards.
"""
import math
import numpy as np

PROGRESS=(0.,.25,.5,.75,1.)


def prefixes(fit, *, require_original=False):
    if fit['guard_reference'] not in ('original_full','own_anchor'):
        raise ValueError('unknown fixed reference')
    if require_original and fit['guard_reference']!='original_full':
        raise ValueError('original-full path admission required')
    if not fit['initial_guard']['accepted']:
        raise ValueError('inadmissible initial path metric')
    fs=list(fit['initial_features']);w=list(fit['initial_weights']); rows=[]
    if not fs or fs!=sorted(set(fs)) or len(w)!=len(fs):
        raise ValueError('invalid initial path support')
    def metric(step,guard):
        if not np.isfinite(w).all() or min(w)<0:
            raise ValueError('finite nonnegative prefix weights required')
        return {'features':fs.copy(),'weights':w.copy(),'path_step':step,
            'guard_reference':fit['guard_reference'],'guard':guard}
    rows.append(metric(0,fit['initial_guard']));previous_id=-1
    mass=sum(w)
    for step,pid in enumerate(fit['chosen_ids'],1):
        if int(pid)!=pid or not previous_id<pid<len(fit['checks']):
            raise ValueError('chosen path IDs must strictly increase')
        point=fit['checks'][pid]
        if point['id']!=pid or not point['eligible'] or not point['accepted']:
            raise ValueError('unaccepted path endpoint')
        if point['features_before']!=fs or point['weights_before']!=w:
            raise ValueError('discontinuous path prefix')
        next_fs=list(point['features']);next_w=list(point['weights'])
        if (next_fs!=sorted(set(next_fs)) or len(next_fs)!=len(fs)-1
            or not set(next_fs)<set(fs) or len(next_w)!=len(next_fs)
            or abs(sum(next_w)-mass)>1e-8):
            raise ValueError('prefix must omit exactly one attribute and preserve mass')
        fs,w=next_fs,next_w;previous_id=pid
        rows.append(metric(step,{k:point[k] for k in ('certificate','class_certificate','accepted')}))
    if fs!=fit['features'] or w!=fit['weights']:
        raise ValueError('path terminal metric mismatch')
    return rows


def progress_options(fit, *, require_original=False):
    path=prefixes(fit,require_original=require_original); length=len(path)-1
    # Fixed recipes across inner folds, even when their accepted lengths differ.
    return [{'progress':p,**path[int(math.ceil(length*p))]} for p in PROGRESS]
