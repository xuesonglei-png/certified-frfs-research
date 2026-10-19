"""Training hard5NN candidates from affine neighbor-order changes.

For nonnegative cityblock weights, real-arithmetic distances are affine on
a weight segment. This prototype enumerates per-object pair crossings and
merges intervals of constant training correctness. Interval representatives
are candidate proposals only: ordinary floating arithmetic and a finite guard
budget do not certify an optimum over the continuous certified segment.
No validation observations are accepted by this API.
"""
import numpy as np


def segment_candidates(A, y, anchor, target, *, max_candidates=16, chunk_size=256):
    A, y = np.asarray(A, float), np.asarray(y)
    anchor, target = np.asarray(anchor, float), np.asarray(target, float)
    if (A.ndim != 2 or len(A) < 7 or y.shape != (len(A),)
            or anchor.shape != (A.shape[1],) or target.shape != anchor.shape
            or not np.isfinite(A).all() or not np.isfinite(anchor).all() or not np.isfinite(target).all()
            or anchor.min() < 0 or target.min() < 0 or anchor.sum() <= 0
            or abs(anchor.sum()-target.sum()) > 1e-8
            or int(max_candidates) != max_candidates or max_candidates < 7
            or int(chunk_size) != chunk_size or chunk_size < 1):
        raise ValueError('invalid training segment or candidate budget')
    n = len(A); classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    D0, D1 = np.zeros((n,n)), np.zeros((n,n))
    for a in range(A.shape[1]):
        D0 += abs(A[:,a,None]*anchor[a]-A[None,:,a]*anchor[a])
        D1 += abs(A[:,a,None]*target[a]-A[None,:,a]*target[a])
    slope = D1-D0
    alpha = 1/(len(classes)*counts[encoded])
    changes = {}; initial = 0.; crossing_count = 0; correctness_changes = 0
    first, second = np.triu_indices(n-1, 1)
    for i in range(n):
        ids = np.flatnonzero(np.arange(n) != i); base, delta = D0[i,ids], slope[i,ids]
        denominator = delta[first]-delta[second]
        different = denominator != 0
        crossing = (base[second[different]]-base[first[different]])/denominator[different]
        crossing = np.unique(crossing[np.isfinite(crossing)&(crossing > 0)&(crossing < 1)])
        crossing_count += len(crossing)
        bounds = np.concatenate(([0.], crossing, [1.]))
        midpoint = bounds[:-1]+(bounds[1:]-bounds[:-1])/2
        correct = np.empty(len(midpoint), bool)
        for start in range(0,len(midpoint),chunk_size):
            t = midpoint[start:start+chunk_size]
            distance = base[None,:]+t[:,None]*delta[None,:]
            neighbors = ids[np.argsort(distance,axis=1,kind='stable')[:,:5]]
            votes = np.stack([(encoded[neighbors] == c).sum(axis=1) for c in range(len(classes))],axis=1)
            correct[start:start+len(t)] = votes.argmax(axis=1) == encoded[i]
        initial += alpha[i]*correct[0]
        changed = np.flatnonzero(correct[1:] != correct[:-1])+1
        correctness_changes += len(changed)
        for j in changed:
            at = float(bounds[j]); amount = alpha[i]*(int(correct[j])-int(correct[j-1]))
            changes[at] = changes.get(at,0.)+amount
    intervals = []; current = float(initial); left = 0.
    # A merged correctness interval can contain neighbor-order changes;
    # it only asserts that the summed training correctness is unchanged.
    for right in sorted(changes):
        intervals.append({'left':left,'right':right,'estimated_hard_LOO_BA':current})
        current += changes[right]; left = right
    intervals.append({'left':left,'right':1.,'estimated_hard_LOO_BA':current})
    ranked = sorted(intervals,key=lambda r:(-round(r['estimated_hard_LOO_BA'],12),-(r['right']-r['left']),r['left']))
    # Preserve every previous fixed point. Remaining budget proposes interior
    # representatives; all require fresh actual votes and original guards.
    fixed = (0.,.05,.1,.25,.5,.75,1.)
    proposals = [{'t':t,'origin':'fixed_grid','estimated_hard_LOO_BA':None} for t in fixed]
    seen = set(fixed)
    for region in ranked:
        if len(proposals) >= max_candidates: break
        t = float(region['left']+(region['right']-region['left'])/2)
        if not region['left'] < t < region['right'] or t in seen: continue
        proposals.append({'t':t,'origin':'hard_correctness_interval',**region}); seen.add(t)
        if len(proposals) == max_candidates: break
    return proposals, {'pair_crossings':crossing_count,'correctness_changes':correctness_changes,
        'merged_intervals':len(intervals),'estimated_best_open_interval_BA':float(max(r['estimated_hard_LOO_BA'] for r in intervals)),
        'candidate_count':len(proposals),'max_candidates':max_candidates,
        'scope':'ordinary_float_affine_order_candidates_not_continuous_certified_optimum_or_population_gain'}
