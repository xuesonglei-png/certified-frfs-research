"""Independent endpoint pair minima and distinct-friend rank replay."""
import json
from collections import Counter
import numpy as np
from run_nested_joint_path import ROOT,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import interval_bounds
from certified_frfs.majority_metric import MajorityTerms
from audit_forest_compact_transfer import equivalent

SOURCE = ROOT/'results/empirical-product-fixed-count'; FOLDER = ROOT/'results/flexible-frs-screen'

def oracle(A,y,fs,w):
    lo,hi = interval_bounds(A,.01); lower,upper = lo[:,fs],hi[:,fs]; weights = np.array(w)[fs]
    bounds = np.full(len(A),-np.inf)
    for i in range(len(A)):
        friends = [j for j in range(len(A)) if j != i and y[j] == y[i]]
        if len(friends) < 3: continue
        friends.sort(key=lambda j:(float(abs(A[j,fs]-A[i,fs])@weights),j)); friends = friends[:8]
        enemies = [j for j in range(len(A)) if y[j] != y[i]]
        if len(enemies) < 3: bounds[i] = np.inf; continue
        matrix = np.zeros((len(enemies),len(friends)))
        for a,weight in enumerate(weights):
            values = []
            for x in (lower[i,a],upper[i,a]):
                near = np.maximum(0,np.maximum(lower[enemies,a]-x,x-upper[enemies,a]))
                far = np.maximum(abs(x-lower[friends,a]),abs(x-upper[friends,a]))
                values.append(near[:,None]-far[None,:])
            matrix += weight*np.minimum(values[0],values[1])
        enemy_bounds = np.sort(matrix,axis=1)[:,-3]; bounds[i] = sorted(enemy_bounds)[2]
    return bounds-1e-10*(1+abs(np.where(np.isfinite(bounds),bounds,0)))

if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); names = [r['dataset'] for r in summary['rows']]
    for name in names:
        artifact = json.loads((FOLDER/(name+'.json')).read_text()); source = json.loads((SOURCE/(name+'.json')).read_text()); X,y = load_dataset(name); cache = {}; rescued_refs = set()
        for rel,sha in artifact['fingerprints'].items(): assert digest(ROOT/rel) == sha
        for row in artifact['rows']:
            ref = next(r for r in source['records'] if r['outer_index'] == row['outer'] and r['inner_index'] == row['inner'])
            attempt = ref['paths']['empirical']['attempts'][row['attempt']]; candidate = attempt['candidate']; fs = candidate['features']
            assert fs == row['features'] and candidate['certificate'] == row['old']
            key = (row['outer'],row['inner'],tuple(fs))
            if key not in cache:
                ri,a = np.array(ref['outer_reference_indices']),np.array(ref['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; A,_ = preprocess_fit(raw,raw)
                full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); protected = MajorityTerms(A,labels,weights=full).bounds(range(A.shape[1]),allowed_enemies=2) > 1e-10
                values = oracle(A,labels,fs,candidate['weights']); np.testing.assert_allclose(row['new_bounds'],values,atol=1e-10,rtol=1e-10)
                groups = [(labels == c)&protected for c in np.unique(labels)]
                required = [int(np.ceil(.9*g.sum()-1e-12)) for g in groups]; retained = [int(((values > 1e-10)&g).sum()) for g in groups]
                majority = all(n >= r for n,r in zip(retained,required)); new = row['new']
                assert new['required'] == required and new['retained_counts'] == retained and new['majority_feasible'] == majority
                equivalent(new['dependency_certificate'],row['old']['dependency_certificate'])
                assert new['feasible'] == (majority and new['dependency_certificate']['upper_bound'] <= .1)
                oldbounds = np.array(row['old_bounds']); finite = np.isfinite(values)&np.isfinite(oldbounds)
                assert np.all(values[finite] >= oldbounds[finite]-1e-10)
                cache[key] = not row['old']['feasible'] and new['feasible']; counts['unique_candidates'] += 1; counts['unique_rescued'] += cache[key]
                counts['independent_pair_rank_replays'] += 1
            rescue = cache[key]; counts['attempts'] += 1; counts['previous_rejected_attempts'] += not row['old']['feasible']; counts['rescued_attempts'] += rescue
            if rescue: rescued_refs.add((row['outer'],row['inner']))
        counts['references'] += len(source['records']); counts['rescued_references'] += len(rescued_refs)
        print(name,'independent flexible FRS screen passed',flush=True)
    for k,v in summary['counts'].items(): assert counts[k] == v
    gate = counts['rescued_attempts'] >= .1*counts['previous_rejected_attempts'] and counts['rescued_references'] >= 10
    assert summary['resource_gate'] == gate and not summary['query_labels_used'] and not summary['goal_achieved']
    result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'auditor_sha256':digest(ROOT/'scripts/audit_flexible_frs_screen.py'),
        'scope':'independent_all_attempt_endpoint_pair_order_statistics_original_mask_dependency_provenance_no_query_labels'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
