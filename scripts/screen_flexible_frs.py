"""Training-only screen: same masks, weights, loss bound; adaptive witnesses."""
import json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.flexible_witness import flexible_bounds
from certified_frfs.flexible_frs_certificate import flexible_weight_certificate
from audit_forest_compact_transfer import equivalent

SOURCE = ROOT/'results/empirical-product-fixed-count'; FOLDER = ROOT/'results/flexible-frs-screen'

def run(name):
    threadpool_limits(1); start = time.perf_counter(); artifact = SOURCE/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete']
    audit = json.loads((SOURCE/(name+'-audit.json')).read_text()); assert audit['fingerprints'][name+'.json'] == digest(artifact)
    hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in [artifact,SOURCE/(name+'-audit.json'),
        ROOT/'research/flexible-frs-screen-protocol.md',ROOT/'scripts/screen_flexible_frs.py',
        ROOT/'src/certified_frfs/flexible_frs_certificate.py',ROOT/'src/certified_frfs/flexible_witness.py']}
    X,y = load_dataset(name); rows = []; counts = Counter()
    for r in saved['records']:
        ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices'])
        raw,labels = X[ri[a]],y[ri[a]]; A,_ = preprocess_fit(raw,raw); d = A.shape[1]
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        cache = {}; rescued_reference = False
        for ai,attempt in enumerate(r['paths']['empirical']['attempts']):
            candidate = attempt['candidate']; fs = candidate['features']; key = tuple(fs)
            if key not in cache:
                reduced = np.array(candidate['weights'])[fs]; old = candidate['certificate']
                fixed_bounds = MajorityTerms(A[:,fs],labels,weights=reduced).bounds(range(len(fs)),allowed_enemies=2)
                embedded = np.zeros(d); embedded[fs] = reduced
                new_bounds = flexible_bounds(A,labels,fs,embedded,pool_size=8)
                finite = np.isfinite(fixed_bounds)&np.isfinite(new_bounds)
                assert np.all(new_bounds[finite] >= fixed_bounds[finite]-1e-10)
                assert np.all(np.isposinf(new_bounds[np.isposinf(fixed_bounds)]))
                new = flexible_weight_certificate(A,labels,fs,original,reduced,protected=protected,pool_size=8)
                equivalent(old['dependency_certificate'],new['dependency_certificate']); assert old['required'] == new['required']
                assert all(n >= o for n,o in zip(new['retained_counts'],old['retained_counts']))
                assert not old['feasible'] or new['feasible']
                cache[key] = {'features':fs,'old':old,'new':new,'old_bounds':fixed_bounds.tolist(),'new_bounds':new_bounds.tolist(),
                    'minimum_finite_bound_gain':float(np.min(new_bounds[finite]-fixed_bounds[finite])) if np.any(finite) else None}
                counts['unique_candidates'] += 1; counts['unique_rescued'] += not old['feasible'] and new['feasible']
            item = cache[key]; old,new = item['old'],item['new']; rescue = not old['feasible'] and new['feasible']
            rescued_reference |= rescue; counts['attempts'] += 1; counts['previous_rejected_attempts'] += not old['feasible']; counts['rescued_attempts'] += rescue
            rows.append({'outer':r['outer_index'],'inner':r['inner_index'],'attempt':ai,**item})
        counts['references'] += 1; counts['rescued_references'] += rescued_reference
        print(name,r['outer_index'],r['inner_index'],'flexible FRS screen','rescued',rescued_reference,flush=True)
    result = {'dataset':name,'counts':dict(counts),'rows':rows,'fingerprints':hashes,'seconds':time.perf_counter()-start,
        'query_labels_used':False,'scope':'training_feasibility_screen_not_prediction_experiment'}
    (FOLDER/(name+'.json')).write_text(json.dumps(result,separators=(',',':')),encoding='utf-8'); return result

if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(run,n) for n in names])]
    counts = sum((Counter(r['counts']) for r in rows),Counter()); assert counts['references'] == 128 and counts['previous_rejected_attempts'] == 177
    result = {'counts':dict(counts),'rows':[{'dataset':r['dataset'],'counts':r['counts'],'seconds':r['seconds']} for r in rows],
        'resource_gate':bool(counts['rescued_attempts'] >= .1*counts['previous_rejected_attempts'] and counts['rescued_references'] >= 10),
        'query_labels_used':False,'goal_achieved':False,'scope':'opportunity_for_fresh_classifier_count_controlled_pilot_not_prediction_gain'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
