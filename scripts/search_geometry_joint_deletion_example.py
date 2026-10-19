"""Synthetic certificate counterexample search, not a classification benchmark."""
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_geometry_forest import ROOT
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.validated_pool import preprocess_fit


def main():
    threadpool_limits(1); rng = np.random.default_rng(20261013); first = None
    for trial in range(400):
        y = np.repeat([0, 1], 6); signal = rng.uniform(.05, .8, 5)
        latent = np.column_stack([2*y-1, rng.normal(size=12), rng.normal(size=12)])
        loading = rng.normal(size=(3, 5)); loading[0] = signal
        raw = latent @ loading+rng.normal(scale=rng.uniform(.15, .65), size=(12, 5))
        A, _ = preprocess_fit(raw, raw); context = GeometryStageContext(A, y)
        if min(sum(context.protected & (y == c)) for c in np.unique(y)) == 0: continue
        full = context.candidate(range(5)); assert full['certificate']['feasible']
        singles = {a:context.candidate([z for z in range(5) if z != a]) for a in range(5)}
        for a, b in combinations(range(5), 2):
            if singles[a]['certificate']['feasible'] or singles[b]['certificate']['feasible']: continue
            pair = context.candidate([z for z in range(5) if z not in (a, b)])
            if pair['certificate']['feasible']:
                record = {'trial':trial,'X':A.tolist(),'y':y.tolist(),'removed_pair':[a,b],
                    'protected':context.protected.tolist(),'full':full,'singles':singles,'pair':pair,
                    'all_single_deletions_infeasible':all(not r['certificate']['feasible'] for r in singles.values()),
                    'scope':'nonmonotone_certified_feasibility_under_correlation_redistribution_not_population_accuracy_or_a_new_general_theorem'}
                if first is None: first = record
                if record['all_single_deletions_infeasible']:
                    first = record; break
        if first is not None and first['all_single_deletions_infeasible']: break
        if trial % 50 == 0: print('searched',trial+1,'weaker_example',first is not None,flush=True)
    result = {'found':first is not None,'trials':trial+1,'seed':20261013,'example':first,
        'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'src/certified_frfs/geometry_only_forest_stage.py',ROOT/'src/certified_frfs/feasible_transport.py',
             ROOT/'src/certified_frfs/redistributed_loss.py',ROOT/'src/certified_frfs/majority_metric.py')},
        'limits':'Synthetic_algorithmic_diagnostic_only; finite_search_failure_does_not_prove_monotonicity; no_benchmark_improvement_claim.'}
    (ROOT/'results/geometry-joint-deletion-example.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('example','fingerprints')},flush=True)
    if first: print('removed',first['removed_pair'],'all singles fail',first['all_single_deletions_infeasible'],flush=True)


if __name__ == '__main__': main()
