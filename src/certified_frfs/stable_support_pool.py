"""Fixed broad pool, two-partition worst-improvement selection, FRS admission.

This controls a declared adaptive training criterion, not generalization risk.
"""


def support_pool(initial, d, proposals=()):
    initial = tuple(sorted(initial)); k = len(initial); pool = {initial:{'old'}}
    for remove in initial:
        for add in range(d):
            if add in initial: continue
            fs = tuple(sorted([a for a in initial if a != remove]+[add]))
            pool.setdefault(fs,set()).add('old_one_exchange')
    for origin,features in proposals:
        fs = tuple(sorted(features))
        if len(fs) != k or len(set(fs)) != k or any(int(a) != a or not 0 <= a < d for a in fs):
            raise ValueError('proposal count/support mismatch')
        pool.setdefault(fs,set()).add(origin)
    return [{'features':list(fs),'origins':sorted(origins)} for fs,origins in sorted(pool.items())]


def select_stable_pool(initial, pool, risks, candidate):
    if len(risks) != 2: raise ValueError('exactly two fixed selection partitions required')
    initial = sorted(initial); baseline = [risk(initial) for risk in risks]
    certificate = candidate(initial)
    if not certificate['certificate']['feasible']: raise ValueError('certified fallback required')
    scored,eligible = [],[]
    for entry in pool:
        fs = entry['features']
        if len(fs) != len(initial): raise ValueError('all supports must preserve count')
        values = [risk(fs) for risk in risks]
        gains = [round(b['balanced_error'],12)-round(v['balanced_error'],12) for b,v in zip(baseline,values)]
        key = (-round(min(gains),12),-round(sum(gains)/2,12),round(sum(v['balanced_hinge'] for v in values)/2,12),tuple(fs))
        row = {**entry,'risk_by_partition':values,'gains':gains,'eligible':all(g > 1e-12 for g in gains)}
        scored.append(row)
        if row['eligible']: eligible.append((key,row))
    eligible.sort(key=lambda r:r[0]); attempts = []; selected = initial; selected_values = baseline
    for _,row in eligible:
        geometry = candidate(row['features']); ok = geometry['certificate']['feasible']
        attempts.append({'features':row['features'],'gains':row['gains'],'candidate':geometry,'accepted':ok})
        if ok: selected = row['features']; certificate = geometry; selected_values = row['risk_by_partition']; break
    assert certificate['certificate']['feasible'] and len(selected) == len(initial)
    return {'features':selected,'initial_features':initial,'candidate':certificate,'initial_risk_by_partition':baseline,
        'final_risk_by_partition':selected_values,'scored_pool':scored,'ranked_eligible':[r['features'] for _,r in eligible],
        'attempts':attempts,'selection_seeds':[r.seed for r in risks],
        'scope':'adaptive_two_partition_worst_improvement_certified_pool_not_native_baseline_or_test_risk'}
