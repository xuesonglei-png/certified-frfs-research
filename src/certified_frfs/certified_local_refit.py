"""Exhaustive feasible one-exchange descent, without top-pool/round cutoffs.

Stationarity concerns the fixed adaptive CV objective and reference certificate,
not population prediction accuracy or a globally optimal reduct.
"""
from .crossfit_rbf_refit import refit_risk_key,foldwise_improves


def certified_local_refit(initial, risk, candidate):
    current = sorted(initial); certificate = candidate(current)
    if not certificate['certificate']['feasible']: raise ValueError('certified starting support required')
    value = initial_risk = risk(current); stages, attempts, seen = [], [], {tuple(current)}
    step = 0
    while True:
        choices = []; mean_improving = 0
        for remove in current:
            for add in range(risk.d):
                if add in current: continue
                fs = sorted([a for a in current if a != remove]+[add]); score = risk(fs)
                mean_improving += round(score['balanced_error'],12) < round(value['balanced_error'],12)
                if foldwise_improves(score,value): choices.append((refit_risk_key(score),tuple(fs),remove,add,score))
        choices.sort(key=lambda row:(row[0],row[1]))
        stage = {'round':step,'features':current.copy(),'risk':value,
            'mean_improving_candidates':int(mean_improving),'foldwise_improving_candidates':len(choices),
            'ranked_choices':[{'features':list(fs),'remove':remove,'add':add,'risk':score} for _,fs,remove,add,score in choices]}
        stages.append(stage); accepted = None
        for _,fs,remove,add,score in choices:
            geometry = candidate(list(fs)); ok = geometry['certificate']['feasible']
            attempts.append({'round':step,'remove':remove,'add':add,'risk':score,'candidate':geometry,'accepted':ok})
            if ok: accepted = list(fs); value = score; certificate = geometry; break
        if accepted is None: break
        assert tuple(accepted) not in seen; seen.add(tuple(accepted)); current = accepted; step += 1
    assert len(current) == len(initial) and certificate['certificate']['feasible']
    return {'features':current,'candidate':certificate,'initial_features':sorted(initial),'initial_risk':initial_risk,
        'final_risk':value,'stages':stages,'attempts':attempts,'C':risk.C,'seed':risk.seed,
        'accepted_steps':step,'termination':'exhaustive_one_exchange_foldwise_stationary',
        'scope':'adaptive_training_local_stationarity_not_global_reduct_or_population_risk'}
