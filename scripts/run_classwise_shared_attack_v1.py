"""Frozen, training-only classwise attack cohort with independent loss replay."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from classwise_shared_attack_v1 import attack
from diagnose_repair_bottlenecks_v7 import ROOT, NAMES, digest

FOLDER = ROOT/'results/classwise-shared-attack-v1'
FAMILIES = ('uniform_sqrt', 'half_majority', 'learned_majority', 'half_local_soft', 'local_soft')


def memberships(A, y, fs, w):
    distance = np.zeros((len(A), len(A)))
    for a, weight in zip(fs, w, strict=True):
        distance += weight*abs(A[:,a,None]-A[None,:,a])
    distance[np.asarray(y)[:,None] == np.asarray(y)[None,:]] = np.inf
    return np.minimum(1., distance.min(axis=1))


def references(c):
    return [np.array(c[k]['dependency_certificate'].get('full_weights', c['weights']))
            for k in ('certificate', 'family_certificate')]


def tasks(parent, A, y):
    """No validation fields read; exact restored-trace metrics supply candidates."""
    block = parent['rows'][0]['fit']['inner'][0]
    unique = {}; nominal = {}; d = A.shape[1]; classes = np.unique(y)
    strata = {'classwise_only': [], 'global_blocked': []}
    count = Counter()
    for family in FAMILIES:
        for recipe in block['families'][family]['candidates']:
            for c in recipe['trace']:
                refs = references(c)
                key = family, tuple(c['features']), tuple(c['weights']), tuple(refs[0]), tuple(refs[1])
                unique[key] = c
    def membership(fs, weights):
        key = tuple(fs), tuple(weights)
        if key not in nominal:
            nominal[key] = memberships(A,y,fs,weights)
        return nominal[key]
    for key, c in unique.items():
        global_rejected = any(c[k]['dependency_certificate']['upper_bound'] > .1
                              for k in ('certificate', 'family_certificate'))
        class_rejected = not (c['classwise']['feasible'] and c['family_classwise']['feasible'])
        count['unique_training_candidates'] += 1
        if not (global_rejected or class_rejected):
            continue
        count['dependency_blocked'] += 1
        reduced = membership(c['features'], c['weights'])
        refs = references(c)
        nominal_losses = [membership(range(d), ref)-reduced for ref in refs]
        violating = any(v.mean() > .1+1e-10 or any(v[y == q].mean() > .1+1e-10 for q in classes)
                        for v in nominal_losses)
        if violating:
            count['already_nominal_violating_excluded'] += 1
            continue
        stratum = 'global_blocked' if global_rejected else 'classwise_only'
        strata[stratum].append((key,c))
        count['eligible_'+stratum] += 1
    selected = []
    for stratum, candidates in strata.items():
        ordered = sorted(candidates, key=lambda pair: (len(pair[1]['features']),
            FAMILIES.index(pair[0][0]), pair[0][1], pair[0][2]))
        selected += [{'stratum': stratum, 'family': key[0], 'candidate': c}
                     for key,c in ordered[:2]]
    return selected, dict(count)


def run(name, hashes):
    path = FOLDER/(name+'.json')
    if path.exists():
        raise FileExistsError(path)
    parent_path = ROOT/'results/repair-relation-selector-v4'/(name+'.json')
    parent = json.loads(parent_path.read_text()); assert parent['complete']
    panel = parent['rows'][0]; block = panel['fit']['inner'][0]
    ri, ii = np.array(panel['train_indices']), np.array(block['fit_indices'])
    X,y = load_dataset(name); A,_ = preprocess_fit(X[ri[ii]],X[ri[ii]])
    labels = y[ri[ii]]; d = A.shape[1]; classes = np.unique(labels)
    selected, counts = tasks(parent,A,labels); reports = []; start = time.perf_counter()
    for index, item in enumerate(selected):
        c = item['candidate']; fs = c['features']; wr = np.array(c['weights'])
        embedded = np.zeros(d); embedded[fs] = wr; seen = set(); attacks = []
        for side, ck, cl, wf in zip(('original','learned'), ('certificate','family_certificate'),
                ('classwise','family_classwise'), references(c), strict=True):
            if c[cl]['feasible']:
                continue
            ci = int(np.argmax(c[cl]['bounds'])); target = classes[ci]
            key = tuple(wf), int(target)
            if key in seen:
                continue
            seen.add(key)
            seed = 20261012 + 1000*NAMES.index(name) + 10*index + ci
            fit = attack(A, labels, wf, embedded, target, seed=seed)
            Z = np.array(fit['perturbed_table'])
            assert np.max(abs(Z-A)) <= .01+1e-12 and Z.min() >= 0 and Z.max() <= 1
            before = memberships(A,labels,range(d),wf)-memberships(A,labels,fs,wr)
            losses = memberships(Z,labels,range(d),wf)-memberships(Z,labels,fs,wr)
            nominal = float(before[labels == target].mean()); actual = float(losses[labels == target].mean())
            np.testing.assert_allclose(nominal, fit['nominal_signed_class_loss'], atol=1e-10,rtol=0)
            np.testing.assert_allclose(actual, fit['observed_signed_class_loss_lower'], atol=1e-10,rtol=0)
            bound = c['certificate' if side == 'original' else 'family_certificate']['dependency_certificate']
            object_bounds = np.array(bound.get('redistributed_object_bounds', np.zeros(len(labels))))
            assert np.all(np.maximum(losses,0.) <= object_bounds+1e-9)
            assert actual <= c[cl]['bounds'][ci]+1e-9 and nominal <= .1+1e-10
            assert actual+1e-12 >= nominal
            attacks.append({'side': side, 'full_weights': wf.tolist(), 'reduced_weights': embedded.tolist(),
                'saved_class_upper_bound': c[cl]['bounds'][ci], 'independent_replayed_loss': actual,
                'independent_replayed_global_loss': float(losses.mean()), 'fit': fit})
            print(name,index,item['stratum'],side,'class',int(target),'nominal',nominal,
                  'attack',actual,'violation',fit['class_violation_witness'],flush=True)
        reports.append({**item, 'attacks': attacks,
            'class_violation_witness': any(a['fit']['class_violation_witness'] for a in attacks),
            'original_majority_feasible': c['certificate']['majority_feasible'],
            'learned_majority_feasible': c['family_certificate']['majority_feasible']})
    for rel,value in hashes.items():
        assert digest(ROOT/rel) == value
    record = {'dataset': name, 'complete': True, 'counts': counts, 'selected_candidates': len(reports),
              'candidate_class_violations': sum(r['class_violation_witness'] for r in reports),
              'candidate_unresolved': sum(not r['class_violation_witness'] for r in reports),
              'attacks': sum(len(r['attacks']) for r in reports), 'reports': reports,
              'seconds': time.perf_counter()-start, 'source_hashes': hashes,
              'new_classifier_predictions': False, 'validation_or_outer_observations_used': False,
              'goal_achieved': False, 'scope': 'classwise_loss_attack_witness_not_robust_feasibility_or_accuracy'}
    path.write_text(json.dumps(record, separators=(',',':')),encoding='utf8')
    return record


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    paths = [Path(__file__), ROOT/'scripts/classwise_shared_attack_v1.py',
             ROOT/'scripts/diagnose_repair_bottlenecks_v7.py', ROOT/'src/certified_frfs/datasets.py',
             ROOT/'src/certified_frfs/validated_pool.py', ROOT/'data/manifest.json',
             ROOT/'tests/test_classwise_shared_attack_v1.py', ROOT/'research/classwise-shared-attack-v1-protocol.md']
    paths += [ROOT/'results/repair-relation-selector-v4'/(n+'.json') for n in NAMES]
    paths += [ROOT/'data/processed'/(n+'.npz') for n in NAMES]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    freeze = FOLDER/'source-freeze.json'; assert not freeze.exists()
    freeze.write_text(json.dumps({'hashes':hashes,'created_unix':time.time()},indent=2))
    try:
        rows = [run(n,hashes) for n in NAMES]
        strata = {}
        for key in ('classwise_only','global_blocked'):
            vals = [r for row in rows for r in row['reports'] if r['stratum'] == key]
            strata[key] = {'selected':len(vals),'class_violation_witnesses':sum(r['class_violation_witness'] for r in vals),
                           'unresolved':sum(not r['class_violation_witness'] for r in vals)}
        summary = {'complete':True,'datasets':len(rows),'selected_candidates':sum(r['selected_candidates'] for r in rows),
            'attacks':sum(r['attacks'] for r in rows),'strata':strata,
            'dataset_rows':[{k:r[k] for k in ('dataset','counts','selected_candidates','candidate_class_violations',
                                            'candidate_unresolved','attacks','seconds')} for r in rows],
            'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in NAMES}, 'goal_achieved':False,
            'new_classifier_predictions':False,'robust_feasibility_established':False,
            'limits':['Local attacks are lower witnesses, never proof of safety or a global optimum.',
                      'Only two deterministic candidates per stratum in one inner training block per dataset.',
                      'Saved perturbation upper bounds are consistency inputs, not independently recomputed here.',
                      'No validation/test classification scores or improved method deployment.']}
        (FOLDER/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        print(json.dumps(summary,indent=2))
    except Exception:
        (FOLDER/('failure-'+str(time.time_ns())+'.json')).write_text(json.dumps({'traceback':traceback.format_exc(),
            'source_hashes':hashes,'goal_achieved':False},indent=2),encoding='utf8')
        raise


if __name__ == '__main__':
    main()
