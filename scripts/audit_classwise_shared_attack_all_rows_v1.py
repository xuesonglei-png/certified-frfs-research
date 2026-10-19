"""Independent saved-witness replay; does not run attacks or classifiers.

Uses its own anchor-by-coordinate enemy distances and reconstructs the fixed
cohort without importing the production runner/attack. Final tables only are
available, so intermediate trajectories are checked for metadata consistency.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT/'results/classwise-shared-attack-v1'
NAMES = ('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')
FAMILIES = ('uniform_sqrt','half_majority','learned_majority','half_local_soft','local_soft')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_memberships(A, y, fs, w):
    """Independent anchor loops; no cdist, gradients or FRS certificate code."""
    answer = np.empty(len(A))
    for i in range(len(A)):
        enemies = np.flatnonzero(y != y[i])
        distances = np.zeros(len(enemies))
        for a, weight in zip(fs, w, strict=True):
            distances += weight*np.abs(float(A[i,a])-A[enemies,a])
        answer[i] = min(1., float(min(distances)))
    return answer


def references(candidate):
    return [np.asarray(candidate[k]['dependency_certificate'].get('full_weights',candidate['weights']))
            for k in ('certificate','family_certificate')]


def key_of(family, c):
    refs = references(c)
    return family, tuple(c['features']), tuple(c['weights']), tuple(refs[0]), tuple(refs[1])


def reconstruct(parent, A, y):
    block = parent['rows'][0]['fit']['inner'][0]
    candidates = {}
    for family in FAMILIES:
        for recipe in block['families'][family]['candidates']:
            for c in recipe['trace']:
                candidates[key_of(family,c)] = c
    cached = {}; strata = {'classwise_only': [], 'global_blocked': []}; count = Counter()
    def membership(fs, w):
        key = tuple(fs), tuple(w)
        if key not in cached:
            cached[key] = scalar_memberships(A,y,fs,w)
        return cached[key]
    for key, c in candidates.items():
        glob = any(c[k]['dependency_certificate']['upper_bound'] > .1
                   for k in ('certificate','family_certificate'))
        cls = not (c['classwise']['feasible'] and c['family_classwise']['feasible'])
        count['unique_training_candidates'] += 1
        if not (glob or cls):
            continue
        count['dependency_blocked'] += 1
        reduced = membership(c['features'],c['weights'])
        losses = [membership(range(A.shape[1]),ref)-reduced for ref in references(c)]
        violates = any(float(loss.mean()) > .1+1e-10 or
                       any(float(loss[y==q].mean()) > .1+1e-10 for q in np.unique(y))
                       for loss in losses)
        if violates:
            count['already_nominal_violating_excluded'] += 1
            continue
        stratum = 'global_blocked' if glob else 'classwise_only'
        strata[stratum].append((key,c)); count['eligible_'+stratum] += 1
    selection = []
    for stratum, values in strata.items():
        ordered = sorted(values, key=lambda kv: (len(kv[1]['features']),FAMILIES.index(kv[0][0]),kv[0][1],kv[0][2]))
        selection.extend({'stratum':stratum,'family':key[0],'candidate':c} for key,c in ordered[:2])
    return selection, dict(count)


def audit(name, hashes, artifact_hashes):
    path = FOLDER/(name+'.json'); saved = json.loads(path.read_text(encoding='utf8'))
    assert saved['complete'] and digest(path) == artifact_hashes[name]
    assert saved['source_hashes'] == hashes
    assert saved['new_classifier_predictions'] is False and saved['validation_or_outer_observations_used'] is False
    parent = json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
    panel = parent['rows'][0]; block = panel['fit']['inner'][0]
    indices = np.asarray(panel['train_indices'])[np.asarray(block['fit_indices'])]
    X,y = load_dataset(name); A,_ = preprocess_fit(X[indices],X[indices]); y = y[indices]
    selected, counts = reconstruct(parent,A,y)
    assert counts == saved['counts'] and len(selected) == saved['selected_candidates'] == len(saved['reports'])
    classes = np.unique(y); checked = Counter(); details = []
    for ci,(expected,report) in enumerate(zip(selected,saved['reports'],strict=True)):
        for key in ('stratum','family','candidate'):
            assert report[key] == expected[key]
        c = report['candidate']; d = A.shape[1]
        wr = np.zeros(d); wr[c['features']] = c['weights']; nominal_red = scalar_memberships(A,y,range(d),wr)
        expected_attacks = []; seen = set()
        for side,cl,ck,wf in zip(('original','learned'),('classwise','family_classwise'),
                               ('certificate','family_certificate'),references(c),strict=True):
            if c[cl]['feasible']:
                continue
            # np.argmax means the first class in ascending class order wins ties.
            target_i = int(np.argmax(c[cl]['bounds'])); target = classes[target_i]
            key = tuple(wf),target.item()
            if key not in seen:
                expected_attacks.append((side,cl,ck,wf,target_i,target)); seen.add(key)
        assert len(expected_attacks) == len(report['attacks'])
        candidate_violates = False
        for expected_attack,record in zip(expected_attacks,report['attacks'],strict=True):
            side,cl,ck,wf,target_i,target = expected_attack; fit = record['fit']
            assert record['side'] == side and fit['target_class'] == target
            np.testing.assert_array_equal(record['full_weights'],wf)
            np.testing.assert_array_equal(record['reduced_weights'],wr)
            assert fit['eps'] == .01 and fit['starts'] == 4 and fit['steps'] == 16
            assert fit['seed'] == 20261012+1000*NAMES.index(name)+10*ci+target_i
            assert len(fit['runs']) == 4
            Z = np.asarray(fit['perturbed_table']); assert Z.shape == A.shape and np.isfinite(Z).all()
            assert np.all(Z >= np.maximum(0.,A-.01)-1e-12)
            assert np.all(Z <= np.minimum(1.,A+.01)+1e-12)
            before = scalar_memberships(A,y,range(d),wf)-nominal_red
            losses = scalar_memberships(Z,y,range(d),wf)-scalar_memberships(Z,y,range(d),wr)
            nominal = float(before[y==target].mean()); actual = float(losses[y==target].mean())
            glob = float(losses.mean()); upper = c[cl]['bounds'][target_i]
            np.testing.assert_allclose(nominal,fit['nominal_signed_class_loss'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(actual,[fit['observed_signed_class_loss_lower'],record['independent_replayed_loss']],rtol=0,atol=1e-10)
            np.testing.assert_allclose(glob,record['independent_replayed_global_loss'],rtol=0,atol=1e-10)
            assert record['saved_class_upper_bound'] == upper
            assert actual <= upper+1e-9 and nominal <= .1+1e-10 and actual+1e-12 >= nominal
            object_upper = c[ck]['dependency_certificate'].get('redistributed_object_bounds',[0.]*len(y))
            assert np.all(np.maximum(0.,losses) <= np.asarray(object_upper)+1e-9)
            violation = actual > .1+1e-10
            assert fit['class_violation_witness'] == violation
            candidate_violates |= violation
            terminals = [nominal]
            for start,run in enumerate(fit['runs']):
                trace,scales = run['loss_trace'],run['accepted_scales']
                assert run['start'] == start and 1 <= len(trace) <= 17 and len(scales) == len(trace)-1
                assert np.isfinite(trace).all() and all(v in (1.,.5,.25,.125,.0625) for v in scales)
                assert all(b > a+1e-12 for a,b in zip(trace[:-1],trace[1:]))
                if start == 0:
                    np.testing.assert_allclose(trace[0],nominal,rtol=0,atol=1e-10)
                terminals.append(trace[-1]); checked['run_metadata_checks'] += 1
            np.testing.assert_allclose(max(terminals),actual,rtol=0,atol=1e-10)
            checked['shared_tables'] += 1; checked['training_object_loss_replays'] += len(y)
            checked['violating_targets'] += violation
            details.append({'candidate':ci,'stratum':report['stratum'],'side':side,
                'target_class':target.item(),'nominal_signed_class_loss':nominal,
                'replayed_signed_class_loss':actual,'replayed_signed_global_loss':glob,
                'witness_violation':violation,'rows':len(y)})
        assert report['class_violation_witness'] == candidate_violates
        assert report['original_majority_feasible'] == c['certificate']['majority_feasible']
        assert report['learned_majority_feasible'] == c['family_certificate']['majority_feasible']
        checked['selected_candidates'] += 1
        checked['violating_candidates'] += candidate_violates
        checked[report['stratum']+'_selected'] += 1
        checked[report['stratum']+'_violating'] += candidate_violates
    assert checked['shared_tables'] == saved['attacks']
    assert checked['violating_candidates'] == saved['candidate_class_violations']
    assert checked['selected_candidates']-checked['violating_candidates'] == saved['candidate_unresolved']
    return {'dataset':name,'complete':True,'checks':dict(checked),'details':details,
            'artifact_sha256':digest(path),'new_classifier_predictions':False}


def main():
    threadpool_limits(1); out = FOLDER/'all-rows-independent-audit-v1.json'; assert not out.exists()
    freeze = json.loads((FOLDER/'source-freeze.json').read_text()); hashes = freeze['hashes']
    summary = json.loads((FOLDER/'summary.json').read_text()); assert summary['complete']
    for rel,sha in hashes.items():
        assert digest(ROOT/rel) == sha,rel
    start = time.perf_counter(); rows = [audit(n,hashes,summary['artifact_hashes']) for n in NAMES]
    tally = Counter()
    for row in rows:
        tally.update(row['checks'])
    assert tally['selected_candidates'] == 21 and tally['shared_tables'] == 30 and tally['violating_candidates'] == 9
    result = {'complete':True,'datasets':6,'checks':dict(tally),'dataset_rows':rows,
        'seconds':time.perf_counter()-start,'auditor_sha256':digest(__file__),
        'new_classifier_predictions':False,'new_attacks_or_optimizer_runs':False,
        'robust_feasibility_established':False,'limits':[
            'Final shared tables and fixed selection independently replayed; intermediate tables were not saved.',
            'Run metadata checks do not independently verify intermediate gradient trajectories.',
            'Legacy object/class upper bounds are consistency inputs; no independent derivation here.',
            'Preprocessing is reused. Floating point tolerance is not directed-rounding verification.',
            'A feasible counterexample refutes safety of its exact metric, but absence does not prove safety.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps({k:v for k,v in result.items() if k!='dataset_rows'},indent=2))


if __name__ == '__main__':
    main()
