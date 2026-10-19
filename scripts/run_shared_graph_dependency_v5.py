"""Freeze and run the complete fixed training-only graph-bound screen."""
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
from shared_graph_dependency_v5 import class_upper, memberships
from signed_classwise_dependency import signed_classwise_dependency

NAMES = ('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')
OUT = ROOT/'results/shared-graph-dependency-v5'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(hashes):
    for rel, expected in hashes.items():
        if digest(ROOT/rel) != expected:
            raise RuntimeError('frozen source changed: '+rel)


def run(name, hashes):
    parent = json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text())
    attacks = json.loads((ROOT/'results/classwise-shared-attack-v1'/(name+'.json')).read_text())
    panel = parent['rows'][0]; inner = panel['fit']['inner'][0]
    ids = np.asarray(panel['train_indices'])[np.asarray(inner['fit_indices'])]
    X,y = load_dataset(name); A,_ = preprocess_fit(X[ids],X[ids]); y = y[ids]
    classes = np.unique(y); rows = []; start = time.perf_counter(); cache = {}
    folder = OUT/name; folder.mkdir(exist_ok=True)
    for ci, source in enumerate(attacks['reports']):
        path = folder/(str(ci)+'.json')
        if path.exists():
            saved = json.loads(path.read_text()); assert saved['source_hashes'] == hashes
            rows.append(saved); continue
        c = source['candidate']; full_red = np.zeros(A.shape[1]); full_red[c['features']] = c['weights']
        guards = []
        for side,key in (('original','certificate'),('family','family_certificate')):
            old = c[key]['dependency_certificate']; wf = np.asarray(old.get('full_weights',c['weights']))
            identity = tuple(wf),tuple(c['features']),tuple(c['weights'])
            if identity in cache:
                guard = {**cache[identity], 'side':side, 'reused_identical_metric':True}
                guards.append(guard); continue
            signed = signed_classwise_dependency(A,y,c['features'],wf,c['weights'],base_dependency=old)
            objects = np.asarray(signed['signed_object_report']['signed_object_upper_bounds'])
            bounds = signed['class_certificate']['upper_bounds']; reports = []; improved = []; positive_reports = []; positive_bounds = []
            lo, hi = np.maximum(0.,A-.01), np.minimum(1.,A+.01)
            enemy_cuts = {}
            for obj in signed['signed_object_report']['objects']:
                i = obj['object']; possible = obj['possible_enemies']
                lower = np.maximum(0.,np.maximum(lo[i]-hi[possible],lo[possible]-hi[i])) @ full_red
                enemy_cuts[i] = {j: min(float(1.-l),float(u)) for j,l,u in zip(
                    possible,lower,obj['candidate_signed_upper_bounds'],strict=True)}
            for label, bound in zip(classes,bounds,strict=True):
                if bound <= .1:
                    reports.append({'target_class':label.item(),'skipped_existing_feasible':True,
                                    'old_upper_bound':bound,'new_upper_bound':bound})
                    improved.append(bound); continue
                solve = class_upper(A,y,wf,full_red,label,seconds=8.,object_upper=objects,enemy_upper=enemy_cuts,positive=False)
                upper = min(bound,solve['upper_bound']); improved.append(upper)
                for previous in source['attacks']:
                    fit = previous['fit']
                    if np.array_equal(previous['full_weights'],wf) and fit['target_class']==label:
                        assert fit['observed_signed_class_loss_lower'] <= upper+1e-7
                reports.append({'target_class':label.item(),'skipped_existing_feasible':False,
                    'old_upper_bound':bound,'new_upper_bound':upper,'solve':solve})
                print(name,ci,side,'class',label.item(),'old',bound,'new',upper,
                      'status',solve['solver_status'],'secs',round(solve['seconds'],2),flush=True)
            positive_old = c['classwise' if side == 'original' else 'family_classwise']['bounds']
            for label,bound in zip(classes,positive_old,strict=True):
                if bound <= .1:
                    positive_reports.append({'target_class':label.item(),'skipped_existing_feasible':True,
                        'old_upper_bound':bound,'new_upper_bound':bound})
                    positive_bounds.append(bound); continue
                solve = class_upper(A,y,wf,full_red,label,seconds=8.,object_upper=objects,
                                    enemy_upper=enemy_cuts,positive=True)
                upper = min(bound,solve['upper_bound']);positive_bounds.append(upper)
                for previous in source['attacks']:
                    if np.array_equal(previous['full_weights'],wf) and previous['fit']['target_class']==label:
                        Z = np.asarray(previous['fit']['perturbed_table'])
                        actual = np.maximum(0., memberships(Z,y,wf)-memberships(Z,y,full_red))[y==label].mean()
                        assert actual <= upper+1e-7
                positive_reports.append({'target_class':label.item(),'skipped_existing_feasible':False,
                    'old_upper_bound':bound,'new_upper_bound':upper,'solve':solve})
                print(name,ci,side,'POSITIVE class',label.item(),'old',bound,'new',upper,
                      'status',solve['solver_status'],'secs',round(solve['seconds'],2),flush=True)
            guard = {'side':side,'full_weights':wf.tolist(),'signed_class_bounds':bounds,
                'new_class_bounds':improved,'signed_report':signed,'classes':reports,
                'positive_classes':positive_reports,'positive_class_bounds':positive_bounds,
                'positive_feasible':all(b<=.1 for b in positive_bounds),
                'signed_feasible':all(b<=.1 for b in bounds),'new_feasible':all(b<=.1 for b in improved),
                'reused_identical_metric':False}
            cache[identity]=guard; guards.append(guard)
        legacy = bool(c['joint_feasible']); signed_pass = all(g['signed_feasible'] for g in guards)
        new_pass = all(g['new_feasible'] for g in guards)
        positive_pass = all(g['positive_feasible'] for g in guards)
        majority = bool(c['certificate']['majority_feasible'] and c['family_certificate']['majority_feasible'])
        assert not source['class_violation_witness'] or not new_pass
        check(hashes)
        row = {'dataset':name,'candidate_index':ci,'complete':True,'family':source['family'],
            'stratum':source['stratum'],'features':c['features'],'weights':c['weights'],'guards':guards,
            'known_violation':source['class_violation_witness'],'legacy_joint_feasible':legacy,
            'signed_dependency_feasible':signed_pass,'new_dependency_feasible':new_pass,
            'original_majority_feasible':majority,'signed_joint_feasible':signed_pass and majority,
            'new_joint_feasible':new_pass and majority,'source_hashes':hashes,
            'positive_dependency_feasible':positive_pass,'positive_joint_feasible':positive_pass and majority,
            'new_classifier_predictions':False,'validation_or_outer_observations_used':False}
        path.write_text(json.dumps(row,separators=(',',':')),encoding='utf8');rows.append(row)
    result = {'dataset':name,'complete':True,'rows':rows,'seconds':time.perf_counter()-start,
              'source_hashes':hashes,'new_classifier_predictions':False}
    (OUT/(name+'.json')).write_text(json.dumps(result,separators=(',',':')),encoding='utf8')
    return result


def main():
    threadpool_limits(1); OUT.mkdir(exist_ok=True)
    paths = [Path(__file__),ROOT/'scripts/shared_graph_dependency_v5.py',
        ROOT/'scripts/signed_classwise_dependency.py',ROOT/'scripts/classwise_star_dependency.py',
        ROOT/'tests/test_shared_graph_dependency_v5.py',ROOT/'research/shared-graph-dependency-v5-protocol.md',
        ROOT/'data/manifest.json']
    paths += list((ROOT/'src/certified_frfs').glob('*.py'))
    for name in NAMES:
        paths += [ROOT/'results/repair-relation-selector-v4'/(name+'.json'),
                  ROOT/'results/classwise-shared-attack-v1'/(name+'.json'),ROOT/'data/processed'/(name+'.npz')]
    frozen = OUT/'source-freeze.json'
    if frozen.exists():
        hashes = json.loads(frozen.read_text())['hashes']; check(hashes)
    else:
        hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in paths}
        frozen.write_text(json.dumps({'hashes':hashes,'created_unix':time.time()},indent=2),encoding='utf8')
    records = [run(name,hashes) for name in NAMES]; rows = [r for d in records for r in d['rows']]
    new_dep = [r for r in rows if r['new_dependency_feasible'] and not r['signed_dependency_feasible']]
    new_joint = [r for r in rows if r['new_joint_feasible'] and not r['signed_joint_feasible']]
    positive_dep = [r for r in rows if r['positive_dependency_feasible'] and not r['legacy_joint_feasible']]
    positive_joint = [r for r in rows if r['positive_joint_feasible'] and not r['legacy_joint_feasible']]
    dataset_counts = {name:dict(Counter({
        'candidates':sum(r['dataset']==name for r in rows),
        'known_violations':sum(r['dataset']==name and r['known_violation'] for r in rows),
        'new_dependency_admissions':sum(r['dataset']==name for r in new_dep),
        'new_joint_admissions':sum(r['dataset']==name for r in new_joint),
        'positive_dependency_admissions':sum(r['dataset']==name for r in positive_dep),
        'positive_joint_admissions':sum(r['dataset']==name for r in positive_joint)})) for name in NAMES}
    solves = [q['solve'] for r in rows for g in r['guards'] if not g['reused_identical_metric']
              for q in g['classes']+g['positive_classes'] if not q['skipped_existing_feasible']]
    summary = {'complete':True,'datasets':6,'candidates':len(rows),'known_violations':sum(r['known_violation'] for r in rows),
        'new_dependency_admissions':len(new_dep),'new_joint_admissions':len(new_joint),
        'positive_dependency_admissions':len(positive_dep),'positive_joint_admissions':len(positive_joint),
        'positive_resource_gate':len(positive_dep)>=3 and len({r['dataset'] for r in positive_dep})>=2 and len(positive_joint)>=1,
        'resource_gate':len(new_dep)>=3 and len({r['dataset'] for r in new_dep})>=2 and len(new_joint)>=1,
        'dataset_counts':dataset_counts,'solver_calls':len(solves),
        'solver_status_counts':dict(Counter(str(q['solver_status']) for q in solves)),
        'mean_class_tightening':float(np.mean([q['old_upper_bound']-q['new_upper_bound']
            for r in rows for g in r['guards'] if not g['reused_identical_metric'] for q in g['classes']])),
        'seconds':sum(r['seconds'] for r in records),
        'artifact_hashes':{n:digest(OUT/(n+'.json')) for n in NAMES},
        'new_classifier_predictions':False,'goal_achieved':False,
        'classification_superiority_established':False,
        'limits':['Finite floating solver dual evidence, not formal directed rounding.',
                  'Fixed 21 training-only diagnostic candidates; no new deployment or prediction evidence.',
                  'Some full membership witnesses and absolute-value equalities are relaxed.',
                  'A relaxed incumbent objective is not an attained attack.']}
    check(hashes);(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
