"""Independent DC/LP/guard/vote/selection replay, including failed fits."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_third_enemy_inner import ROOT, SOURCE, PREVIOUS, FOLDER, METHODS, digest
from third_enemy_hinge_stage import GRID
from audit_third_enemy_fit import check_fit
from audit_predictive_target_knn_pilot import check_stage as old_check_stage
from independent_geometric_evaluation import independent_evaluate
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1); path = FOLDER/(name+'.json'); saved = json.loads(path.read_text()); counts = Counter(datasets=1)
    assert saved['complete'] and not saved['new_outer_evaluation_performed']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    source = json.loads((SOURCE/(name+'.json')).read_text()); prior = json.loads((PREVIOUS/(name+'.json')).read_text())
    for folder in (SOURCE, PREVIOUS):
        report = json.loads((folder/'complete-audit.json').read_text())
        assert report['passed'] and report['artifact_hashes'][name] == digest(folder/(name+'.json'))
    assert saved['reference_indices'] == source['reference_indices'] == prior['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for fi, (b, old) in enumerate(zip(saved['inner'], source['inner'], strict=True)):
        ca, cb = np.asarray(b['reference_indices']), np.asarray(b['validation_indices'])
        assert not set(ca)&set(cb)
        assert b['reference_indices'] == old['reference_indices'] and b['validation_indices'] == old['validation_indices']
        A, B = preprocess_fit(raw[ca], raw[cb]); stage = b['stage']
        old_check_stage(A, labels[ca], {'learners': [], 'certificate_checks': stage['certificate_checks']}, counts)
        seen = set()
        for learner in stage['learners']:
            key = (tuple(learner['features']), tuple(learner['anchor']), learner['family']); assert key not in seen; seen.add(key)
            assert learner['family'] in ('third_enemy_unconstrained', 'third_enemy_constrained')
            assert learner['fit']['constrained'] == (learner['family'] == 'third_enemy_constrained')
            check_fit(A, labels[ca], learner, counts)
        used = set()
        for m in METHODS:
            anchor = old['options'][m][source['selection']['selected'][m]]['metric']
            base = {k: anchor[k] for k in ('features', 'weights')} | {'origin': 'frozen_atomic_v2_selected_metric'}
            equivalent(base, b['bases'][m]); assert len(b['options'][m]) == len(GRID)
            for ti, op in enumerate(b['options'][m]):
                assert op['target_index'] == ti
                family, t = GRID[ti]; metric = op['metric']; learner = stage['learners'][metric['learner_id']]
                assert metric['features'] == base['features'] == learner['features']
                assert metric['target_family'] == learner['family'] == family and metric['interpolation'] == t
                equivalent(base['weights'], learner['anchor'])
                request = (1-t)*np.asarray(base['weights'])+t*np.asarray(learner['target'])
                equivalent(request.tolist(), metric['requested_weights'])
                if m == 'proposed':
                    guard = stage['certificate_checks'][metric['guard_id']]; equivalent(guard['weights'], request.tolist())
                    assert metric['guard_fallback'] == (not guard['accepted'])
                    deployed = stage['certificate_checks'][metric['deployed_guard_id']]; assert deployed['accepted']
                    actual = base['weights'] if metric['guard_fallback'] else request.tolist()
                    equivalent(actual, deployed['weights']); equivalent(actual, metric['weights'])
                else:
                    assert metric['guard_id'] is metric['deployed_guard_id'] is None and not metric['guard_fallback']
                    equivalent(request.tolist(), metric['weights'])
                rid = op['record_id']; r = b['records'][rid]; used.add(rid)
                equivalent({k: metric[k] for k in ('features', 'weights')}, {k: r['metric'][k] for k in ('features', 'weights')})
        assert used == set(range(len(b['records'])))
        for r in b['records']:
            equivalent(independent_evaluate(A, labels[ca], B, labels[cb], r['metric']), r['evaluation'])
            counts['independent_inner_predictions'] += 1
        counts['inner_folds'] += 1
        print(name, 'inner', fi+1, 'independent DC faces/primal-dual/guard/vote replayed', flush=True)
    for m in METHODS:
        rows = []
        for ti in range(len(GRID)):
            values = [b['records'][b['options'][m][ti]['record_id']]['evaluation']['metrics'] for b in saved['inner']]
            rows.append({'target_index': ti, 'mean_BA': float(np.mean([r['clean_BA'] for r in values])),
                         'mean_logloss': float(np.mean([r['balanced_logloss'] for r in values]))})
        chosen = min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_BA'], 12), round(rows[i]['mean_logloss'], 12), i))
        equivalent(saved['selection'][m], {'options': rows, 'selected': chosen,
                   'selected_inner_gain_pp': 100*(rows[chosen]['mean_BA']-rows[0]['mean_BA'])})
    return {'dataset': name, 'counts': dict(counts), 'sha256': digest(path), 'selection': saved['selection']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args()
    if args.dataset:
        r = audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **r}, indent=2), encoding='utf-8')
        return
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); reports = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    assert counts['datasets'] == 15 and counts['inner_folds'] == summary['inner_folds'] == 44
    assert counts['target_optimizer_failures'] == summary['target_optimizer_failures']
    rows = []
    for r in sorted(reports, key=lambda r: r['dataset']):
        prior = json.loads((PREVIOUS/(r['dataset']+'.json')).read_text())
        rows.append({'dataset': r['dataset'], 'anchor_gain_pp': {m: r['selection'][m]['selected_inner_gain_pp'] for m in METHODS},
                     'vs_convex_pp': {m: 100*(r['selection'][m]['options'][r['selection'][m]['selected']]['mean_BA']
                                             -prior['selection'][m]['options'][prior['selection'][m]['selected']]['mean_BA']) for m in METHODS}})
    equivalent(rows, summary['rows'])
    for m in METHODS:
        equivalent(float(np.mean([r['anchor_gain_pp'][m] for r in rows])), summary['mean_anchor_gain_pp'][m])
        equivalent(float(np.mean([r['vs_convex_pp'][m] for r in rows])), summary['mean_vs_convex_pp'][m])
    wins = sum(r['vs_convex_pp']['proposed'] > 1e-10 for r in rows)
    assert wins == summary['improved_vs_convex_datasets']
    assert summary['pilot_gate_passed'] == (summary['mean_vs_convex_pp']['proposed'] >= .5 and wins >= 6 and summary['target_optimizer_failures'] == 0)
    hashes = {r['dataset']: r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    out = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
           'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False, 'new_outer_evaluation_performed': False,
           'scope': 'independent_DC_training_bounds_faces_LP_duals_guards_inner_scores_and_choices_not_prediction_superiority'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
