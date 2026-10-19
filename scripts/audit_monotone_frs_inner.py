"""Independent cell/LP/guard/prediction audits, preserving inherited choices."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_monotone_frs_inner import ROOT, SOURCE, PREVIOUS, FOLDER, METHODS, digest
from monotone_frs_hinge_stage import GRID, OLD_GRID
from audit_frs_one_sided_fit_stored import check_one_sided_fit
from independent_frs_one_sided_stored import reconstruct_cell, scalar_cell_evaluation
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
        audit_path = folder/'complete-audit.json'
        if audit_path.exists():
            report = json.loads(audit_path.read_text())
            assert report['passed'] and report['artifact_hashes'][name] == digest(folder/(name+'.json'))
        else:
            report = json.loads((folder/(name+'.smoke-audit.json')).read_text())
            assert report['passed'] and report['sha256'] == digest(folder/(name+'.json'))
    assert saved['reference_indices'] == source['reference_indices'] == prior['reference_indices']
    X, y = load_dataset(name); ri = np.asarray(saved['reference_indices']); raw, labels = X[ri], y[ri]
    for fi, (b, old, pb) in enumerate(zip(saved['inner'], source['inner'], prior['inner'], strict=True)):
        ca, cb = np.asarray(b['reference_indices']), np.asarray(b['validation_indices'])
        assert not set(ca)&set(cb)
        assert b['reference_indices'] == old['reference_indices'] == pb['reference_indices']
        assert b['validation_indices'] == old['validation_indices'] == pb['validation_indices']
        A, B = preprocess_fit(raw[ca], raw[cb]); stage = b['stage']
        old_check_stage(A, labels[ca], {'learners': [], 'certificate_checks': stage['certificate_checks']}, counts)
        inherited = len(pb['stage']['learners'])
        equivalent(stage['learners'][:inherited], pb['stage']['learners'])
        equivalent(stage['cell_certificate_checks'], pb['stage']['cell_certificate_checks'])
        counts['previous_audited_fits_reused'] += inherited
        seen = set()
        for learner in stage['learners'][inherited:]:
            key = (tuple(learner['features']), tuple(learner['anchor']), learner['certified_arm'])
            assert key not in seen; seen.add(key); assert learner['family'] == 'one_sided_cell_constrained'
            fit = learner['fit']; probe = learner['original_cell_probe']
            assert probe['full_weights'] == (np.ones(A.shape[1])/np.sqrt(A.shape[1])).tolist()
            if fit['reference_scope'] == 'original_full' and not fit['cell_unavailable']:
                equivalent(probe, fit['class_cell'])
            else:
                reconstruct_cell(A, labels[ca], probe, counts)
            assert fit['reference_scope'] == ('original_full' if probe['anchor_strictly_feasible'] or learner['certified_arm'] else 'own_anchor')
            if fit['cell_unavailable']:
                assert learner['certified_arm'] and not probe['anchor_strictly_feasible']
                assert fit['status'] == 'original_full_cell_unavailable_anchor_retained'
                equivalent(learner['anchor'], learner['target']); equivalent(probe, fit['class_cell'])
                counts['unavailable_proposed_cells'] += 1
            else:
                check_one_sided_fit(A, labels[ca], learner, counts)
                assert fit['solver_primal_feasibility_tolerance'] == fit['solver_dual_feasibility_tolerance'] == 1e-9
        for gi, guard in enumerate(stage['monotone_certificate_checks']):
            assert guard['id'] == gi
            lid = guard['learner_id']; learner = stage['learners'][lid]; fit = learner['fit']
            assert learner['certified_arm'] and fit['reference_scope'] == 'original_full'
            old_guard = stage['certificate_checks'][guard['old_guard_id']]
            equivalent(guard['weights'], old_guard['weights']); assert guard['features'] == old_guard['features'] == learner['features']
            value = scalar_cell_evaluation(fit['class_cell'], guard['weights'])
            equivalent(value, guard['cell_evaluation'])
            accepted = bool(not fit['cell_unavailable'] and value['feasible'] and old_guard['certificate']['majority_feasible'])
            assert guard['cell_accepted'] == accepted and guard['old_accepted'] == old_guard['accepted']
            assert guard['accepted'] == bool(old_guard['accepted'] or accepted)
            counts['independent_cell_guard_replays'] += 1
            counts['new_cell_guard_admissions'] += bool(accepted and not old_guard['accepted'])
        used = set()
        for m in METHODS:
            anchor = old['options'][m][source['selection']['selected'][m]]['metric']
            base = {k: anchor[k] for k in ('features', 'weights')} | {'origin': 'frozen_atomic_v2_selected_metric'}
            equivalent(base, b['bases'][m]); assert len(b['options'][m]) == len(GRID)
            for ti, op in enumerate(b['options'][m]):
                assert op['target_index'] == ti; family, t = GRID[ti]
                metric = op['metric']; learner = stage['learners'][metric['learner_id']]
                assert metric['features'] == base['features'] == learner['features']
                assert metric['target_family'] == learner['family'] == family and metric['interpolation'] == t
                equivalent(base['weights'], learner['anchor'])
                request = (1-t)*np.asarray(base['weights'])+t*np.asarray(learner['target'])
                equivalent(request.tolist(), metric['requested_weights'])
                if ti < len(OLD_GRID):
                    equivalent({k:v for k,v in metric.items() if not k.startswith('monotone_')}, pb['options'][m][ti]['metric'])
                    assert metric['monotone_guard_id'] is metric['monotone_deployed_guard_id'] is None
                if m == 'proposed':
                    guards = stage['cell_certificate_checks'] if 14 <= ti < len(OLD_GRID) else stage['certificate_checks'] if ti < 14 else stage['monotone_certificate_checks']
                    gkey = 'cell_guard_id' if 14 <= ti < len(OLD_GRID) else 'guard_id' if ti < 14 else 'monotone_guard_id'
                    dkey = 'cell_deployed_guard_id' if 14 <= ti < len(OLD_GRID) else 'deployed_guard_id' if ti < 14 else 'monotone_deployed_guard_id'
                    guard = guards[metric[gkey]]; equivalent(guard['weights'], request.tolist())
                    assert metric['guard_fallback'] == bool(not guard['accepted'])
                    deployed = guards[metric[dkey]]; assert deployed['accepted']
                    actual = base['weights'] if metric['guard_fallback'] else request.tolist()
                    equivalent(actual, deployed['weights']); equivalent(actual, metric['weights'])
                    if ti >= len(OLD_GRID):
                        assert learner['certified_arm'] and guard['learner_id'] == metric['learner_id']
                        assert metric['guard_id'] == guard['old_guard_id'] and metric['deployed_guard_id'] == deployed['old_guard_id']
                else:
                    assert metric['guard_id'] is metric['deployed_guard_id'] is metric['monotone_guard_id'] is metric['monotone_deployed_guard_id'] is None
                    assert not metric['guard_fallback']; equivalent(request.tolist(), metric['weights'])
                    if ti >= len(OLD_GRID):
                        assert not learner['certified_arm']
                rid = op['record_id']; r = b['records'][rid]; used.add(rid)
                equivalent({k: metric[k] for k in ('features', 'weights')}, {k: r['metric'][k] for k in ('features', 'weights')})
                if ti < len(OLD_GRID):
                    equivalent(r['evaluation'], pb['records'][pb['options'][m][ti]['record_id']]['evaluation'])
        assert used == set(range(len(b['records'])))
        for r in b['records']:
            equivalent(independent_evaluate(A, labels[ca], B, labels[cb], r['metric']), r['evaluation'])
            counts['independent_inner_predictions'] += 1
        counts['inner_folds'] += 1
        print(name, 'inner', fi+1, 'independent monotone-FRS/LP/guard/vote replayed', flush=True)
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
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed': True, **r}, indent=2), encoding='utf-8'); return
    summary = json.loads((FOLDER/'complete.json').read_text()); counts = Counter(); reports = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); reports.append(r); counts.update(r['counts'])
    assert counts['datasets'] == 15 and counts['inner_folds'] == summary['inner_folds'] == 44
    assert counts['target_optimizer_failures'] == summary['target_optimizer_failures']
    rows = []
    for sr in summary['rows']:
        name = sr['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text()); prior = json.loads((PREVIOUS/(name+'.json')).read_text())
        chosen = lambda s, m: s['selection'][m]['options'][s['selection'][m]['selected']]['mean_BA']
        rows.append({'dataset': name, 'anchor_gain_pp': {m: saved['selection'][m]['selected_inner_gain_pp'] for m in METHODS},
            'vs_previous_DC_pp': {m: 100*(chosen(saved, m)-chosen(prior, m)) for m in METHODS},
            'selected_inner_BA': {m: chosen(saved, m) for m in METHODS}})
    equivalent(rows, summary['rows'])
    for m in METHODS:
        equivalent(float(np.mean([r['anchor_gain_pp'][m] for r in rows])), summary['mean_anchor_gain_pp'][m])
        equivalent(float(np.mean([r['vs_previous_DC_pp'][m] for r in rows])), summary['mean_vs_previous_DC_pp'][m])
    wins = sum(r['vs_previous_DC_pp']['proposed'] > 1e-10 for r in rows)
    assert wins == summary['improved_vs_previous_DC_datasets']
    differences = {m: float(np.mean([100*(r['selected_inner_BA']['proposed']-r['selected_inner_BA'][m]) for r in rows]))
        for m in ('full_refined', 'fgs_refined', 'sfrip_refined')}
    equivalent(differences, summary['mean_proposed_minus_controls_pp'])
    assert summary['pilot_gate_passed'] == (summary['mean_vs_previous_DC_pp']['proposed'] >= .5 and wins >= 6
        and counts['target_optimizer_failures'] == 0 and min(differences.values()) > 0)
    hashes = {r['dataset']: r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    out = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
           'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False, 'new_outer_evaluation_performed': False,
           'scope': 'independent_FRS_cells_DC_faces_LP_duals_guards_inner_predictions_not_confirmatory_superiority'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8'); print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
